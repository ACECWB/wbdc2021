# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com
    songwei, magic_24k@163.com

Reference:
    [1] [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
"""
import torch
import torch.nn as nn

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN, PredictionLayer, CIN, concat_fun, InteractingLayer
import pandas as pd

class MMOELayer(nn.Module):
    """
    The Multi-gate Mixture-of-Experts layer in MMOE model
      Input shape
        - 2D tensor with shape: ``(batch_size,units)``.

      Output shape
        - A list with **num_tasks** elements, which is a 2D tensor with shape: ``(batch_size, output_dim)`` .

      Arguments
        - **input_dim** : Positive integer, dimensionality of input features.
        - **num_tasks**: integer, the number of tasks, equal to the number of outputs.
        - **num_experts**: integer, the number of experts.
        - **output_dim**: integer, the dimension of each output of MMOELayer.

    References
      - [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
    """

    def __init__(self, input_dim, num_tasks, num_experts, output_dim):
        super(MMOELayer, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.expert_network = nn.Linear(self.input_dim, self.num_experts * self.output_dim, bias=True)
        self.gating_networks = nn.ModuleList(
            [nn.Linear(self.input_dim, self.num_experts, bias=False) for _ in range(self.num_tasks)])
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)

    def forward(self, inputs):
        outputs = []
        expert_out = self.expert_network(inputs)
        expert_out = expert_out.reshape([-1, self.output_dim, self.num_experts])
        for i in range(self.num_tasks):
            gate_out = self.gating_networks[i](inputs)
            gate_out = gate_out.softmax(1).unsqueeze(-1)
            output = torch.bmm(expert_out, gate_out).squeeze()
            outputs.append(output)
        return outputs


class MMOE(BaseModel):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param device: str, ``"cpu"`` or ``"cuda:0"``

    :return: A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, task_dnn_units=None, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, device='cpu', gpus=[0, 1]):
        
        super(MMOE, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   l2_reg_embedding=l2_reg_embedding, seed=seed, device=device)
        if num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(tasks) != num_tasks:
            raise ValueError("num_tasks must be equal to the length of tasks")
        for task in tasks:
            if task not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task))

        self.tasks = tasks
        self.task_dnn_units = task_dnn_units
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)
        self.mmoe_layer = MMOELayer(dnn_hidden_units[-1]+1000+400+400, num_tasks, num_experts, expert_dim)
        if task_dnn_units is not None:
            # the last layer of task_dnn should be expert_dim
            self.task_dnn = nn.ModuleList([DNN(expert_dim, task_dnn_units+(expert_dim,)) for _ in range(num_tasks)])
        self.tower_network = nn.ModuleList([nn.Linear(expert_dim, 1, bias=False) for _ in range(num_tasks)])
        self.out = nn.ModuleList([PredictionLayer(task) for task in self.tasks])
        self.to(device)
        
        # ??????cin
        cin_layer_size=(256, 128, 64)
        self.cin_layer_size=cin_layer_size
        cin_split_half=True
        cin_activation='relu'
        l2_reg_cin=0
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        if self.use_cin:
            field_num = len(self.embedding_dict)
            if cin_split_half == True:
                self.featuremap_num = sum(
                    cin_layer_size[:-1]) // 2 + cin_layer_size[-1]
            else:
                self.featuremap_num = sum(cin_layer_size)
            self.cin = CIN(field_num, cin_layer_size,
                           cin_activation, cin_split_half, l2_reg_cin, seed, device=device)
            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False).to(device)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0], self.cin.named_parameters()),
                                           l2=l2_reg_cin)
        
        # multi-head atten
        att_embedding_size=20
        att_head_num=10
        att_layer_num=2
        att_res=True
        self.int_layers = nn.ModuleList(
            [InteractingLayer(self.embedding_size if i == 0 else att_embedding_size * att_head_num,
                              att_embedding_size, att_head_num, att_res, device=device) for i in range(att_layer_num)])

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                           self.embedding_dict)
        # ??????cin??????
        if self.use_cin:
#             print(sparse_embedding_list)
            cin_input = torch.cat(sparse_embedding_list, dim=1)
            cin_output = self.cin(cin_input) # 1024, 256
            cin_logit = self.cin_linear(cin_output)
        # muti-head
        att_input = concat_fun(sparse_embedding_list, axis=1)
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)
#             print(cin_output.shape)
        # ??????dnn
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list) # 1024, 101
#         print(dnn_input.shape)
#         dnn_input = torch.cat((dnn_input, cin_output), dim=1) # 0.631
        dnn_output = self.dnn(dnn_input)
#         print(dnn_output.shape)
        dnn_output = concat_fun([att_output, dnn_output])
#         print(dnn_output.shape)
#         dnn_output = torch.cat((dnn_output, cin_output), dim=1) # 0.634
        mmoe_outs = self.mmoe_layer(dnn_output)
        if self.task_dnn_units is not None:
            mmoe_outs = [self.task_dnn[i](mmoe_out) for i, mmoe_out in enumerate(mmoe_outs)]

        task_outputs = []
        for i, mmoe_out in enumerate(mmoe_outs):
            # cin???????????????
#             logit = self.tower_network[i](torch.cat((mmoe_out, cin_output), dim=1)) + cin_logit # 0.640
            logit = self.tower_network[i](mmoe_out) + cin_logit 

            output = self.out[i](logit)
            task_outputs.append(output)

        task_outputs = torch.cat(task_outputs, -1)
        return task_outputs
