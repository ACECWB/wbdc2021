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
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input, embedding_lookup, maxlen_lookup
from deepctr_torch.layers import DNN, PredictionLayer, CIN, concat_fun, InteractingLayer,SENETLayer, BilinearInteraction, FM
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer
import pandas as pd


class SequencePoolingLayer(nn.Module):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.

    """

    def __init__(self, mode='mean', supports_masking=False, device='cpu'):

        super(SequencePoolingLayer, self).__init__()
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError('parameter mode should in [sum, mean, max]')
        self.supports_masking = supports_masking
        self.mode = mode
        self.device = device
        self.eps = torch.FloatTensor([1e-8]).to(device)
        self.to(device)

    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        # Returns a mask tensor representing the first N positions of each cell.
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def forward(self, seq_value_len_list):
        if self.supports_masking:
            uiseq_embed_list, mask = seq_value_len_list  # [B, T, E], [B, 1]
            mask = mask.float()
            user_behavior_length = torch.sum(mask, dim=-1, keepdim=True)
            mask = mask.unsqueeze(2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list  # [B, T, E], [B, 1]
            mask = self._sequence_mask(user_behavior_length, maxlen=uiseq_embed_list.shape[1],
                                       dtype=torch.float32)  # [B, 1, maxlen]
            mask = torch.transpose(mask, 1, 2)  # [B, maxlen, 1]

        embedding_size = uiseq_embed_list.shape[-1]

        mask = torch.repeat_interleave(mask, embedding_size, dim=2)  # [B, maxlen, E]

        if self.mode == 'max':
            hist = uiseq_embed_list - (1 - mask) * 1e9
            hist = torch.max(hist, dim=1, keepdim=True)[0]
            return hist
        hist = uiseq_embed_list * mask.float()
        hist = torch.sum(hist, dim=1, keepdim=False)

        if self.mode == 'mean':
            self.eps = self.eps.to(user_behavior_length.device)
            hist = torch.div(hist, user_behavior_length.type(torch.float32) + self.eps)

        hist = torch.unsqueeze(hist, dim=1)
        return hist



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

    def __init__(self, dnn_feature_columns, history_feature_list, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
                 l2_reg_embedding=1e-5, l2_reg_dnn=1e-5, init_std=0.0001, task_dnn_units=None, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, device='cpu', gpus=[0, 1]):
        
        super(MMOE, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   l2_reg_embedding=l2_reg_embedding, seed=seed, device=device)
#         self.device = [0]
        if num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(tasks) != num_tasks:
            raise ValueError("num_tasks must be equal to the length of tasks")
        for task in tasks:
            if task not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task))
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        
        # atten tag key
        self.history_feature_list = history_feature_list
        self.item_features = history_feature_list

        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)
        self.fm = FM()
        
        # din component
        att_emb_dim = self._compute_interest_dim()
        use_negsampling = False
        att_hidden_size=(128, 64)
        self.use_negsampling = use_negsampling
        att_activation='relu'
        att_weight_normalization=False
        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)

        # 加入cin
        cin_layer_size=(256, 128, 64)
        self.cin_layer_size=cin_layer_size
        cin_split_half=True
        cin_activation='relu'
        l2_reg_cin=1e-5
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        if self.use_cin:
#             field_num = len(self.embedding_dict)
            field_num = len(self.sparse_feature_columns)

            if cin_split_half == True:
                self.featuremap_num = sum(
                    cin_layer_size[:-1]) // 2 + cin_layer_size[-1]
            else:
                self.featuremap_num = sum(cin_layer_size)
            self.cin = CIN(field_num, cin_layer_size,
                           cin_activation, cin_split_half, l2_reg_cin, seed, device=device)
            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False).to(device)
#             self.add_regularization_weight(filter(lambda x: 'weight' in x[0], self.cin.named_parameters()),
#                                            l2=l2_reg_cin)
        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        # multi-head atten
        att_embedding_size=30
        att_head_num=15
        att_layer_num=5
        att_res=True
#         self.int_layers = nn.ModuleList(
#             [InteractingLayer(self.embedding_size if i == 0 else att_embedding_size * att_head_num,
#                               att_embedding_size, att_head_num, att_res, device=device) for i in range(att_layer_num)])

        # InteractingLayer (used in AutoInt) = multi-head self-attention + Residual Network
        self.vector_wise_net = InteractingLayer(self.embedding_size, att_embedding_size,
                                                att_head_num, att_res, scaling=True, device=device)

        self.bit_wise_net = DNN(self.compute_input_dim(dnn_feature_columns, include_dense=False)-500,
                                         dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn,
                                         dropout_rate=dnn_dropout,
                                         use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.sparse_feat_num = len(list(filter(lambda x: isinstance(x, SparseFeat) ,
                                               dnn_feature_columns)))

        self.transform_matrix_P_vec = nn.Linear(
            self.sparse_feat_num*att_embedding_size*att_head_num, self.sparse_feat_num, bias=False).to(device)
        self.transform_matrix_P_bit = nn.Linear(
            dnn_hidden_units[-1], self.sparse_feat_num, bias=False).to(device)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.vector_wise_net.named_parameters()),
            l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.bit_wise_net.named_parameters()),
            l2=l2_reg_dnn)
        self.add_regularization_weight(self.transform_matrix_P_vec.weight, l2=l2_reg_dnn)
        self.add_regularization_weight(self.transform_matrix_P_bit.weight, l2=l2_reg_dnn)
        
        
        if len(dnn_hidden_units) and att_layer_num > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1] + \
                                    field_num * att_embedding_size * att_head_num
        elif len(dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_num * att_embedding_size * att_head_num
        else:
            raise NotImplementedError
        
        reduction_ratio=3
        bilinear_type='interaction'
#         self.filed_size = len(self.embedding_dict)
        self.filed_size = len(self.sparse_feature_columns)
        self.SE = SENETLayer(self.filed_size, reduction_ratio, seed, device)
        self.Bilinear = BilinearInteraction(self.filed_size, self.embedding_size, bilinear_type, seed, device)

        print('sparse_features: ', self.sparse_feature_columns)
        print('varlen_features: ', self.varlen_sparse_feature_columns)
        
         # MMOE
        self.tasks = tasks
        self.task_dnn_units = task_dnn_units
        self.dnn = DNN(
#             self.compute_input_dim(self.sparse_feature_columns + self.dense_feature_columns) +16 + dnn_linear_in_feature + self.featuremap_num, 
            self.compute_input_dim(self.sparse_feature_columns + self.dense_feature_columns) + 4500+1700-3400-1680+880-2001+3200-1400+1800+1500+3200+300-200-300-8100+8600-900,   
            dnn_hidden_units,
            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)
        self.mmoe_layer = MMOELayer(dnn_hidden_units[-1]+2800-1300+1500-3000, num_tasks, num_experts, expert_dim)
        if task_dnn_units is not None:
            # the last layer of task_dnn should be expert_dim
            self.task_dnn = nn.ModuleList([DNN(expert_dim, task_dnn_units+(expert_dim,), dropout_rate=0.1) for _ in range(num_tasks)])
        self.tower_network = nn.ModuleList([nn.Linear(expert_dim, 1, bias=False) for _ in range(num_tasks)])
        self.out = nn.ModuleList([PredictionLayer(task) for task in self.tasks])
        self.to(device)
        
    
    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim
    
    
    def forward(self, X):
#         print(self.embedding_dict)
        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                           self.embedding_dict)
        sparse_embedding_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              to_list=True)
        sparse_embedding_input_ori = torch.cat(sparse_embedding_list, dim=-1).squeeze(1)
        sparse_embedding_input_bi = torch.cat(sparse_embedding_list, dim=1)
#         #DIN
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.history_feature_list, to_list=True)
        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         return_feat_list=self.history_fc_names, to_list=True)
        
        query_emb = torch.cat(query_emb_list, dim=-1)                     # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)                       # [B, T, E]
        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]
        hist = self.attention(query_emb, keys_emb, keys_length).squeeze(1)           # [B, 1, E]


#         加入cin模块
        if self.use_cin:
            cin_input = torch.cat(sparse_embedding_list, dim=1)
            cin_output = self.cin(cin_input) # 1024, 256
#             cin_logit = self.cin_linear(cin_output)
            
          # BI
        senet_output = self.SE(sparse_embedding_input_bi)
        senet_bilinear_out = self.Bilinear(senet_output)
        bilinear_out = self.Bilinear(sparse_embedding_input_bi)
        temp = torch.split(torch.cat((senet_bilinear_out, bilinear_out), dim=1), 1, dim=1)
        bi = combined_dnn_input(temp, [])

        # DIFM
        att_input = concat_fun(sparse_embedding_list, axis=1)
        att_out = self.vector_wise_net(att_input)
        att_out = att_out.reshape(att_out.shape[0], -1)
        m_vec = self.transform_matrix_P_vec(att_out)
        
        dnn_input = combined_dnn_input(sparse_embedding_list, [])
        dnn_output = self.bit_wise_net(dnn_input)
        m_bit = self.transform_matrix_P_bit(dnn_output)
        m_x = m_vec + m_bit
        
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        refined_fm_input = fm_input * m_x.unsqueeze(-1)
        fm_logit = self.fm(refined_fm_input)

        sparse_embedding_input = sparse_embedding_input_ori.reshape((-1, 100, self.sparse_feat_num))
        
        sparse_embedding_input = sparse_embedding_input * m_x.unsqueeze(1)
        sparse_embedding_input = sparse_embedding_input.reshape((X.size(0), -1))
        sparse_embedding_input = concat_fun([sparse_embedding_input, bi])
#         sparse_embedding_input = concat_fun([sparse_embedding_input, sparse_embedding_input_ori.reshape((X.size(0), -1))])
        sparse_embedding_input = concat_fun([sparse_embedding_input, hist])
        
        dnn_output = self.dnn(sparse_embedding_input)
        mmoe_outs = self.mmoe_layer(dnn_output)
    
        if self.task_dnn_units is not None:
            mmoe_outs = [self.task_dnn[i](mmoe_out) for i, mmoe_out in enumerate(mmoe_outs)]

        task_outputs = []
        for i, mmoe_out in enumerate(mmoe_outs):
            logit = self.tower_network[i](mmoe_out) + fm_logit
            output = self.out[i](logit)
            task_outputs.append(output)
            
        task_outputs = torch.cat(task_outputs, -1)
        return task_outputs


dense_features = [
 'videoplayseconds',
]

import os
import torch
import pandas as pd
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '.'
sys.path.append(os.path.join(BASE_DIR, '../../config'))
sys.path.append(os.path.join(BASE_DIR, '../model'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from config import *
from time import time
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from sklearn.preprocessing import MinMaxScaler
import datatable as dt
# from mmoe import MMOE
from evaluation import evaluate_deepctr
import pickle
import gc

# 训练相关参数设置
ONLINE_FLAG = True # 是否准备线上提交

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# 
vocab_dict = {
    'bgm_song_id': 25158+1,
    'bgm_singer_id': 17499+1,
    'userid': 199999,
#     'feedid': 106444,# 112871+1
    'feedid':112871+1,
    'authorid': 18788+1,
    'device' : 3
}

# if __name__ == "__main__":
epochs = 1
batch_size = 1024
embedding_dim = 100
max_hist_seq_len = 100

target = ['read_comment', 'like', 'click_avatar', 'forward', 'comment', 'favorite', 'follow']
# tagids = ['manual_tag_' + str(tagid) for tagid in range(11)] # 11
# keyids = ['manual_key_' + str(keyid) for keyid in range(11)] # 18
sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
# dense_features += ['videoplayseconds', ]

feed = dt.fread(FEED_INFO)
feed = feed.to_pandas()
# tag = dt.fread(FEATURE_PATH + '/feed_info_tags_keys_des_seq_len.csv')
# tag = tag.to_pandas()[tagids + keyids + ['feedid', 'tag_seq_len', 'key_seq_len']]

pkl = open(FEATURE_PATH + '/user_encoder.pkl', 'rb')
userid_map = pickle.load(pkl)
pkl.close()
pkl = open(FEATURE_PATH + '/feedid_encoder.pkl', 'rb')
feedid_map = pickle.load(pkl)
pkl.close()

mms = MinMaxScaler(feature_range=(0, 1))

user_emb1 = np.load(FEATURE_PATH + '/user_emb_normal_100.npy')
# user_emb1 = mms.fit_transform(user_emb1)
user_emb1 = torch.from_numpy(user_emb1).float().to(device)

user_emb2 = np.load(FEATURE_PATH + '/user_emb_adjust_100.npy')
# user_emb2 = mms.fit_transform(user_emb2)
user_emb2 = torch.from_numpy(user_emb2).float().to(device)

# user_emb1 = np.load(FEATURE_PATH + '/user_emb_normal_50.npy')
# user_emb1 = torch.from_numpy(user_emb1).float().to(device)
# user_emb2 = np.load(FEATURE_PATH + '/user_emb_adjust_50.npy')
# user_emb2 = torch.from_numpy(user_emb2).float().to(device)

feed_emb = np.load(FEATURE_PATH + '/feed_embedding_100.npy')
# feed_emb = mms.fit_transform(feed_emb)
feed_emb = torch.from_numpy(feed_emb).float().to(device)


# svd_user = np.load(FEATURE_PATH + '/svd_user_50.npy')
# svd_user = torch.from_numpy(svd_user).float().to(device)
# svd_feed = np.load(FEATURE_PATH + '/svd_feed_50.npy')
# svd_feed = torch.from_numpy(svd_feed).float().to(device)

hist_file = open(FEATURE_PATH + '/hist_data_action_begin_100.pkl', 'rb')
hist_data = pickle.load(hist_file)
hist_file.close()

# feed[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
# feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
#     feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
# feed['bgm_song_id'] = feed['bgm_song_id'].astype('int64')
# feed['bgm_singer_id'] = feed['bgm_singer_id'].astype('int64')


if ONLINE_FLAG:
    data = pd.read_csv(FEATURE_PATH + '/online_train_100.csv', iterator=True)
    test = pd.read_csv(FEATURE_PATH + '/online_test_100.csv')
#     test['feedid'] = feedid_map.transform(test['feedid'])
else:
    val = pd.read_csv(FEATURE_PATH + '/offline_val_100.csv')
    data = pd.read_csv(FEATURE_PATH + '/offline_train_100.csv', iterator=True)
#     val['feedid'] = feedid_map.transform(val['feedid'])

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=vocab_dict[feat] + 1, embedding_dim=embedding_dim)
                      for feat in vocab_dict.keys()] + [DenseFeat(feat, 1) for feat in dense_features]

fixlen_feature_columns += [SparseFeat('feed_embedding', vocab_dict['feedid'], embedding_dim=embedding_dim)]
fixlen_feature_columns += [SparseFeat('user_embedding_normal', 200000, embedding_dim=embedding_dim)]
fixlen_feature_columns += [SparseFeat('user_embedding_adjust', 200000, embedding_dim=embedding_dim)]
# fixlen_feature_columns += [SparseFeat('svd_user', 200000, embedding_dim=embedding_dim)]
# fixlen_feature_columns += [SparseFeat('svd_feed', 106444, embedding_dim=embedding_dim)]

if ONLINE_FLAG:
    # 加入test
#     test = dt.fread(TEST_FILE)
#     test = test.to_pandas()
#     test = test.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
#                       on='feedid')
#     test = test.merge(tag, how='left', on='feedid')
    fixlen_feature_columns += [VarLenSparseFeat(SparseFeat('hist_feed_embedding', vocabulary_size=vocab_dict['feedid'], embedding_dim=embedding_dim), max_hist_seq_len, length_name='test_seq_len')]
    fixlen_feature_columns += [VarLenSparseFeat(SparseFeat('hist_feedid', vocabulary_size=vocab_dict['feedid'], embedding_dim=embedding_dim), max_hist_seq_len, length_name='test_seq_len')]
    fixlen_feature_columns += [VarLenSparseFeat(SparseFeat('hist_authorid', vocabulary_size=vocab_dict['authorid']+1, embedding_dim=embedding_dim), max_hist_seq_len, length_name='test_seq_len')]
    fixlen_feature_columns += [VarLenSparseFeat(SparseFeat('hist_bgm_song_id', vocabulary_size=vocab_dict['bgm_song_id']+1, embedding_dim=embedding_dim), max_hist_seq_len, length_name='test_seq_len')]
    fixlen_feature_columns += [VarLenSparseFeat(SparseFeat('hist_bgm_singer_id', vocabulary_size=vocab_dict['bgm_singer_id']+1, embedding_dim=embedding_dim), max_hist_seq_len, length_name='test_seq_len')]

    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)

#     test[dense_features] = test[dense_features].fillna(0, )
#     test[dense_features] = mms.fit_transform(test[dense_features])
#     test['userid'] = userid_map.transform(test['userid'])

    test_model_input = {name: test[name] for name in feature_names if 'hist' not in name and 'seq_len'not in name and name not in ['svd_feed', 'svd_user', 'seq_len', 'feed_emb', 'feed_embedding', 'user_embedding_normal', 'user_embedding_adjust', 'hist_tagids', 'hist_keyids', 'tagids', 'keyids']}
    test_model_input['user_embedding_normal'] = test_model_input['userid']
    test_model_input['user_embedding_adjust'] = test_model_input['userid']
#     test_model_input['svd_user'] = test_model_input['userid']
#     test_model_input['svd_feed'] = test_model_input['feedid']
    test_model_input['hist_feedid'] = hist_data['hist_feedid'][test_model_input['userid']][:, :max_hist_seq_len]
    test_model_input['hist_feed_embedding'] = hist_data['hist_feedid'][test_model_input['userid']][:, :max_hist_seq_len]
    test_model_input['feed_embedding'] = test_model_input['feedid']
    
    test_model_input['hist_authorid'] = hist_data['hist_authorid'][test_model_input['userid']][:, :max_hist_seq_len]
    test_model_input['hist_bgm_song_id'] = hist_data['hist_bgm_song_id'][test_model_input['userid']][:, :max_hist_seq_len]
    test_model_input['hist_bgm_singer_id'] = hist_data['hist_bgm_singer_id'][test_model_input['userid']][:, :max_hist_seq_len]
    test_model_input['test_seq_len'] = test['real_seq_len_100'].values
#     test_model_input['test_seq_len'] = hist_data['test_seq_len'][test_model_input['userid']]
else:
#     val = val.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
#                   on='feedid')
    fixlen_feature_columns += [VarLenSparseFeat(SparseFeat('hist_feed_embedding', vocabulary_size=vocab_dict['feedid'], embedding_dim=embedding_dim), max_hist_seq_len, length_name='val_seq_len')]
    
    fixlen_feature_columns += [VarLenSparseFeat(SparseFeat('hist_feedid', vocabulary_size=vocab_dict['feedid'], embedding_dim=embedding_dim), max_hist_seq_len, length_name='val_seq_len')]
    fixlen_feature_columns += [VarLenSparseFeat(SparseFeat('hist_authorid', vocabulary_size=vocab_dict['authorid']+1, embedding_dim=embedding_dim), max_hist_seq_len, length_name='val_seq_len')]
    fixlen_feature_columns += [VarLenSparseFeat(SparseFeat('hist_bgm_song_id', vocabulary_size=vocab_dict['bgm_song_id']+1, embedding_dim=embedding_dim), max_hist_seq_len, length_name='val_seq_len')]
    fixlen_feature_columns += [VarLenSparseFeat(SparseFeat('hist_bgm_singer_id', vocabulary_size=vocab_dict['bgm_singer_id']+1, embedding_dim=embedding_dim), max_hist_seq_len, length_name='val_seq_len')]

    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)
    
#     val[dense_features] = val[dense_features].fillna(0, )
#     val[dense_features] = mms.fit_transform(val[dense_features])
#     val['userid'] = userid_map.transform(val['userid'])
    
    val_model_input = {name: val[name] for name in feature_names if 'hist' not in name and 'seq_len' not in name and name not in ['svd_feed', 'svd_user', 'feed_emb', 'hist_tagids', 'hist_keyids', 'feed_embedding', 'user_embedding_normal', 'user_embedding_adjust', 'tagids', 'keyids']}
    
    val_model_input['hist_feedid'] = hist_data['hist_feedid'][val_model_input['userid']][:, :max_hist_seq_len]
    val_model_input['hist_feed_embedding'] = hist_data['hist_feedid'][val_model_input['userid']][:, :max_hist_seq_len]
    val_model_input['feed_embedding'] = val_model_input['feedid']

    val_model_input['hist_authorid'] = hist_data['hist_authorid'][val_model_input['userid']][:, :max_hist_seq_len]
    val_model_input['hist_bgm_song_id'] = hist_data['hist_bgm_song_id'][val_model_input['userid']][:, :max_hist_seq_len]
    val_model_input['hist_bgm_singer_id'] = hist_data['hist_bgm_singer_id'][val_model_input['userid']][:, :max_hist_seq_len]
#     val_model_input['val_seq_len'] = hist_data['val_seq_len'][val_model_input['userid']]
    val_model_input['val_seq_len'] = val['real_seq_len_100'].values

    val_model_input['user_embedding_normal'] = val_model_input['userid']
    val_model_input['user_embedding_adjust'] = val_model_input['userid']
#     val_model_input['svd_user'] = val_model_input['userid']
#     val_model_input['svd_feed'] = val_model_input['feedid']

    userid_list = val['userid'].astype(str).tolist()
    val_labels = [val[y].values for y in target]
    

print('use features: ', feature_names)


hist_features = ['feedid', 'feed_embedding', 'authorid', 'bgm_singer_id', 'bgm_song_id']
# hist_features = ['feedid', 'feed_embedding']

# hist_features = []

train_model = MMOE(dnn_feature_columns, history_feature_list=hist_features, num_tasks=7, num_experts=15, expert_dim=128, dnn_hidden_units=(512, 256, 64),
                   task_dnn_units=(128, 128, 64),
                   tasks=['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary'], device=device)
print(train_model.feature_index)

train_model.compile("adagrad", loss='binary_crossentropy')
train_model.embedding_dict['user_embedding_normal'] = nn.Embedding.from_pretrained(user_emb1, freeze=False)
train_model.embedding_dict['user_embedding_adjust'] = nn.Embedding.from_pretrained(user_emb2, freeze=False)
train_model.embedding_dict['feed_embedding'] = nn.Embedding.from_pretrained(feed_emb, freeze=False)

# train_model.embedding_dict['user_embedding_normal'] = nn.Embedding.from_pretrained(user_emb1)
# train_model.embedding_dict['user_embedding_adjust'] = nn.Embedding.from_pretrained(user_emb2)
# train_model.embedding_dict['feed_embedding'] = nn.Embedding.from_pretrained(feed_emb)

# train_model.embedding_dict['svd_user'] = nn.Embedding.from_pretrained(svd_user, freeze=False)
# train_model.embedding_dict['svd_feed'] = nn.Embedding.from_pretrained(svd_feed, freeze=False)
loop = True
cnt = 0
while loop:
    try:
        cnt += 1
        print('chunk: ', cnt)
        train = data.get_chunk(100*10000)
#         train['feedid'] = feedid_map.transform(train['feedid'])
        
#         train = train.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
#                   on='feedid')
#         train[dense_features] = train[dense_features].fillna(0, )
#         train[dense_features] = mms.fit_transform(train[dense_features])
#         train['userid'] = userid_map.transform(train['userid'])                

        train_model_input = {name: train[name] for name in feature_names  if 'hist' not in name and 'seq_len'not in name and name not in ['svd_feed', 'svd_user', 'feed_emb', 'hist_tagids', 'hist_keyids', 'feed_embedding', 'user_embedding_normal', 'user_embedding_adjust', 'tagids', 'keyids']}
    
        train_model_input['hist_feedid'] = hist_data['hist_feedid'][train_model_input['userid']][:, :max_hist_seq_len]
        train_model_input['hist_feed_embedding'] = hist_data['hist_feedid'][train_model_input['userid']][:, :max_hist_seq_len]
        train_model_input['feed_embedding'] = train_model_input['feedid']

        train_model_input['hist_authorid'] = hist_data['hist_authorid'][train_model_input['userid']][:, :max_hist_seq_len]
        train_model_input['hist_bgm_song_id'] = hist_data['hist_bgm_song_id'][train_model_input['userid']][:, :max_hist_seq_len]
        train_model_input['hist_bgm_singer_id'] = hist_data['hist_bgm_singer_id'][train_model_input['userid']][:, :max_hist_seq_len]
        
        if ONLINE_FLAG:
#             train_model_input['test_seq_len'] = hist_data['test_seq_len'][train_model_input['userid']]    
            train_model_input['test_seq_len'] = train['real_seq_len_100'].values   

        else:
#             train_model_input['val_seq_len'] = hist_data['val_seq_len'][train_model_input['userid']]    
            train_model_input['val_seq_len'] = train['real_seq_len_100'].values 

        train_model_input['user_embedding_normal'] = train_model_input['userid']
        train_model_input['user_embedding_adjust'] = train_model_input['userid']
#         train_model_input['svd_user'] = train_model_input['userid']
#         train_model_input['svd_feed'] = train_model_input['feedid']

        train_labels = train[target].values

        for epoch in range(epochs):
            history = train_model.fit(train_model_input, train_labels,
                              batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
        if not ONLINE_FLAG:
            val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
            # 模型predict()返回值格式为(?, 4)，与tf版mmoe不同。因此下方用到了transpose()进行变化。
            evaluate_deepctr(val_labels, val_pred_ans.transpose(), userid_list, target)

    except StopIteration:
        loop=False
        print('Finished all train')

        
# val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
# evaluate_deepctr(val_labels, val_pred_ans.transpose(), userid_list, target)

# 继续train一波 最近的数据
del data, train
if ONLINE_FLAG:
    data = pd.read_csv(FEATURE_PATH + '/online_train_100.csv', iterator=True)
    train = data.get_chunk(6500*10000)
else:
    data = pd.read_csv(FEATURE_PATH + '/offline_train_100.csv', iterator=True)
    train = data.get_chunk(6000*10000)
loop = True
cnt = 0
while loop:
    try:
        cnt += 1
        print('chunk: ', cnt)
        train = data.get_chunk(100*10000)
#         train['feedid'] = feedid_map.transform(train['feedid'])
        
#         train = train.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
#                   on='feedid')
#         train[dense_features] = train[dense_features].fillna(0, )
#         train[dense_features] = mms.fit_transform(train[dense_features])
#         train['userid'] = userid_map.transform(train['userid'])                

        train_model_input = {name: train[name] for name in feature_names  if 'hist' not in name and 'seq_len'not in name and name not in ['svd_feed', 'svd_user', 'feed_emb', 'hist_tagids', 'hist_keyids', 'feed_embedding', 'user_embedding_normal', 'user_embedding_adjust', 'tagids', 'keyids']}
    
        train_model_input['hist_feedid'] = hist_data['hist_feedid'][train_model_input['userid']][:, :max_hist_seq_len]
        train_model_input['hist_feed_embedding'] = hist_data['hist_feedid'][train_model_input['userid']][:, :max_hist_seq_len]
        train_model_input['feed_embedding'] = train_model_input['feedid']

        train_model_input['hist_authorid'] = hist_data['hist_authorid'][train_model_input['userid']][:, :max_hist_seq_len]
        train_model_input['hist_bgm_song_id'] = hist_data['hist_bgm_song_id'][train_model_input['userid']][:, :max_hist_seq_len]
        train_model_input['hist_bgm_singer_id'] = hist_data['hist_bgm_singer_id'][train_model_input['userid']][:, :max_hist_seq_len]
        
        if ONLINE_FLAG:
#             train_model_input['test_seq_len'] = hist_data['test_seq_len'][train_model_input['userid']]    
            train_model_input['test_seq_len'] = train['real_seq_len_100'].values   

        else:
#             train_model_input['val_seq_len'] = hist_data['val_seq_len'][train_model_input['userid']]    
            train_model_input['val_seq_len'] = train['real_seq_len_100'].values 

        train_model_input['user_embedding_normal'] = train_model_input['userid']
        train_model_input['user_embedding_adjust'] = train_model_input['userid']
#         train_model_input['svd_user'] = train_model_input['userid']
#         train_model_input['svd_feed'] = train_model_input['feedid']

        train_labels = train[target].values

        for epoch in range(epochs):
            history = train_model.fit(train_model_input, train_labels,
                              batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
        if not ONLINE_FLAG:
            val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
            # 模型predict()返回值格式为(?, 4)，与tf版mmoe不同。因此下方用到了transpose()进行变化。
            evaluate_deepctr(val_labels, val_pred_ans.transpose(), userid_list, target)

    except StopIteration:
        loop=False
        print('Finished all train')
    
    
    
del data

if ONLINE_FLAG:
    t1 = time()
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 4)
    pred_ans = pred_ans.transpose()
    t2 = time()
    print('7个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test), (t2 - t1) * 1000.0))
    ts = (t2 - t1) * 1000.0* 2000.0 / (len(test)*7.0) 
    print('7个目标行为2000条样本平均预测耗时（毫秒）：%.3f' % ts)

    # # 5.生成提交文件
    for i, action in enumerate(target):
        test[action] = pred_ans[i]
    test['userid'] = userid_map.inverse_transform(test['userid'])
#     test['feedid'] = feedid_map.inverse_transform(test['feedid'])
    test[['userid', 'feedid'] + target].to_csv(SUBMIT_DIR + '/3mmoe_din_cin_difm_bi_1024_1.csv', index=None, float_format='%.6f')
    print('to_csv ok')
    
    
   # 674669