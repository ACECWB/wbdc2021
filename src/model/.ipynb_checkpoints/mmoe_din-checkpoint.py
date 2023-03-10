# -*- coding:utf-8 -*-

"""
@author : LMC_ZC

reference : https://github.com/shenweichen/DeepCTR-Torch/

"""
import os
import sys
from model.sequence import AttentionSequencePoolingLayer
from inputs import *
from model.core import PredictionLayer, DNN
from dataset import CustomerTensorDataset


class GateLayer(nn.Linear):

    def __init__(self, input_dim, num_expert, device='cpu'):
        super(GateLayer, self).__init__(input_dim, num_expert)
        self.input_dim = input_dim
        self.num_expert = num_expert
        self.act = nn.Softmax(dim=1)
        self.to(device)

    def __call__(self, inputs):
        return self.act(self.forward(inputs))


class DeepInterestNetworkMMoe(nn.Module):
    def __init__(self, dnn_feature_columns, history_feature_list, seq_length_list,
                 att_hidden_size=(64, 32), att_activation='Dice', att_weight_normalization=False,
                 dnn_share_bottom_units=(256,), dnn_hidden_expert_units=(128,), dnn_hidden_units=(64,),
                 dnn_activation='relu', dnn_dropout=0.1,act_dropout=0.0,
                 l2_reg_dnn=1e-5, l2_reg_embedding=1e-5, dnn_use_bn=False,
                 task='binary', num_task=7, num_expert=5, device='cpu'):
        super(DeepInterestNetworkMMoe, self).__init__()

        self.num_task = num_task
        self.num_expert = num_expert
        self.dnn_feature_columns = dnn_feature_columns
        self.history_feature_list = history_feature_list
        self.seq_length_list = seq_length_list
        self.device = device

        self.aux_loss = torch.zeros((1,), device=device)

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        self.embedding_dict = self._create_embedding_matrix(dnn_feature_columns)
        self.feature_index = self._build_input_features(dnn_feature_columns)

        att_emb_dim = self._compute_interest_dim()
        dnn_in_dim = self._compute_input_dim(dnn_feature_columns)

        # attention + share_bottom + expert + tower
        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       dropout_rate=act_dropout,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)

        self.dnn_share_bottom = DNN(inputs_dim=dnn_in_dim,
                                    hidden_units=dnn_share_bottom_units,
                                    activation=dnn_activation,
                                    dropout_rate=dnn_dropout,
                                    use_bn=dnn_use_bn)

        self.dnn_expert = nn.ModuleList([
            DNN(inputs_dim=dnn_share_bottom_units[-1],
                hidden_units=dnn_hidden_expert_units,
                activation=dnn_activation,
                dropout_rate=dnn_dropout,
                use_bn=dnn_use_bn)
            for _ in range(num_expert)])

        self.gate_list = nn.ModuleList([
            GateLayer(dnn_share_bottom_units[-1], num_expert)
            for _ in range(num_task)])

        self.dnn_task = nn.ModuleList([
            DNN(inputs_dim=dnn_hidden_expert_units[-1],
                hidden_units=dnn_hidden_units,
                activation=dnn_activation,
                dropout_rate=dnn_dropout,
                use_bn=dnn_use_bn)
            for _ in range(num_task)])

        self.dnn_linear_task = nn.ModuleList([
            nn.Linear(dnn_hidden_units[-1], 1, bias=False)
            for _ in range(num_task)])

        self.out_task = nn.ModuleList([
            PredictionLayer(task)
            for _ in range(num_task)])

        self.regularization_weight = []
        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.dnn_task.parameters(), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_expert.parameters(), l2=l2_reg_dnn)

        self.to(device)

    def forward(self, X):

        _, dense_value_list = self._input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.history_feature_list, to_list=True)
        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         return_feat_list=self.history_fc_names, to_list=True)
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              to_list=True)

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)

        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)

        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)  # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)  # [B, T, E]

        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]

        hist = self.attention(query_emb, keys_emb, keys_length)  # [B, 1, E]

        # after attention
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)

        share_bottom_output = self.dnn_share_bottom(dnn_input)
        gate_output = [self.gate_list[k](share_bottom_output) for k in range(self.num_task)]
        dnn_expert_output = [self.dnn_expert[k](share_bottom_output) for k in range(self.num_expert)]

        tower_input = [torch.mean(torch.unsqueeze(weight, dim=1) * torch.stack(dnn_expert_output, dim=-1), dim=-1)
                       for weight in gate_output]

        outputs = []
        for k in range(self.num_task):
            inputs = tower_input[k]
            out = self.dnn_task[k](inputs)
            out = self.dnn_linear_task[k](out)
            out = self.out_task[k](out)
            outputs += [out]

        return torch.cat(outputs, dim=1)

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim

    def _compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def _create_embedding_matrix(self, feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        embedding_dict = nn.ModuleDict(
            {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1,
                                               sparse=sparse)
             for feat in
             sparse_feature_columns + varlen_sparse_feature_columns}
        )

        for tensor in embedding_dict.values():
            nn.init.xavier_normal_(tensor.weight)

        return embedding_dict.to(device)

    def _input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        varlen_sparse_embedding_list = get_varlen_pooling_list(self.embedding_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def _build_input_features(self, feature_columns):
        # Return OrderedDict: {feature_name:(start, start+dimension)}

        features = OrderedDict()

        start = 0
        for feat in feature_columns:
            feat_name = feat.name
            if feat_name in features:
                continue
            if isinstance(feat, SparseFeat):
                features[feat_name] = (start, start + 1)
                start += 1
            elif isinstance(feat, DenseFeat):
                features[feat_name] = (start, start + feat.dimension)
                start += feat.dimension
            elif isinstance(feat, VarLenSparseFeat):
                features[feat_name] = (start, start + feat.maxlen)
                start += feat.maxlen
                if feat.length_name is not None and feat.length_name not in features:
                    features[feat.length_name] = (start, start + 1)
                    start += 1
            else:
                raise TypeError("Invalid feature column type,got", type(feat))
        return features

    def generate_loader(self, x, y, table=None, return_table=False):

        x_, x_table = [], []
        # ???????????????seq len??????table?????????????????????????????????????????????hist?????????
#         if isinstance(x, dict):
#             for feature in self.feature_index:
#                 if feature in self.history_fc_names or feature in self.seq_length_list:
#                     x_table.append(x[feature])
#                 else:
#                     x_.append(x[feature])
        if isinstance(x, dict):
            for feature in self.feature_index:
                if feature in self.history_fc_names:
                    x_table.append(x[feature])
                else:
                    x_.append(x[feature])

        for i in range(len(x_)):
            if len(x_[i].shape) == 1:
                x_[i] = np.expand_dims(x_[i], axis=1)

        for i in range(len(x_table)):
            if len(x_table[i].shape) == 1:
                x_table[i] = np.expand_dims(x_table[i], axis=1)

        if table is not None:
            x_table = table

        if y is not None:
            y = torch.from_numpy(y)
        else:
            y = None

        train_tensor_data = CustomerTensorDataset(torch.from_numpy(np.concatenate(x_, axis=-1)), y, x_table)
        if return_table:
            return train_tensor_data, x_table
        else:
            return train_tensor_data

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss





# class GateLayer(nn.Linear):
#     def __init__(self, input_dim, num_expert, device='cpu'):
#         super(GateLayer, self).__init__(input_dim, num_expert)
#         self.input_dim = input_dim
#         self.num_expert = num_expert
#         self.act = nn.Softmax(dim=1)
#         self.to(device)
        
#     def __call__(self, x):
#         return self.act(self.forward(x))

# class DeepInterestNetworkMMoe(nn.Module):
#     def __init__(self, dnn_feature_columns, history_feature_list, seq_length_list,
#                  att_hidden_size=(64, 32), att_activation='Dice', att_weight_normalization=False,
#                  dnn_share_bottom_units=(256,), dnn_hidden_expert_units=(128, ), dnn_hidden_units=(64, ),
#                  dnn_activation='relu', dnn_dropout=0.1, act_dropout=0.0, l2_reg_dnn=1e-5, l2_reg_embedding=1e-5,
#                  dnn_use_bn=False, task='binary', num_task=7, num_expert=5, device='cpu'):
#         super(DeepInterestNetworkMMoe, self).__init__()
#         self.num_task = num_task
#         self.num_expert = num_expert
#         self.dnn_feature_columns = dnn_feature_columns
#         self.history_feature_list = history_feature_list
#         self.seq_length_list = seq_length_list
#         self.device = device
        
#         self.aux_loss = torch.zeros((1,), device=device)
        
#         self.sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
#         self.varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
#         self.dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
#         self.history_feature_columns = []
#         self.sparse_varlen_feature_columns = []
#         self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
#         for fc in varlen_sparse_feature_columns:
#             feat_name = fc.name
#             if feat_name in self.history_fc_names:
#                 self.history_feature_columns.append(fc)
#             else:
#                 self.sparse_varlen_feature_columns.append(fc)
                
#         self.embedding_dict = self._create_embedding_matrix(dnn_feature_columns)
#         self.feature_index = self._build_input_features(dnn_feature_columns)
#         att_emb_dim = self._compute_interest_dim()
#         dnn_in_dim = self._compute_input_dim(dnn_feature_columns)
        
#         self.attention = AttentionSequencePoolingLayer(att_hidden_size=att_hidden_size,
#                                                        embedding_dim=att_emb_dim,
#                                                        att_activation=att_activation,
#                                                        return_score=False,
#                                                        dropout_rate=act_dropout,
#                                                        supports_masking=False,
#                                                        weight_normalization=att_weight_normalization)
#         self.dnn_share_bottom = DNN(inputs_dim=dnn_in_dim,
#                                     hidden_units=dnn_share_bottom_units,
#                                     activation=dnn_activation,
#                                     dropout_rate=dnn_dropout,
#                                     use_bn=dnn_use_bn)
#         self.dnn_expert = nn.ModuleList([
#             DNN(inputs_dim=dnn_share_bottom_units[-1],
#                 hidden_units=dnn_hidden_expert_units,
#                 activation=dnn_activation,
#                 dropout_rate=dnn_dropout,
#                 use_bn=dnn_use_bn)
#             for _ in range(num_expert)
#         ])
        
#         self.gate_list = nn.ModuleList([
#             GateLayer(dnn_hidden_units[-1], num_expert) for _ in range(num_task)
#         ])
        
#         self.dnn_task = nn.ModuleList([
#             DNN(inputs_dim=dnn_hidden_expert_units[-1],
#                 hidden_units=dnn_hidden_units,
#                 activation=dnn_activation,
#                 dropout_rate=dnn_dropout,
#                 use_bn=dnn_use_bn)
#             for _ in range(num_task)
#         ])
        
#         self.dnn_task_output = nn.ModuleList([
#             nn.Linear(dnn_hidden_units[-1], 1)
#             for _ in range(num_task)
#         ])
        
#         self.dnn_task_predict = nn.ModuleList([
#             PredictionLayer(task)
#             for _ in range(num_task)
#         ])
        
#         self.regularization_weight = []
#         self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
#         self.add_regularization_weight(self.dnn_task.parameters(), l2=l2_reg_dnn)
#         self.add_regularization_weight(self.dnn_expert.parameters(), l2=l2_reg_dnn)
#         self.to(device)
        
        
#     def forward(self, X):
#         _, dense_value_list = self._input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
#         query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
#                                           return_feat_list=self.history_feature_list, to_list=True)
#         keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
#                                          return_feat_list=self.history_fc_names, to_list=True)
#         dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feat_columns, to_list=True)
#         sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_varlen_feature_columns)
#         sequence_emb_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index, self.sparse_varlen_feature_columns, self.device)
        
#         dnn_input_emb_list += sequence_emb_list
#         deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)
        
#         query_emb = torch.cat(query_emb_list, dim=-1)
#         key_emb = torch.cat(keys_emb_list, dim=-1)
#         key_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if feat.length_name is not None]
#         key_length = torch.squeeze(maxlen_lookup(X, self.feature_index, key_length_feature_name), 1)
#         hist = self.attention(query_emb, key_emb, key_length)
#         deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
#         deep_input_emb = deep_input_emb.view((deep_input_emb.shape[0], -1))
#         dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
#         share_bottom_output = self.dnn_share_bottom(dnn_input)
#         gate_outputs = [self.gate_list[k](share_bottom_output) for k in range(num_task)]
#         dnn_expert_outputs = [self.dnn_expert[k](share_bottom_output) for k in range(num_task)]
#         tower_input = [torch.mean(torch.unsqueeze(weight, dim=1) * torch.stack(dnn_expert_outputs, dim=-1),dim=-1) for weight in gate_outputs]
#         outputs = []
#         for i in range(num_task):
#             inputs = tower_input[i]
#             out = self.dnn_task[i](inputs)
#             out = self.dnn_task_output[i](out)
#             out = self.dnn_task_predict[i](out)
#             outputs += [out]
#         return torch.cat(outputs, dim=1)
        
    
#     def _input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):
#         sparse_feat_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
#         dense_feat_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
#         varlen_feat_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []
#         sparse_emb_list = [embedding_dict[feat.embedding_name](
#             X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in sparse_feat_columns]
#         dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long() for feat in dense_feat_columns]
#         varlen_emb_list = get_varlen_pooling_list(self.embedding_dict, X, self.feature_index, varlen_feat_columns, self.device)
#         return sparse_emb_list + varlen_emb_list, dense_value_list
        
#     def _create_embedding_matrix(self, feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
#         sparse_feat_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
#         varlen_sparse_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []
#         embedding_dict = nn.ModuleDict({
#             feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
#             for feat in sparse_feat_columns+varlen_sparse_columns
#         })
#         for tensor in embedding_dict.values():
#             nn.init.xavier_normal_(tensor.weight)
#         return embedding_dict.to(device)
    
#     def _build_input_features(self, feature_columns):
#         features = OrderedDict()
#         start = 0
#         for feat in feature_columns:
#             feat_name = feat.name
#             if feat_name in features:
#                 continue
#             if isinstance(feat, SparseFeat):
#                 features[feat_name] = (start, start+1)
#                 start += 1
#             elif isinstance(feat, DenseFeat):
#                 features[feat_name] = (start, start+feat.dimension)
#                 start += feat.dimension
#             elif isinstance(feat, VarLenSparseFeat):
#                 features[feat_name] = (start, start+feat.maxlen)
#                 start += feat.maxlen
#                 if feat.length_name not in features and feat.length_name is not None:
#                     features[feat.length_name] = (start, start+1)
#                     start += 1
#             else:
#                 raise TypeError("Invalid feature column type,got", type(feat))
#         return features
#     def _compute_interest_dim(self):
#         interest_dim = 0
#         for feat in self.sparse_feat_columns:
#             if feat.name in self.history_feature_list:
#                 interest_dim += feat.embedding_dim
#         return interest_dim
    
#     def _compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
#         sparse_feat_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
#         dense_feat_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
#         dense_input_dim = sum(map(lambda x: x.dimension, dense_feat_columns))
#         if feature_group:
#             sparse_input_dim = len(sparse_feat_columns)
#         else:
#             sparse_input_dim = sum(map(lambda x: x.embedding_dim, sparse_feat_columns))
#         input_dim = 0
#         if include_dense:
#             input_dim += dense_input_dim
#         if include_sparse:
#             input_dim += sparse_input_dim
#         return input_dim
#     def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
#         if isinstance(weight_list, torch.nn.parameter.Parameter):
#             weight_list = [weight_list]
#         else:
#             weight_list = list(weight_list)
#         self.regularization_weight.append((weight_list, l1, l2))
        
#     def get_regularization_loss(self):
#         total_reg_loss = torch.zeros((1,), device=self.device)
#         for weight, l1, l2 in self.regularization_weight:
#             for w in weight:
#                 if isinstance(weight, tuple):
#                     parameter = w[1]
#                 else:
#                     paramter = w
#                 if l1 > 0:
#                     total_reg_loss += torch.sum(torch.abs(parameter) * l1)
#                 if l2 > 0:
#                     try:
#                         total_reg_loss += torch.sum(l2 * torch.square(parameter))
#                     except AttributeError:
#                         total_reg_loss += torch.sum(l2 * parameter * parameter)
#         return total_reg_loss
#     def generate_loader(self, x, y, table=None, return_table=None):
#         x_, x_table = [], []
#         if isinstance(x, dict):
#             for feat in self.feature_index:
#                 if feat in self.history_fc_names or feat in self.seq_length_list:
#                     x_table.append(x[feat])
#                 else:
#                     x_.append(x[feat])
#         for i in range(len(x_)):
#             if len(x_[i].shape) == 1:
#                 x_[i] = np.expand_dims(x_[i], axis=1)
#         for i in range(len(x_table)):
#             if len(x_table[i]) == 1:
#                 x_table[i] = np.expand_dims(x_table[i], axis=1)
#         if table is not None:
#             x_table = table
#         if y is not None:
#             y = torch.from_numpy(y)
#         train_tensor_data = CustomerTensorDataset(torch.from_numpy(np.concatenate(x_, axis=-1)), y, x_table)
#         if return_table:
#             return train_tensor_data, x_table
#         else:
#             return train_tensor_data