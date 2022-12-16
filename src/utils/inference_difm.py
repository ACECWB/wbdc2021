# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, './utils'))
sys.path.append(os.path.join(BASE_DIR, './model'))
sys.path.append(os.path.join(BASE_DIR, '../config'))
from config import *
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names,VarLenSparseFeat
print(sys.path)
from frt import *
from difm_model import MyDIFM
from utils import *
import datatable as dt

ACTION_LIST = ['read_comment', 'like', 'click_avatar', 'forward', 'comment', 'favorite', 'follow']
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
#
NUM_EPOCH_DICT = {"read_comment": 2, "like": 1, "click_avatar": 1,"forward": 1,
                                "comment": 1, "follow": 1, "favorite": 1, }

def task(action,myseed):
    print('-----------action-----------', action)
    USE_FEAT = [action] + SELECT_FRTS
    dt_train = dt.fread(FEATURE_PATH + '/train_data.csv')
    train = dt_train.to_pandas()[USE_FEAT]
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    print("posi prop:")
    print(sum((train[action] == 1) * 1) / train.shape[0])
    dt_test = dt.fread(TEST_FILE)
    test = dt_test.to_pandas()
    target = [action]
    test[target[0]] = 0
    # test['date_'] = 15

    # test = test[USE_FEAT]

    # test add spare
    dt_feed_info = dt.fread(FEED_INFO)
    # print(dt_feed_info.keys())
    # exit()
    dt_feed_info = dt_feed_info.to_pandas()[['feedid', 'authorid',
                                             'bgm_song_id', 'bgm_singer_id',
                                             'videoplayseconds', ]]
    test = test.merge(dt_feed_info, how='left', on='feedid')

    data = pd.concat((train, test)).reset_index(drop=True)
    dt_feed = dt.fread(FEATURE_PATH + '/feed_embeddings_PCA32.csv')
    feed = dt_feed.to_pandas()[feature_list('feed_emb') + ['feedid']]
    data = pd.merge(data, feed, how='left', on='feedid')
    dt_tag_key_des = dt.fread(FEATURE_PATH + '/feed_info_tags_keys_des.csv')
    tag = dt_tag_key_des.to_pandas()[feature_list('tag') + feature_list('key') + feature_list('des') + ['feedid']]
    data = pd.merge(data, tag, how='left', on='feedid')
    dt_ocr_asr_ocrc_asrc_desc = dt.fread(FEATURE_PATH + '/feed_info_splited_nlp.csv')
    tag = dt_ocr_asr_ocrc_asrc_desc.to_pandas()[
        feature_list('ocr') + feature_list('asr') + feature_list('ocr_char') + feature_list('asr_char') + feature_list(
            'des_char') + ['feedid']]
    data = pd.merge(data, tag, how='left', on='feedid')

    print(train.shape, test.shape, data.shape)
    print("all_feature", list(data.keys()))
    dense_features = DENSE_FEATURE
    sparse_features = [i for i in USE_FEAT if i not in dense_features and i not in target]

    data[sparse_features] = data[sparse_features].fillna(0)
    data[dense_features] = data[dense_features].fillna(0)

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features+feature_list("feed_emb")] = mms.fit_transform(data[dense_features+feature_list("feed_emb")])

    # 2.count #unique features for each sparse field,and record dense feature field name
    var_features = list(my_name_dict.keys())
    var_features.remove('ocr_char')
    var_features.remove('asr_char')
    var_features.remove('des_char')
    # var_features.remove('ocr')
    # var_features.remove('asr')
    # var_features.remove('des')
    # var_features.remove('key')
    var_features.remove('feed_emb')
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(),50)
                        for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]\
                             # +[DenseFeat('feed_emb', 32, )]
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat(feat,
                                                          vocabulary_size=my_vocab_dict[feat] + 1, embedding_dim=50),
                                               maxlen=my_len_dict[feat], combiner='mean') for feat in var_features if feat != "feed_emb" ]

    fixlen_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    #linear_feature_columns = [SparseFeat(feat, data[feat].nunique())
        #                         for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = data.iloc[:train.shape[0]].reset_index(drop=True), data.iloc[train.shape[0]:].reset_index(drop=True)
    train_model_input = {name: train[name] for name in feature_names
                            if name in SELECT_FRTS}
    test_model_input = {name: test[name] for name in feature_names
                            if name in SELECT_FRTS}
    for feat in my_name_dict.keys():
        train_model_input[feat] = train[feature_list(feat)].values
        test_model_input[feat] = test[feature_list(feat)].values
    #-------
    eval_ratio=0.
    eval_df=train[int((1-eval_ratio)*train.shape[0]):].reset_index(drop=True)
    userid_list=eval_df['userid'].astype(str).tolist()
    print('val len:',len(userid_list))

    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    # model = MyDeepFM(linear_feature_columns=linear_feature_columns,
    #                 dnn_feature_columns=dnn_feature_columns,
    #                 use_fm=True,
    #                 dnn_hidden_units=(256,128),
    #                 l2_reg_linear=1e-1, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
    #                 dnn_dropout=0.,
    #                 dnn_activation='relu',
    #                 dnn_use_bn=False, task='binary', device=device)
    # model = MyAutoInt(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                 att_layer_num=3, att_embedding_size=64, att_head_num=2,device=device,seed=2021)
    # model =  MyDIFM(
    #              linear_feature_columns, dnn_feature_columns, att_embedding_size=8, att_head_num=8,
    #              att_res=True, dnn_hidden_units=(256, 128),
    #              l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=121,
    #              dnn_dropout=0,
    #              dnn_activation='relu', dnn_use_bn=False, task='binary', device=device, gpus=None)
    # model.compile("adam", "binary_crossentropy", metrics=["auc"])
    #
    # history = model.fit(train_model_input, train[target].values, batch_size=1024,
    #                     epochs=NUM_EPOCH_DICT[action], verbose=1,
    #                     validation_split=eval_ratio,userid_list=userid_list)
    model = torch.load(MODEL_PATH+'/upmodel_{0}_seed={1}.bin'.format(action, myseed))
    pred_ans = model.predict(test_model_input, 128)
    #submit[action] = pred_ans
    torch.cuda.empty_cache()
    return pred_ans

def main():
    submit = pd.read_csv(TEST_FILE)[['userid', 'feedid']]
    for action in ACTION_LIST:
        for myseed in [2021,1024,121]:
            pred_ans=task(action,myseed)
            submit[action] = pred_ans
            submit.to_csv(SUBMIT_DIR+"/difm_{0}_{1}.csv".format(action, myseed), index=False)
if __name__ == "__main__":
    main()