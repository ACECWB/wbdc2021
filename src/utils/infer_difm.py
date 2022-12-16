# -*- coding: utf-8 -*-
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, '../config'))
# sys.path.append(os.path.join(BASE_DIR, 'utils'))
# from utils import *

sys.path.append('utils')
from utils.utils import *
import pickle
from config import *
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names,VarLenSparseFeat
# print(sys.path)
from frt import *
from difm_model import MyDIFM
import time
import datatable as dt

ACTION_LIST = ['read_comment', 'like', 'click_avatar', 'forward', 'comment', 'favorite', 'follow']
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
#
NUM_EPOCH_DICT = {"read_comment": 2, "like": 1, "click_avatar": 1,"forward": 1,
                                "comment": 1, "follow": 1, "favorite": 1, }

# NUM_EPOCH_DICT = {"read_comment": 5, "like": 4, "click_avatar": 4,"forward": 4,
#                                 "comment": 4, "follow": 4, "favorite": 4, }

# NUM_EPOCH_DICT = {"read_comment": 3, "like": 2, "click_avatar": 2,"forward": 2,
#                                 "comment": 2, "follow": 2, "favorite": 2, }

#
# NUM_EPOCH_DICT = {"read_comment": 1, "like": 1, "click_avatar": 1,"forward":1,
#                                 "comment": 1, "follow": 1, "favorite": 1, }

def task(action,myseed):
    print('-----------action-----------', action)
    USE_FEAT = [action] + SELECT_FRTS
    dt_test = dt.fread(FEATURE_PATH + '/test_data_info.csv')
    test = dt_test.to_pandas()
    target = [action]
    test[target[0]] = 0
    dt_feed = dt.fread(FEATURE_PATH + '/feed_embeddings_PCA64.csv')
    feed = dt_feed.to_pandas()[feature_list('feed_emb') + ['feedid']]
    test = pd.merge(test, feed, how='left', on='feedid')
    dt_tag_key_des = dt.fread(FEATURE_PATH + '/feed_info_tags_keys_des.csv')
#     tag = dt_tag_key_des.to_pandas()[feature_list('tag') + feature_list('key') + feature_list('des') + ['feedid']]
    tag = dt_tag_key_des.to_pandas()[feature_list('tag') + feature_list('key') + ['feedid']]
    test = pd.merge(test, tag, how='left', on='feedid')
#     dt_ocr_asr_ocrc_asrc_desc = dt.fread(FEATURE_PATH + '/feed_info_splited_nlp.csv')
#     tag = dt_ocr_asr_ocrc_asrc_desc.to_pandas()[
#         feature_list('ocr') + feature_list('asr') + feature_list('ocr_char') + feature_list('asr_char') + feature_list(
#             'des_char') + ['feedid']]
#     test = pd.merge(test, tag, how='left', on='feedid')
    
    map_features = ['feedid', 'userid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    map_dict = {}
    for feat in map_features:
        pickle_file = open(FEATURE_PATH + '/' + feat + '_mapping.pkl', 'rb')
        map_dict[feat] = pickle.load(pickle_file)
        test[feat] = test[feat].fillna(my_vocab_dict[feat] + 1)
        test[feat] = map_dict[feat].transform(test[feat])
        print('finish mapping ', feat)
        pickle_file.close()
 
    dense_features = DENSE_FEATURE
    sparse_features = [i for i in USE_FEAT if i not in dense_features and i not in target]
    print('dense: ', dense_features)
    print('sparse: ', sparse_features)
    
    var_features = list(my_name_dict.keys())
#     var_features.remove('ocr_char')
#     var_features.remove('asr_char')
#     var_features.remove('des_char')
    # var_features.remove('ocr')
    # var_features.remove('asr')
    # var_features.remove('des')
    # var_features.remove('key')
    var_features.remove('feed_emb')
    
    fixlen_feature_columns = [SparseFeat(feat, my_vocab_dict[feat]+1,50)
                        for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]\
                             +[DenseFeat('feed_emb', 64, )]
    
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

    test_model_input = {name: test[name] for name in feature_names
                            if name in SELECT_FRTS}
    for feat in my_name_dict.keys():
        test_model_input[feat] = test[feature_list(feat)].values
#     print(list(test.columns))
    #-------
#     device = 'cpu'
#     use_cuda = True
#     if use_cuda and torch.cuda.is_available():
#         print('cuda ready...')
#         device = 'cuda:0'
    gpus = [0, 1]
    model = torch.load(MODEL_PATH+'/upmodel_noemb_{0}_epoch{1}_seed={2}.bin'.format(action, NUM_EPOCH_DICT[action], myseed))
    model.gpus = gpus
#     model = torch.nn.DataParallel(model, devices)
#     print(test_model_input.keys())
#     model = model.to(device)
    pred_ans = model.predict(test_model_input, 4096*10)
    torch.cuda.empty_cache()
    return pred_ans

def main():
#     for myseed in [2021,1024,121]:
    for myseed in [2021]:

        start = time.time()
        submit = pd.read_csv(TEST_FILE)[['userid', 'feedid']]
        for action in ACTION_LIST:
            pred_ans=task(action,myseed)
            submit[action] = pred_ans
            
        submit.to_csv(SUBMIT_DIR + "/difm_{0}_epoch2_1.csv".format(myseed), index=False)
        spend_time = (time.time() - start)*1000*2000/(4252097*7.0)
        print('seed: {0} spend time: '.format(myseed), spend_time)
if __name__ == "__main__":
    main()