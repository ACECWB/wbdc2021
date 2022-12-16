import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, './model'))
sys.path.append(os.path.join(BASE_DIR, '../config'))
import pickle
from config import *
from my_mmoe import MMOE
import numpy as np
import pandas as pd
import torch
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names,VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
import time
import datatable as dt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("test_path", type=str, help='test_path')
args = parser.parse_args()
test_path = args.test_path
print(test_path)

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_dict = {
    'bgm_song_id': 25158+1,
    'bgm_singer_id': 17499+1,
    'userid': 199999,
#     'feedid': 106444,# 112871+1
    'feedid':112871+1,
    'authorid': 18788+1,
    'device' : 3
}

dense_features = [
 'videoplayseconds',
]

batch_size = 1024
embedding_dim = 100

target = ['read_comment', 'like', 'click_avatar', 'forward', 'comment', 'favorite', 'follow']
sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']

pkl = open(FEATURE_PATH + '/user_encoder.pkl', 'rb')
userid_map = pickle.load(pkl)
pkl.close()
pkl = open(FEATURE_PATH + '/feedid_encoder.pkl', 'rb')
feedid_map = pickle.load(pkl)
pkl.close()

feed = dt.fread(FEED_INFO)
feed = feed.to_pandas()
test = dt.fread(test_path)
test = test.to_pandas()
test = test.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
                      on='feedid')
mms = MinMaxScaler(feature_range=(0, 1))

feed[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
    feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
feed['bgm_song_id'] = feed['bgm_song_id'].astype('int64')
feed['bgm_singer_id'] = feed['bgm_singer_id'].astype('int64')
test[dense_features] = test[dense_features].fillna(0, )
test[dense_features] = mms.fit_transform(test[dense_features])
test['userid'] = userid_map.transform(test['userid'])

user_emb1 = np.load(FEATURE_PATH + '/user_emb_normal_100.npy')
user_emb1 = torch.from_numpy(user_emb1).float().to(device)
user_emb2 = np.load(FEATURE_PATH + '/user_emb_adjust_100.npy')
user_emb2 = torch.from_numpy(user_emb2).float().to(device)
feed_emb = np.load(FEATURE_PATH + '/feed_embedding_100.npy')
feed_emb = torch.from_numpy(feed_emb).float().to(device)

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=vocab_dict[feat] + 1, embedding_dim=embedding_dim)
                      for feat in vocab_dict.keys()] + [DenseFeat(feat, 1) for feat in dense_features]

fixlen_feature_columns += [SparseFeat('feed_embedding', vocab_dict['feedid'], embedding_dim=embedding_dim)]
fixlen_feature_columns += [SparseFeat('user_embedding_normal', vocab_dict['userid'], embedding_dim=embedding_dim)]
fixlen_feature_columns += [SparseFeat('user_embedding_adjust', vocab_dict['userid'], embedding_dim=embedding_dim)]

dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(dnn_feature_columns)

for myseed in [1024, 21, 512]:
# for myseed in [1024, 21, 512, 622, 1229]:
    test_model_input = {name: test[name] for name in feature_names if 'hist' not in name and 'seq_len'not in name and name not in ['svd_feed', 'svd_user', 'seq_len', 'feed_emb', 'feed_embedding', 'user_embedding_normal', 'user_embedding_adjust', 'hist_tagids', 'hist_keyids', 'tagids', 'keyids']}
    test_model_input['user_embedding_normal'] = test_model_input['userid']
    test_model_input['user_embedding_adjust'] = test_model_input['userid']
    test_model_input['feed_embedding'] = test_model_input['feedid']
    print('use features: ', feature_names)
#     if myseed==21:
#         device = 'cpu'
    train_model = MMOE(dnn_feature_columns, history_feature_list=[], num_tasks=7, num_experts=13, expert_dim=128, dnn_hidden_units=(512, 256, 64),
                   task_dnn_units=(128, 128, 64), seed = myseed,
                   tasks=['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary'], device=device)
    train_model.compile("adagrad", loss='binary_crossentropy')
    train_model.load_state_dict(torch.load(MODEL_PATH + '/mmoe_{0}.pkl'.format(myseed)))
    
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 10)
    pred_ans = pred_ans.transpose()
    for i, action in enumerate(target):
        test[action] = pred_ans[i]
    test['userid'] = userid_map.inverse_transform(test['userid'])
    test[['userid', 'feedid'] + target].to_csv(SUBMIT_DIR + '/res_{0}.csv'.format(myseed), index=None, float_format='%.6f')
    test['userid'] = userid_map.transform(test['userid'])
    print('to_csv ok')

sub1 = pd.read_csv(SUBMIT_DIR + '/res_1024.csv')
sub2 = pd.read_csv(SUBMIT_DIR + '/res_21.csv')
sub3 = pd.read_csv(SUBMIT_DIR + '/res_512.csv')
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'comment', 'favorite', 'follow']
sub = sub1.copy()
for y in y_list:
    sub[y] = sub1[y] *0.3 + sub2[y]*0.3 + sub3[y]*0.4
sub.to_csv(SUBMIT_DIR + '/result.csv', index=False)
    


