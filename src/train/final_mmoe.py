import torch.nn as nn
import pickle
import argparse
import os
import torch
import pandas as pd
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = '.'
sys.path.append(os.path.join(BASE_DIR, '../../config'))
sys.path.append(os.path.join(BASE_DIR, '../model'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from config import *
from time import time
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from sklearn.preprocessing import MinMaxScaler
print(sys.path)

import datatable as dt
from my_mmoe import MMOE
from evaluation import evaluate_deepctr
import pickle
import gc

parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int, help='seed')
args = parser.parse_args()
myseed = args.seed
print(myseed)

# 训练相关参数设置
ONLINE_FLAG = True # 是否准备线上提交

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
epochs = 1
batch_size = 1024
embedding_dim = 100
max_hist_seq_len = 100

target = ['read_comment', 'like', 'click_avatar', 'forward', 'comment', 'favorite', 'follow']
sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']

pkl = open(FEATURE_PATH + '/user_encoder.pkl', 'rb')
userid_map = pickle.load(pkl)
pkl.close()
pkl = open(FEATURE_PATH + '/feedid_encoder.pkl', 'rb')
feedid_map = pickle.load(pkl)
pkl.close()

mms = MinMaxScaler(feature_range=(0, 1))
user_emb1 = np.load(FEATURE_PATH + '/user_emb_normal_100.npy')
user_emb1 = torch.from_numpy(user_emb1).float().to(device)

user_emb2 = np.load(FEATURE_PATH + '/user_emb_adjust_100.npy')
user_emb2 = torch.from_numpy(user_emb2).float().to(device)

feed_emb = np.load(FEATURE_PATH + '/feed_embedding_100.npy')
feed_emb = torch.from_numpy(feed_emb).float().to(device)

hist_file = open(FEATURE_PATH + '/hist_data_action_begin_100.pkl', 'rb')
hist_data = pickle.load(hist_file)
hist_file.close()

if ONLINE_FLAG:
    data = pd.read_csv(FEATURE_PATH + '/online_train_100.csv', iterator=True)
    test = pd.read_csv(FEATURE_PATH + '/online_test_100.csv')
else:
    val = pd.read_csv(FEATURE_PATH + '/offline_val_100.csv')
    data = pd.read_csv(FEATURE_PATH + '/offline_train_100.csv', iterator=True)

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=vocab_dict[feat] + 1, embedding_dim=embedding_dim)
                      for feat in vocab_dict.keys()] + [DenseFeat(feat, 1) for feat in dense_features]

fixlen_feature_columns += [SparseFeat('feed_embedding', vocab_dict['feedid'], embedding_dim=embedding_dim)]
fixlen_feature_columns += [SparseFeat('user_embedding_normal', vocab_dict['userid']+1, embedding_dim=embedding_dim)]
fixlen_feature_columns += [SparseFeat('user_embedding_adjust', vocab_dict['userid']+1, embedding_dim=embedding_dim)]
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(dnn_feature_columns)

if ONLINE_FLAG:
    test_model_input = {name: test[name] for name in feature_names if 'hist' not in name and 'seq_len'not in name and name not in ['svd_feed', 'svd_user', 'seq_len', 'feed_emb', 'feed_embedding', 'user_embedding_normal', 'user_embedding_adjust', 'hist_tagids', 'hist_keyids', 'tagids', 'keyids']}
    test_model_input['user_embedding_normal'] = test_model_input['userid']
    test_model_input['user_embedding_adjust'] = test_model_input['userid']
    test_model_input['feed_embedding'] = test_model_input['feedid']

else:    
    val_model_input = {name: val[name] for name in feature_names if 'hist' not in name and 'seq_len' not in name and name not in ['svd_feed', 'svd_user', 'feed_emb', 'hist_tagids', 'hist_keyids', 'feed_embedding', 'user_embedding_normal', 'user_embedding_adjust', 'tagids', 'keyids']}
    
    val_model_input['feed_embedding'] = val_model_input['feedid']
    val_model_input['user_embedding_normal'] = val_model_input['userid']
    val_model_input['user_embedding_adjust'] = val_model_input['userid']

    userid_list = val['userid'].astype(str).tolist()
    val_labels = [val[y].values for y in target]
    

print('use features: ', feature_names)

# hist_features = ['feedid', 'feed_embedding', 'authorid', 'bgm_singer_id', 'bgm_song_id']
# hist_features = ['feedid', 'feed_embedding']
hist_features = []
train_model = MMOE(dnn_feature_columns, history_feature_list=hist_features, num_tasks=7, num_experts=13, expert_dim=128, dnn_hidden_units=(512, 256, 64),seed = myseed,
                   task_dnn_units=(128, 128, 64),
                   tasks=['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary'], device=device)
print(train_model.feature_index)

train_model.compile("adagrad", loss='binary_crossentropy')
train_model.embedding_dict['user_embedding_normal'] = nn.Embedding.from_pretrained(user_emb1, freeze=False)
train_model.embedding_dict['user_embedding_adjust'] = nn.Embedding.from_pretrained(user_emb2, freeze=False)
train_model.embedding_dict['feed_embedding'] = nn.Embedding.from_pretrained(feed_emb, freeze=False)

loop = True
cnt = 0
while loop:
    try:
        cnt += 1
        print('chunk: ', cnt)
        train = data.get_chunk(2000*10000)           

        train_model_input = {name: train[name] for name in feature_names  if 'hist' not in name and 'seq_len'not in name and name not in ['svd_feed', 'svd_user', 'feed_emb', 'hist_tagids', 'hist_keyids', 'feed_embedding', 'user_embedding_normal', 'user_embedding_adjust', 'tagids', 'keyids']}
    
        train_model_input['feed_embedding'] = train_model_input['feedid']
        train_model_input['user_embedding_normal'] = train_model_input['userid']
        train_model_input['user_embedding_adjust'] = train_model_input['userid']
        train_labels = train[target].values

        for epoch in range(epochs):
            history = train_model.fit(train_model_input, train_labels,
                              batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
        if not ONLINE_FLAG and cnt % 10 == 0:
            val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
            # 模型predict()返回值格式为(?, 4)，与tf版mmoe不同。因此下方用到了transpose()进行变化。
            evaluate_deepctr(val_labels, val_pred_ans.transpose(), userid_list, target)
            
#         torch.save(train_model.state_dict(), MODEL_PATH + '/mmoe_{0}.pkl'.format(myseed))
        
    except StopIteration:
        loop=False
        print('Finished all train')

# if ONLINE_FLAG:    
torch.save(train_model.state_dict(), MODEL_PATH + '/mmoe_{0}.pkl'.format(myseed))

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
    test[['userid', 'feedid'] + target].to_csv(SUBMIT_DIR + '/my_mmoe_{0}.csv'.format(myseed), index=None, float_format='%.6f')
    print('to_csv ok')
    