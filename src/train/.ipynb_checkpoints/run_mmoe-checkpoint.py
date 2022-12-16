import os
import torch
import pandas as pd
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
sys.path.append(os.path.join(BASE_DIR, '../model'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from config import *
from time import time
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from sklearn.preprocessing import MinMaxScaler
import datatable as dt
from mmoe import MMOE
from evaluation import evaluate_deepctr
import pickle

# 训练相关参数设置
ONLINE_FLAG = True  # 是否准备线上提交

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

vocab_dict = {
    'bgm_song_id': 25158,
    'bgm_singer_id': 17499,
    'userid': 199999,
    'feedid': 106444,
    'authorid': 18788,
}

if __name__ == "__main__":
    epochs = 2
    batch_size = 1024
    embedding_dim = 20
    target = ['read_comment', 'like', 'click_avatar', 'forward', 'comment', 'favorite', 'follow']
    tagids = ['manual_tag_' + str(tagid) for tagid in range(11)] # 11
    keyids = ['manual_key_' + str(keyid) for keyid in range(18)] # 18
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    dense_features = ['videoplayseconds', ]
    
    data = dt.fread(USER_ACTION)
    data = data.to_pandas()
    feed = dt.fread(FEED_INFO)
    feed = feed.to_pandas()
    tag = dt.fread(FEATURE_PATH + '/feed_info_tags_keys_des_seq_len.csv')
    tag = tag.to_pandas()[tagids + keyids + ['feedid', 'tag_seq_len', 'key_seq_len']]
    
    pkl = open(FEATURE_PATH + '/user_encoder.pkl', 'rb')
    userid_map = pickle.load(pkl)
    pkl.close()
    
    user_emb1 = np.load(FEATURE_PATH + '/user_emb_normal.npy')
    user_emb1 = torch.from_numpy(user_emb1).float().to(device)
    user_emb2 = np.load(FEATURE_PATH + '/user_emb_adjust.npy')
    user_emb2 = torch.from_numpy(user_emb2).float().to(device)
    
    feed[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    feed['bgm_song_id'] = feed['bgm_song_id'].astype('int64')
    feed['bgm_singer_id'] = feed['bgm_singer_id'].astype('int64')
    data = data.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
                      on='feedid')
    data = data.merge(tag, how='left', on='feedid')
    
    test = dt.fread(TEST_FILE)
    test = test.to_pandas()
    test = test.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
                      on='feedid')
    test = test.merge(tag, how='left', on='feedid')

    data[dense_features] = data[dense_features].fillna(0, )
    test[dense_features] = test[dense_features].fillna(0, )
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    test[dense_features] = mms.fit_transform(test[dense_features])
    data['userid'] = userid_map.transform(data['userid'])
    test['userid'] = userid_map.transform(test['userid'])

    print('data.shape', data.shape)
    print('data.columns', data.columns.tolist())
    print('unique date_: ', data['date_'].unique())

    if ONLINE_FLAG:
        train = data
    else:
        train = data[data['date_'] < 14]
    val = data[data['date_'] == 14]  # 第14天样本作为验证集，当ONLINE_FLAG=False时使用。

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
                          for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
    fixlen_feature_columns += [VarLenSparseFeat(SparseFeat('tagids', vocabulary_size=350 + 1, embedding_dim=embedding_dim), 11, length_name='tag_seq_len')]
    fixlen_feature_columns += [VarLenSparseFeat(SparseFeat('keyids', vocabulary_size=23262 + 1, embedding_dim=embedding_dim), 18, length_name='key_seq_len')]

    
    # fixlen_feature_columns += [SparseFeat(feat, vocabulary_size=350 + 1, embedding_dim=embedding_dim)
    #                           for feat in tagids]
    # fixlen_feature_columns += [SparseFeat(feat, vocabulary_size=23262 + 1, embedding_dim=embedding_dim)
    #                           for feat in keyids]
    # fixlen_feature_columns += [SparseFeat('feed_embedding', 112871+1, embedding_dim=64)]
    fixlen_feature_columns += [SparseFeat('user_embedding_normal', 200000, embedding_dim=20)]
    fixlen_feature_columns += [SparseFeat('user_embedding_adjust', 200000, embedding_dim=20)]

    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)
    print('use features: ', feature_names)

    train_model_input = {name: train[name] for name in feature_names if name not in ['feed_embedding', 'user_embedding_normal', 'user_embedding_adjust', 'tagids', 'keyids']}
    val_model_input = {name: val[name] for name in feature_names if name not in ['feed_embedding', 'user_embedding_normal', 'user_embedding_adjust', 'tagids', 'keyids']}
    # train_model_input['feed_embedding'] = train_model_input['feedid']
    # val_model_input['feed_embedding'] = val_model_input['feedid']
    train_model_input['tagids'] = train[['manual_tag_' + str(index) for index in range(11)]].values
    train_model_input['keyids'] = train[['manual_key_' + str(index) for index in range(18)]].values
    val_model_input['tagids'] = val[['manual_tag_' + str(index) for index in range(11)]].values
    val_model_input['keyids'] = val[['manual_key_' + str(index) for index in range(18)]].values

    train_model_input['user_embedding_normal'] = train_model_input['userid']
    train_model_input['user_embedding_adjust'] = train_model_input['userid']

    val_model_input['user_embedding_normal'] = val_model_input['userid']
    val_model_input['user_embedding_adjust'] = val_model_input['userid']
    
    userid_list = val['userid'].astype(str).tolist()
    test_model_input = {name: test[name] for name in feature_names if name not in ['feed_embedding', 'user_embedding_normal', 'user_embedding_adjust', 'tagids', 'keyids']}
    test_model_input['user_embedding_normal'] = test_model_input['userid']
    test_model_input['user_embedding_adjust'] = test_model_input['userid']
    test_model_input['tagids'] = test[['manual_tag_' + str(index) for index in range(11)]].values
    test_model_input['keyids'] = test[['manual_key_' + str(index) for index in range(18)]].values

    train_labels = train[target].values
    val_labels = [val[y].values for y in target]

    # 4.Define Model,train,predict and evaluate
    train_model = MMOE(dnn_feature_columns, num_tasks=7, num_experts=11, expert_dim=128, dnn_hidden_units=(128, 128, 64),
                       task_dnn_units=(128, 128, 64),
                       tasks=['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary'], device=device)
    train_model.compile("adagrad", loss='binary_crossentropy')
    # print(train_model.summary())
    for epoch in range(epochs):
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=1)
        if not ONLINE_FLAG:
            val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
            # 模型predict()返回值格式为(?, 4)，与tf版mmoe不同。因此下方用到了transpose()进行变化。
            evaluate_deepctr(val_labels, val_pred_ans.transpose(), userid_list, target)

    t1 = time()
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
    pred_ans = pred_ans.transpose()
    t2 = time()
    print('7个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test), (t2 - t1) * 1000.0))
    ts = (t2 - t1) * 1000.0* 2000.0 / (len(test)*7.0) 
    print('7个目标行为2000条样本平均预测耗时（毫秒）：%.3f' % ts)

    # # 5.生成提交文件
    for i, action in enumerate(target):
        test[action] = pred_ans[i]
    test['userid'] = userid_map.inverse_transform(test['userid'])
    test[['userid', 'feedid'] + target].to_csv(SUBMIT_DIR + '/mmoe_cin_multi_user_tag_key.csv', index=None, float_format='%.6f')
    print('to_csv ok')