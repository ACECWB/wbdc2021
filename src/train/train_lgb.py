import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
import time
import datatable as dt
from sklearn.model_selection import StratifiedKFold
import argparse
import os
import sys
# from sklearn.externals import joblib
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from config import *


parser = argparse.ArgumentParser()
parser.add_argument("fold_cnt", type=int, help='fold count')
args = parser.parse_args()

def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc
            size += 1.0
    user_auc = float(total_auc) / size
    return user_auc



df = dt.fread(FEATURE_PATH + '/b_data_252.csv')
df = df.to_pandas()

# 加入提取好的feed embedding
FEED_EMBEDDING_DIR=FEATURE_PATH + "/feed_embeddings_PCA64.csv"
df_feed=pd.read_csv(FEED_EMBEDDING_DIR)
df = df.merge(df_feed, on='feedid', how='left')
# 根据action做weighted
weighted_user_emb = pd.read_csv(FEATURE_PATH + '/user_emb20_weighted_action.csv')
df = df.merge(weighted_user_emb, on='userid', how='left')

author_emb = pd.read_csv(FEATURE_PATH + '/author_tag_emb.csv')
df = df.merge(author_emb, on='authorid', how='left')

train = df[~df['read_comment'].isna()].reset_index(drop=True)
test = df[df['read_comment'].isna()].reset_index(drop=True)
trn_x = train[train['date_'] < 14].reset_index(drop=True)
val_x = train[train['date_'] == 14].reset_index(drop=True)
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
play_cols = [
    'is_finish', 'play_times', 'play', 'stay'
]
cols = [f for f in train.columns if f not in ['date_', 'sum_4action', 'manual_tag_list', 'manual_keyword_list'] + play_cols + y_list + ['key_emb_' + str(i) for i in range(50)] + ['tag_emb_' + str(i) for i in range(50)]]
print(cols, len(cols))

feat_dict = {}
##################### 线下验证 #####################
uauc_list = []
r_list = []
fold = args.fold_cnt
skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=1)

for y in y_list[:4]:
    trn0 = trn_x[trn_x[y] == 0].copy()
    trn1 = trn_x[trn_x[y] == 1].copy()
    preds = 0
    feat_dict[y] = 0
    for i, (trn_idx, val_idx) in enumerate(skf.split(trn0, trn0['date_'])):
        trn = pd.concat([trn1, trn0.iloc[val_idx].reset_index(drop=True)], axis=0).reset_index(drop=True)

        print('========= ', y, 'Fold ', i, ' =========')
        my_cols = cols

        t = time.time()
        clf = LGBMClassifier(
            learning_rate=0.05,
            n_estimators=7000,
            num_leaves=256,
            reg_alpha=2.99,
            reg_lambda=1.9,
            max_depth=-1,
            subsample=1,
            colsample_bytree=1,
            random_state=2021,
            metric='None'
        )

        clf.fit(
            trn[my_cols], trn[y],
            eval_set=[(val_x[my_cols], val_x[y])],
            eval_metric='auc',
            #         eval_metric= uauc_metric,
            early_stopping_rounds=100,
            #         num
            verbose=50
        )

        r_list.append(clf.best_iteration_)
        print('runtime: {}\n'.format(time.time() - t))
        preds += clf.predict_proba(val_x[my_cols])[:, 1] / fold

    val_x[y + '_score'] = preds
    val_uauc = uAUC(val_x[y], val_x[y + '_score'], val_x['userid'])
    uauc_list.append(val_uauc)
    print(val_uauc)
    feat_dict[y] += clf.feature_importances_

weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]
print(uauc_list)
print(weighted_uauc)

##################### 全量训练 #####################
r_dict = {}
cnt = 0
for y in y_list[:4]:
    for f in range(fold):
        # r_dict[y+str(f)] = 100
        r_dict[y + str(f)] = r_list[cnt]
        cnt += 1
save_path = MODEL_PATH + '/{0}fold_lgb'.format(fold)
if not os.path.exists(save_path):
    os.mkdir(save_path)

skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=1)
for y in y_list[:4]:
    train0 = train[train[y] == 0].copy()
    train1 = train[train[y] == 1].copy()
    preds = 0
    for i, (trn_idx, val_idx) in enumerate(skf.split(train0, train0['date_'])):
        trn = pd.concat([train1, train0.iloc[val_idx].reset_index(drop=True)], axis=0).reset_index(drop=True)
        print('=========', y, 'Fold ', i, '=========')
        t = time.time()
        my_cols = cols
        clf = LGBMClassifier(
            n_estimators=r_dict[y + str(i)],
            learning_rate=0.05,
            num_leaves=256,
            reg_alpha=2.99,
            reg_lambda=1.9,
            max_depth=-1,
            subsample=1,
            colsample_bytree=1,
            random_state=2021,
            metric='None'
        )
        # print(trn[my_cols], trn[y], r_dict, y+str(i))

        clf.fit(
            trn[my_cols], trn[y],
            eval_set=[(trn[my_cols], trn[y])],
            eval_metric='auc',
            early_stopping_rounds=r_dict[y + str(i)],
            verbose=50
        )
        # print('finish')
        # clf.booster_.save_model(save_path + '/{}{}_lgb.txt'.format(y, i))
        joblib.dump(clf, save_path + '/{}{}_lgb.pkl'.format(y, i))
#         preds += clf.predict_proba(test[my_cols])[:, 1] / fold
        print('runtime: {}\n'.format(time.time() - t))
#     test[y] = preds
#
# print(test[['userid', 'feedid'] + y_list[:4]])
#
# test[['userid', 'feedid'] + y_list[:4]].to_csv(
#     SUBMIT_DIR + '/b_5fold_%.6f_%.6f_%.6f_%.6f_%.6f.csv' % (weighted_uauc, uauc_list[0], uauc_list[1], uauc_list[2], uauc_list[3]),
#     index=False
# )

