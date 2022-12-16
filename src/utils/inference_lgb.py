import pandas as pd
import numpy as np
import datatable as dt
import os
import sys
from lightgbm.sklearn import LGBMClassifier
# from sklearn.externals import joblib
import argparse
import joblib
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../config'))
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("fold_cnt", type=int, help='fold count')
args = parser.parse_args()

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

fold = args.fold_cnt
save_path = MODEL_PATH + '/{0}fold_lgb'.format(fold)
my_cols = cols
for y in y_list[:4]:
    preds = 0
    for f in range(fold):
        clf = joblib.load(save_path + '/{}{}_lgb.pkl'.format(y, f))
        preds += clf.predict_proba(test[my_cols])[:, 1] / fold
    test[y] = preds

print(test[['userid', 'feedid'] + y_list[:4]])
test[['userid', 'feedid'] + y_list[:4]].to_csv(
    SUBMIT_DIR + '/b_%dfold.csv' % (fold),
    index=False
)

