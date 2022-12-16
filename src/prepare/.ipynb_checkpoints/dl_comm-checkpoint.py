# coding: utf-8
import os
import time
import logging 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)
import numpy as np
import pandas as pd
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from config import *
END_DAY = 15
SEED = 2021

# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.2, "forward": 0.2,
         "comment": 0.2, "follow": 0.2, "favorite": 0.2}
# 各个阶段数据集的设置的最后一天
STAGE_END_DAY = {"online_train": 14, "offline_train": 13, "evaluate": 14, "submit": 15}
# 构造训练数据的天数
BEFOR_DAY=5
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
FEED_INFO_FRT_LIST=['authorid_count', 'vPlaysecBucket_count', 'authorid_feedid_nunique', 
        'authorid_vPlaysecBucket_nunique', 'vPlaysecBucket_feedid_nunique', 'vPlaysecBucket_authorid_nunique']
# FEA_FEED_LIST=FEA_FEED_LIST+FEED_INFO_FRT_LIST

def group_fea(df,key,target):
    tmp=df.groupby(key,as_index=False)[target].agg(
        {key+'_'+target+'_nunique':'nunique',
        }).reset_index().drop('index',axis=1)
    FEED_INFO_FRT_LIST.append(key+'_'+target+'_nunique')
    return tmp
def make_feed_info_frt():
    #------处理feed info 构造类别特征的count、nounique特征
    CAT_FEATURE_LIST=['feedid','authorid','vPlaysecBucket']
    feed_info=pd.read_csv(FEED_INFO)
    feed_info=feed_info.assign(vPlaysecBucket=pd.cut(feed_info.videoplayseconds,[float('-inf'),
            10,20,40,55,70,float('inf')],
            labels=[0,1,2,3,4,5]))
    #
    feed_info['vPlaysecBucket']=feed_info['vPlaysecBucket'].astype(int)
    #------feed构造count特征------
    for f in CAT_FEATURE_LIST:
        if f!='feedid':
            tmp=feed_info[f].map(feed_info[f].value_counts())
            feed_info[f+'_count']=tmp
            FEED_INFO_FRT_LIST.append(f+'_count')
    #----------feed构造unique特征---
    for f in CAT_FEATURE_LIST[1:]:
        for g in CAT_FEATURE_LIST:
            if f!=g:
                tmp=group_fea(feed_info,f,g)
                feed_info=feed_info.merge(tmp,on=f,how='left')
    #
    print(FEED_INFO_FRT_LIST)
    feed_info.to_csv(FEED_INFO_FRT,index=False)

def statis_feature(start_day=1, before_day=5, agg=['mean','sum','count']):
    """
    统计用户/feed 过去n天各类行为的次数
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数）
    :param agg: String. 统计方法
    """

    print("统计ctr特征-----------------")
    history_data = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    history_data.sort_values(by="date_",inplace=True)
    feature_dir = FEATURE_PATH
    for dim in ["userid", "feedid"]:
        print(dim)
        user_data = history_data[[dim, "date_"] + FEA_COLUMN_LIST]
        res_arr = []
        tmp_name='_'+dim+'_'
        for start in range(2,before_day+1):
            temp=user_data[user_data['date_']<=start]
            temp=temp.drop(columns=['date_'])
            temp=temp.groupby([dim]).agg(agg).reset_index()
            temp.columns=[dim]+list(map(tmp_name.join,temp.columns.values[1:]))
            temp['date_']=start
            res_arr.append(temp)

        for start in range(start_day, END_DAY-before_day+1):
            temp = user_data[((user_data["date_"]) >= start) & (user_data["date_"] < (start + before_day))]
            temp = temp.drop(columns=['date_'])
            temp = temp.groupby([dim]).agg(agg).reset_index()
            temp.columns=[dim]+list(map(tmp_name.join,temp.columns.values[1:]))
            temp['date_']=start+before_day
            res_arr.append(temp)
        dim_feature = pd.concat(res_arr)
        feature_path = os.path.join(feature_dir, dim+"_feature.csv")
        print('Save to: %s'%feature_path)
        dim_feature.to_csv(feature_path, index=False)

def merge_frt(df,mode):
    statis_feed_frt=pd.read_csv(os.path.join(FEATURE_PATH, "feedid_feature.csv"))
    statis_user_frt=pd.read_csv(os.path.join(FEATURE_PATH, "userid_feature.csv"))
    if mode=='test':
        df['date_']=15
        statis_feed_frt=statis_feed_frt[statis_feed_frt['date_']==15].reset_index(drop=True)
        statis_user_frt=statis_user_frt[statis_user_frt['date_']==15].reset_index(drop=True)
    if mode=='train':
        df=df[df['date_']>1].reset_index(drop=True)
        statis_feed_frt=statis_feed_frt[statis_feed_frt['date_']<15].reset_index(drop=True)
        statis_user_frt=statis_user_frt[statis_user_frt['date_']<15].reset_index(drop=True)
    #
    print(df.shape,statis_feed_frt.shape)
    df=df.merge(statis_feed_frt, on=["feedid", "date_"], how="left")
    df=df.merge(statis_user_frt, on=["userid", "date_"], how="left")
    print(df.shape)
    return df

def make_sample():
    #根据测试集id分布进行负采样
#     user_action=pd.read_csv(USER_ACTION)
#     test_a=pd.read_csv(TEST_FILE)
#     test_a_user_ids=test_a.userid.unique()
#     test_a_feed_ids=test_a.feedid.unique()
#     test_off_user_ids=user_action[user_action['date_']==14].userid.unique()
#     test_off_feed_ids=user_action[user_action['date_']==14].feedid.unique()
#     test_select_user_ids=list(set(test_a_user_ids).union(set(test_off_user_ids)))
#     test_select_feed_ids=list(set(test_a_feed_ids).union(set(test_off_feed_ids)))
    #
    #feed信息
    feed_info_df = pd.read_csv(FEED_INFO)
#     feed_info_df=pd.read_csv(FEED_INFO_FRT)
    #user行为数据
    user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid","device"] + FEA_COLUMN_LIST]
    #原始feed enmbedd数据
#     feed_embed = pd.read_csv(FEED_EMBEDDINGS)
    #测试数据
    test = pd.read_csv(TEST_FILE)
    # add feed feature
    train = pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    test = pd.merge(test, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    test["videoplayseconds"] = np.log(test["videoplayseconds"] + 1.0)
#     test=merge_frt(test,mode='test')
    test.to_csv(FEATURE_PATH + f'/test_data.csv', index=False)
    print("测试集前几行:")
    print(test.head(5))
    train.to_csv(FEATURE_PATH + f'/train_data.csv', index=False)
#     for action in ACTION_LIST:
#         print(f"prepare data for {action}")
#         tmp = train.drop_duplicates(['userid', 'feedid', action], keep='last')
#         df_neg = tmp[tmp[action] == 0]
#         #如果用lgb，可以采样，用nn就别采了
#         # df_neg=tmp[(tmp[action] == 0)
#         #     & (tmp['userid'].isin(test_select_user_ids))
#         #     & (tmp['feedid'].isin(test_select_feed_ids))
#         # ]
#         #df_neg = df_neg.sample(frac=ACTION_SAMPLE_RATE[action], random_state=42, replace=False)
#         df_all = pd.concat([df_neg, tmp[tmp[action] == 1]])
#         df_all["videoplayseconds"] = np.log(df_all["videoplayseconds"] + 1.0)
#         df_all=merge_frt(df_all,mode='train')
#         df_all.to_csv(FEATURE_PATH + f'/train_data_for_{action}.csv', index=False)
#         print("{} 训练集前几行:".format(action))
#         print(df_all.head(5))
#
def main():
    t = time.time()
#     make_feed_info_frt()
#     logger.info('Generate statistic feature')
#     statis_feature(start_day=1, before_day=BEFOR_DAY, agg=['mean','sum','count'])
    make_sample()
    print('Time cost: %.2f s'%(time.time()-t))


if __name__ == "__main__":
    main()