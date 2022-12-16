import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import gc
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from config import *

pd.set_option('display.max_columns', None)
np.random.seed(0)

class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)  # 贝塔分布
        I = []
        C = []
        for click_ratio in sample:  # Beta分布生成的click_ratio
            imp = random.random() * imp_upperbound
            # imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''用不动点迭代法 更新Beta里的参数 alpha和beta'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration 不动点迭代x_i+1 = g(x_i)'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            # special.digamma(z)是 在z处取gamma函数值 再求log
            sumfenzialpha += (special.digamma(success[i] + alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i] - success[i] + beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i] + alpha + beta) - special.digamma(alpha + beta))
        return alpha * (sumfenzialpha / sumfenmu), beta * (sumfenzibeta / sumfenmu)

    def update_from_data_by_moment(self, tries, success):  # tries是各组总样本数，success是各组click=1的样本和
        '''用矩估计 更新Beta里的参数 alpha和beta'''
        mean, var = self.__compute_moment(tries, success)
        # print 'mean and variance: ', mean, var
        # self.alpha = mean * (mean*(1-mean)/(var+0.000001) - 1)
        self.alpha = (mean + 0.000001) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)
        # self.beta = (1-mean) * (mean*(1-mean)/(var+0.000001) - 1)
        self.beta = (1.000001 - mean) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''矩估计'''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i]) / (tries[i] + 0.000000001))
        mean = sum(ctr_list) / len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr - mean, 2)  # 方差

        return mean, var / (len(ctr_list) - 1)


def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


## 从官方baseline里面抽出来的评测函数

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


y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']

max_day = 15
## 读取训练集
train = pd.read_csv(USER_ACTION)
print(train.shape)
for y in y_list:
    print(y, train[y].mean())

## 读取测试集
test = pd.read_csv(TEST_FILE)
test['date_'] = max_day
print(test.shape)

## 合并处理
df = pd.concat([train, test], axis=0, ignore_index=True)
print(df.head(3))

## 读取视频信息表
feed_info = pd.read_csv(FEATURE_PATH + '/feed_info_first_key_tag.csv')
## 此份baseline只保留这三列
feed_info = feed_info[[
    'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id','videoplayseconds', 'manual_tag_0', 'manual_key_0'
]]

df = df.merge(feed_info, on='feedid', how='left')
## 视频时长是秒，转换成毫秒，才能与play、stay做运算
df['videoplayseconds'] *= 1000
## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')
df['play_times'] = df['play'] / df['videoplayseconds']
play_cols = [
    'is_finish', 'play_times', 'play', 'stay'
]

## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
n_day = 5
for stat_cols in tqdm([
    ['userid'],
    ['feedid'],
    ['manual_tag_0'],
    ['manual_key_0'],
    ['authorid'],
    ['bgm_singer_id'],
    ['bgm_song_id'],
    ['userid', 'authorid'],
    ['userid', 'manual_tag_0']
]):

    f = '_'.join(stat_cols)
    stat_df = pd.DataFrame()
    for target_day in range(2, max_day + 1):
        left, right = max(target_day - n_day, 1), target_day - 1
        tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
        tmp['date_'] = target_day
        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')

        g = tmp.groupby(stat_cols)
        tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')
        feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]

        for x in play_cols[1:]:
            for stat in ['max', 'mean']:
                tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))

        for y in y_list[:4]:
            tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')
            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
            tmp['{}_{}day_{}_count'.format(f, n_day, y)] = g[y].transform('count')  # 用于计算平滑后的rate

            # rate加入到feats中
            feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y),
                          '{}_{}day_{}_count'.format(f, n_day, y)])

        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        del g, tmp
    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
    del stat_df
    gc.collect()


## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行
for f in tqdm(['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'manual_tag_0', 'manual_key_0']):
    df[f + '_count'] = df[f].map(df[f].value_counts())

## 穿越
for f1, f2 in tqdm([
    ['userid', 'feedid'],
    ['userid', 'authorid'],
    ['userid', 'manual_tag_0']
]):
    df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')
    df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')

## 穿越
for f1, f2 in tqdm([
    ['userid', 'authorid'],
    ['userid', 'manual_tag_0']
]):

    df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')
    df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)
    df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)

df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')
df['videoplayseconds_in_userid_mean_diff_himself'] = df['videoplayseconds'] -  df['videoplayseconds_in_userid_mean']

df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')
df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique')

for stat_cols in tqdm([
    ['userid'],
    ['feedid'],
    ['authorid'],
    ['userid', 'authorid'],
    ['manual_tag_0'],
    ['manual_key_0'],
    ['userid', 'manual_tag_0']
]):

    f = '_'.join(stat_cols)
    for y in y_list[:4]:  # 利用先前1-14天计算好的count、mean来算rate
        df['{}_5day_{}_count'.format(f, y)] = df['{}_5day_{}_count'.format(f, y)].fillna(0)  # 全部用0填充、平滑后求平均有值
        df['{}_5day_{}_mean'.format(f, y)] = df['{}_5day_{}_mean'.format(f, y)].fillna(0)

        HP = HyperParam(1, 1)
        HP.update_from_data_by_moment(df['{}_5day_{}_count'.format(f, y)].values,
                                      df['{}_5day_{}_mean'.format(f, y)].values)
        df['{}_5day_{}_rate'.format(f, y)] = (df['{}_5day_{}_mean'.format(f, y)].values + HP.alpha) / (
                    df['{}_5day_{}_count'.format(f, y)].values + HP.alpha + HP.beta)

# df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])
df.to_csv(FEATURE_PATH + '/b_data_252.csv', index=False)

