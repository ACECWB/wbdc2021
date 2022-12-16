import pandas as pd
import datatable as dt
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from config import *


if not os.path.exists(FEATURE_PATH):
    os.mkdir(FEATURE_PATH)

df = dt.fread(DATASET_PATH + '/feed_info.csv')
df = df.to_pandas()
manual_tag_list = []
idmap = {'nan': 0}
cnt = 1

for tag_list in df['manual_tag_list'].values:
#     print(tag_list)
    if type(tag_list) == float:
        manual_tag_list.append([0])
    else:
        tmp = []
        for tag in tag_list.split(';'):
            if tag != '':
                if tag not in idmap:
                    idmap[tag] = cnt
                    cnt += 1
                tmp.append(idmap[tag])
        manual_tag_list.append(tmp)

feed = pd.concat((df, pd.DataFrame(manual_tag_list)), axis=1)
feed.columns = list(df.columns) + ['manual_tag_' + str(index) for index in range(11)]
for i in range(11):
    feed['manual_tag_' + str(i)] = feed['manual_tag_' + str(i)].fillna(0)
    feed['manual_tag_' + str(i)] = feed['manual_tag_' + str(i)].astype('int')

manual_key_list = []
keyidmap = {'nan': 0}
cnt = 1

for key_list in df['manual_keyword_list'].values:
#     print(tag_list)
    if type(key_list) == float:
        manual_key_list.append([0])
    else:
        tmp = []
        for key in key_list.split(';'):
            if key != '':
                if key not in keyidmap:
                    keyidmap[key] = cnt
                    cnt += 1
                tmp.append(keyidmap[key])
        manual_key_list.append(tmp)
feed2 = pd.concat((feed, pd.DataFrame(manual_key_list)), axis=1)

feed2.columns = list(feed.columns) + ['manual_key_' + str(index) for index in range(18)]
for i in range(18):
    feed2['manual_key_' + str(i)] = feed2['manual_key_' + str(i)].fillna(0)
    feed2['manual_key_' + str(i)] = feed2['manual_key_' + str(i)].astype('int')

des_map = {'': 0}
des_cnt = 1
des_list = []

for i in range(len(feed2)):
    line = feed2['description'][i].split()
    line = line[:20]  # 取des的前10个词
    length = len(line)
    tmp = []
    if length == 0:
        tmp.append(0)
    else:
        for j in line:
            if j not in des_map:
                des_map[j] = int(des_cnt)
                des_cnt += 1
            tmp.append(int(des_map[j]))
    des_list.append(tmp)
feed3 = pd.concat((feed2, pd.DataFrame(des_list)), axis=1)
feed3.columns = list(feed2.columns) + ['des_' + str(index) for index in range(20)]
for i in range(20):
    feed3['des_' + str(i)] = feed3['des_' + str(i)].fillna(0)
    feed3['des_' + str(i)] = feed3['des_' + str(i)].astype('int')
feed3.to_csv(FEATURE_PATH + '/feed_info_tags_keys_des.csv', index=False)





