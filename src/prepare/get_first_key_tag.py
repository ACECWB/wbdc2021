import pandas as pd
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from config import *

df= pd.read_csv(DATASET_PATH + '/feed_info.csv')

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

feed = feed[df.columns.tolist() + ['manual_tag_0']]

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

feed2 = feed2[feed.columns.tolist() + ['manual_key_0']]
feed2.to_csv(FEATURE_PATH + '/feed_info_first_key_tag.csv')
