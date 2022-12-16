import pandas as pd
from tqdm import tqdm
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from config import *

feed_info = pd.read_csv(FEATURE_PATH + '/feed_info_tags_keys_des.csv')
word_map = {'nan': 0}
char_map = {'nan': 0}
def get_splited_data(df, col):
    col_list = []
    idmap = char_map if 'char' in col else word_map
    token_num = 40 if 'char' in col else 20
    cnt = 1

    for data_list in df[col].values:

        if type(data_list) == float:
            col_list.append([0])
        else:
            tmp = []
            for data in data_list.split(' '):
                if data != '':
                    if data not in idmap:
                        idmap[data] = cnt
                        cnt += 1
                    tmp.append(idmap[data])
                    if len(tmp) > token_num:
                        break
            col_list.append(tmp)
    return col_list


need_cols = ['ocr', 'asr', 'ocr_char', 'asr_char', 'description_char']
for col in tqdm(need_cols):
    orig = len(feed_info.columns)
    feed_info = pd.concat((feed_info, pd.DataFrame(get_splited_data(feed_info, col))), axis=1)
    after = len(feed_info.columns)
    length = after - orig

    feed_info.columns = list(feed_info.columns[:orig]) + [col + '_' + str(index) for index in range(length)]
    for i in range(length):
        feed_info[col + '_' + str(i)] = feed_info[col + '_' + str(i)].fillna(0)
        feed_info[col + '_' + str(i)] = feed_info[col + '_' + str(i)].astype('int')
print(feed_info)
feed_info.to_csv(FEATURE_PATH + '/feed_info_splited_nlp.csv')



