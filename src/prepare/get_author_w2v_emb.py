import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from config import *

data = pd.read_csv(DATASET_PATH + '/user_action.csv')
feed = pd.read_csv(DATASET_PATH + '/feed_info.csv')
tmp = feed[['authorid', 'manual_tag_list', 'machine_tag_list']]
emb_cols = [['userid', 'feedid']]
sort_df = data.sort_values('date_').reset_index(drop=True)
def get_tag(taglist):
    if type(taglist) == float:
        return taglist
    tag_prob_list = taglist.split(';')
    res = ''
    flag = 1
    for t in tag_prob_list:
        tag, prob = t.split(' ')
        prob = float(prob)
        if prob > 0.05:
            if flag == 1:
                res += tag
                flag = 0
            else:
                res = res + ';' + tag
    return res
tmp['machine_tag_list'] = tmp['machine_tag_list'].apply(get_tag)
tmp['manual_tag_list'].fillna(tmp['machine_tag_list'], inplace=True)
tmp['manual_tag_list'] = tmp['manual_tag_list'].apply(str).map(lambda x: x.split(';'))
new_tmp = tmp.explode('manual_tag_list')
new_tmp = new_tmp[['authorid', 'manual_tag_list']]
emb_size = 20
tmp = new_tmp
tmp = tmp.groupby('authorid', as_index=False)['manual_tag_list'].agg({'{}_{}_list'.format('authorid', 'manual_tag_list'): list})

sentences = tmp['{}_{}_list'.format('authorid', 'manual_tag_list')].values.tolist()
for i in range(len(sentences)):
    sentences[i] = [str(x) for x in sentences[i]]
model = Word2Vec(sentences, vector_size=emb_size, window=3, min_count=1, sg=0, hs=0, seed=1)
emb_matrix = []
for seq in tqdm(sentences):
    vec = []
    for w in seq:
        if w in model.wv.index_to_key:
            vec.append(model.wv[w])
    if len(vec) > 0:
        emb_matrix.append(np.mean(vec, axis=0))
    else:
        emb_matrix.append([0] * emb_size)
emb_matrix = np.array(emb_matrix)
for i in range(emb_size):
    tmp['{}_{}_emb_{}'.format('authorid', 'manual_tag', i)] = emb_matrix[:, i]

tmp = tmp[['authorid'] + ['authorid_manual_tag_emb_' + str(i) for i in range(20)]]
tmp.to_csv(FEATURE_PATH + '/author_tag_emb_20.csv', index=False)






