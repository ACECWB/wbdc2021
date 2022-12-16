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
emb_cols = [['userid', 'feedid']]
sort_df = data.sort_values('date_').reset_index(drop=True)
# emb adjust
emb_size = 50
tmp = sort_df.groupby('userid', as_index=False)['feedid'].agg({'{}_{}_list'.format('userid', 'feedid'): list})
sort_df['action'] = sort_df['read_comment'] + sort_df['like'] + sort_df['click_avatar'] + sort_df['forward'] + sort_df['follow'] + sort_df['favorite'] + sort_df['comment']
f1 = 'userid'
f2 = 'feedid'
emb_size = 50
tmp = sort_df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
date = sort_df.groupby(f1, as_index=False)['date_'].agg({'{}_{}_list'.format(f1, 'date'): list})
click = sort_df.groupby(f1, as_index=False)['action'].agg({'{}_{}_list'.format(f1, 'action'): list})
sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
for i in tqdm(range(len(sentences))):
    sentences[i] = [str(x) for x in sentences[i]]

def get_weighted_click(clicklist):
    return (np.array(clicklist) + 1).tolist()
click['userid_action_list'] = click['userid_action_list'].apply(get_weighted_click)
print(click)
model = Word2Vec(sentences, vector_size=emb_size, window=6, min_count=5, sg=0, hs=0, seed=1)
def get_softmax_weight(weight):
    max_w = max(weight)
    np_w = np.array(weight) - max_w
    x_exp = np.exp(np_w)
    x_sum = np.sum(x_exp)
    res = x_exp / x_sum
    return res

index_dict = {}
emb_matrix = []
for i in tqdm(range(len(sentences))):
    seq = sentences[i]
    vec = []
    vec_w = []
    seq_w = click['userid_action_list'].values[i]
    for w_i in range(len(seq)):
        w = seq[w_i]
        w_w = seq_w[w_i]
        if w in model.wv.index_to_key:
            vec.append(model.wv[w])
            vec_w.append(w_w)
    if len(vec) > 0:
#         emb_matrix.append(np.mean(vec, axis=0))
        # 使用softmax权重，date/14 * (action+1)
        softmax_weight = get_softmax_weight(vec_w).reshape(-1, 1)
        emb_matrix.append(np.sum((np.array(vec) * softmax_weight), axis=0).tolist())
    else:
        emb_matrix.append([0] * emb_size)
    index_dict[tmp[f1][i]] = i
emb_matrix = np.array(emb_matrix)
for i in range(emb_size):
    tmp['{}_of_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]

for i in range(emb_size):
    tmp['{}_of_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]

# 用user vec mean来作为feed emb
tmp_f2 = sort_df.groupby('feedid', as_index=False)['userid'].agg({'{}_{}_list'.format('feedid', 'userid'): list})
sentences_f2 = tmp_f2['{}_{}_list'.format('feedid', 'userid')].values.tolist()
index_dict_f2 = {}
emb_matrix_f2 = []
for i in tqdm(range(len(sentences_f2))):
    seq = sentences_f2[i]
    vec = []
    for w in seq:
        vec.append(emb_matrix[index_dict[w]])
    if len(vec) > 0:
        emb_matrix_f2.append(np.mean(vec, axis=0))
    else:
        emb_matrix_f2.append([0] * emb_size)
    index_dict_f2[str(tmp_f2[f2][i])] = i
emb_matrix_f2 = np.array(emb_matrix_f2)

emb_matrix_adjust = []
for seq in tqdm(sentences):
    vec = []
    for w in seq:
        vec.append(emb_matrix_f2[index_dict_f2[w]])
    if len(vec) > 0:
        emb_matrix_adjust.append(np.mean(vec, axis=0))
    else:
        emb_matrix_adjust.append([0] * emb_size)
emb_matrix_adjust = np.array(emb_matrix_adjust)
for i in range(emb_size):
    tmp['{}_of_{}_emb_adjust_{}'.format('userid', 'feedid', i)] = emb_matrix_adjust[:, i]


tmp = tmp.drop('{}_{}_list'.format('userid', 'feedid'), axis=1)
word_list = []
emb_matrix2 = []
for w in tqdm(model.wv.index_to_key):
    word_list.append(w)
    emb_matrix2.append(model.wv[w])
emb_matrix2 = np.array(emb_matrix2)
tmp2 = pd.DataFrame()
tmp2[f2] = np.array(word_list).astype('int')
for i in range(emb_size):
    tmp2['{}_emb_{}'.format('feedid', i)] = emb_matrix2[:, i]

tmp.to_csv(FEATURE_PATH + '/user_emb50_weighted_action.csv', index=False)
tmp2.to_csv(FEATURE_PATH + '/feed_emb50_weighted_action.csv', index=False)






