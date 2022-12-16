import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from config import *
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# after run Preprocess.py
class DIN_Process(object):
    def __init__(self, pca_dim=64):
        self.pca_dim = pca_dim
        self.RAW = FEATURE_PATH + '/DIN/raw'
        self.PROCESSED = FEATURE_PATH + '/DIN/processed'
        self.user_action, self.feed_info, self.feed_emb, self.test = self.load()
        self.use_feats = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id',
                        'videoplayseconds', 'device']
        self.ACTION = ['read_comment', 'like', 'comment', 'click_avatar', 'forward', 'follow', 'favorite']
    
    def load(self):
        user_action = pd.read_csv(self.RAW + '/user_action.csv')
        feed_info = pd.read_csv(self.RAW + '/feed_info.csv')
        feed_emb = pd.read_csv(self.RAW + '/feed_embeddings.csv')
        test = pd.read_csv(self.RAW + '/test_a.csv')
        return user_action, feed_info, feed_emb, test
    
    #总的流程处理
    def process(self):
        self.feature_eng(self.pca_dim)
        print('--------------------Finished feature eng--------------------')
        
        self.train = pd.merge(self.user_action, self.feed_info, on='feedid', how='left')
#         print(self.train)
#         print(self.train.columns)
        self.train = self.train[self.use_feats + self.ACTION + ['date_']]
        self.test = pd.merge(self.test, self.feed_info, on='feedid', how='left')
        self.test = self.test[self.use_feats]
        
        self.train[self.use_feats] = self.train[self.use_feats].fillna(0)
        self.test[self.use_feats] = self.test[self.use_feats].fillna(0)
        print('--------------------Finished merge & fillna--------------------')
        
        self.save_all()
        print('--------------------successed saving all files--------------------')

        
    # 特征工程
    def feature_eng(self, pca_dim):
        self.process_feed_emb(pca_dim)
        self.process_videoplayseconds()
        
    # feed emb降维
    def process_feed_emb(self, pca_dim=None):
        self.feed_emb = self.feed_emb.sort_values(by=['feedid'])
        sorted_embeddings = [np.array(list(map(lambda x: eval(x), emb_list.strip(' ').split(' '))), dtype=np.float32) 
                             for emb_list in self.feed_emb['feed_embedding']]
        sorted_embeddings = [np.expand_dims(emb, axis=0) for emb in sorted_embeddings]
        np_embeddings = np.concatenate(sorted_embeddings, axis=0)
        
        if pca_dim is None:
            result = np_embeddings
        else:
            pca = PCA(n_components=pca_dim)
            pca.fit(np_embeddings)
            result = pca.transform(np_embeddings)
        zero_pad = np.zeros((1, result.shape[-1]), dtype=np.float32)
        self.feed_emb = np.concatenate([zero_pad, result], axis=0)
        
    # 归一化    
    def process_videoplayseconds(self):
        eps = 1e-9
        val = self.feed_info['videoplayseconds']
        norm = (val - val.min()) / (val.max() - val.min() + eps)
        self.feed_info['videoplayseconds'] = norm
    
    def save_all(self):
        self.train.to_csv(self.PROCESSED + '/data.csv', index=False)
        self.test.to_csv(self.PROCESSED + '/test.csv', index=False)
        np.save(self.PROCESSED + '/feed_embeddings{0}.npy'.format(self.pca_dim), self.feed_emb)

if __name__ == '__main__':
    p = DIN_Process(pca_dim=64)
    p.process()