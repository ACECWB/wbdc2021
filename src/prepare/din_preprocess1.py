import pickle
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from config import *
import numpy as np
import pandas as pd

class DIN_preprocess(object):
    def __init__(self):
#         print(sys.path)
        
        self.ROOT = FEATURE_PATH + '/DIN'
        self.RAW = self.ROOT + '/raw'
        self.PROCESSED = self.ROOT + '/processed'
        self.my_vocab_dict = {"count_ocr": 59775,
               "count_asr": 59768,
               "count_des": 41241,
               "count_ocr_char": 20995,
               "count_asr_char": 20870,
               "count_des_char": 20988,
               "count_tag": 350,
               "count_key": 23262}
        
        self.my_len_dict = {"ocr": 21,
               "asr": 21,
               "des": 20,
               "ocr_char": 41,
               "asr_char": 41,
               "des_char": 41,
               "tag": 11,
               "key": 18}
        
        if not os.path.exists(FEATURE_PATH):
            os.mkdir(FEATURE_PATH)
        
        if not os.path.exists(self.ROOT):
            os.mkdir(self.ROOT)
            os.mkdir(self.RAW)
            os.mkdir(self.PROCESSED)
            
        self.data_path = DATASET_PATH
        self.user_action, self.feed_info, self.feed_emb, self.test = self.load()
    
    def load(self):
        user_action = pd.read_csv(self.data_path + '/user_action.csv')
        feed_info = pd.read_csv(self.data_path + '/feed_info.csv')
        feed_emb = pd.read_csv(self.data_path + '/feed_embeddings.csv')
        test = pd.read_csv(self.data_path + '/test_a.csv')
        return user_action, feed_info, feed_emb, test
    
    def get_id_mapping(self, mats, col, begin=1):
        mats = list(map(lambda x: x[col], mats))
        ids = np.concatenate(mats, axis=-1)
        ids = np.unique(ids)
        count = ids.shape[0]
        map_ids = np.arange(begin, begin + count, dtype=np.int32)
        mapid_series = pd.Series(index=ids, data=map_ids)
        return count, mapid_series
    
    def map_id(self, mat, mapping, col):
        mapped_id = mat[col].map(lambda x: mapping[x])
        mat[col] = mapped_id
        return mat
    
    def mapping_all_id(self):
        count_uid, uid_series = self.get_id_mapping([self.user_action, self.test], 'userid', begin=0)
        count_fid, fid_series = self.get_id_mapping([self.feed_info], 'feedid', begin=1)
        count_aid, aid_series = self.get_id_mapping([self.feed_info], 'authorid', begin=1)
        _, songid_series = self.get_id_mapping([self.feed_info], 'bgm_song_id', begin=1)
        _, singerid_series = self.get_id_mapping([self.feed_info], 'bgm_singer_id', begin=1)
        songid_series = songid_series[~songid_series.index.duplicated(keep='first')]
        singerid_series = singerid_series[~singerid_series.index.duplicated(keep='first')]
        count_songid = songid_series.values.shape[0]
        count_singerid = singerid_series.values.shape[0]
        
        self.user_action = self.map_id(self.user_action, uid_series, 'userid')
        self.user_action = self.map_id(self.user_action, fid_series, 'feedid')
        self.feed_info = self.map_id(self.feed_info, fid_series, 'feedid')
        self.feed_info = self.map_id(self.feed_info, aid_series, 'authorid')
        self.feed_info = self.map_id(self.feed_info, songid_series, 'bgm_song_id')
        self.feed_info = self.map_id(self.feed_info, singerid_series, 'bgm_singer_id')
        self.feed_emb = self.map_id(self.feed_emb, fid_series, 'feedid')
        self.test = self.map_id(self.test, fid_series, 'feedid')
        self.test = self.map_id(self.test, uid_series, 'userid')
        
        result = {'count_uid': count_uid, 'count_fid':count_fid, 'count_aid':count_aid, 
                  'count_singerid':count_singerid, 'count_songid':count_songid, 
                  'uid_series':uid_series, 'fid_series':fid_series, 'aid_series':aid_series, 
                  'songid_series':songid_series, 'singerid_series':singerid_series}
        result.update(self.my_vocab_dict)
        return result
    
    def format_mapping(self, df):
        index = pd.Series(df.index, name='src')
        value = pd.Series(df.values, name='des')
        return pd.concat([index, value], axis=1)
    
    def save(self, dict_data):
        with open(self.PROCESSED + '/statistic.pkl', 'wb') as f:
            pickle.dump([dict_data['count_uid'], 
                         dict_data['count_fid'], 
                         dict_data['count_aid'], 
                         dict_data['count_songid'],
                         dict_data['count_singerid'],
                         self.user_action['device'].unique().shape[0]], f, pickle.HIGHEST_PROTOCOL)
        self.format_mapping(dict_data['uid_series']).to_csv(self.RAW + '/uid_mapping.csv', index=False)
        self.format_mapping(dict_data['fid_series']).to_csv(self.RAW + '/fid_mapping.csv', index=False)
        self.format_mapping(dict_data['aid_series']).to_csv(self.RAW + '/aid_mapping.csv', index=False)
        self.format_mapping(dict_data['songid_series']).to_csv(self.RAW + '/songid_mapping.csv', index=False)
        self.format_mapping(dict_data['singerid_series']).to_csv(self.RAW + '/singerid_mapping.csv', index=False)
        
        self.user_action.to_csv(self.RAW + '/user_action.csv', index=False)
        self.feed_info.to_csv(self.RAW + '/feed_info.csv', index=False)
        self.feed_emb.to_csv(self.RAW + '/feed_embeddings.csv', index=False)
        self.test.to_csv(self.RAW + '/test_a.csv', index=False)
if __name__ == '__main__':
    preprocess = DIN_preprocess()
    print('--------------------Finished init--------------------')
    dict_data = preprocess.mapping_all_id()
    print('--------------------Finished mapping all ids--------------------')
    preprocess.save(dict_data)
    print('--------------------successed saving all file & statistic file--------------------')
