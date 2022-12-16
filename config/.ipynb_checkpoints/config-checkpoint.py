import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(BASE_DIR, '../data')
FEATURE_PATH = os.path.join(ROOT_PATH, 'wedata/feature')
DATASET_PATH = os.path.join(ROOT_PATH, 'wedata/wechat_algo_data2')
USER_ACTION = os.path.join(DATASET_PATH, 'user_action.csv')
FEED_INFO = os.path.join(DATASET_PATH, 'feed_info.csv')
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, 'feed_embeddings.csv')
TEST_FILE = os.path.join(ROOT_PATH, 'wedata/wechat_algo_data2/test_a.csv')
SUBMIT_DIR = os.path.join(ROOT_PATH, 'submission')
FEED_INFO_FRT = os.path.join(FEATURE_PATH, 'feed_info_frt.csv')
MODEL_PATH = os.path.join(ROOT_PATH, 'model')