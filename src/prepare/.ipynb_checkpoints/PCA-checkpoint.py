from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from config import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('comp', type=int, help='n_components')
args = parser.parse_args()

feed_embedding=pd.read_csv(DATASET_PATH + '/feed_embeddings.csv')
feed_embedd=[]
cnt=0
for i in feed_embedding['feed_embedding'].values:
    cnt+=1
    feed_embedd.append([float(ii) for ii in i.split(' ') if ii!=''])
#
model_pca=PCA(n_components=args.comp)
feed_embedd=model_pca.fit_transform(np.array(feed_embedd))
feed_embedding=pd.concat((feed_embedding,pd.DataFrame(feed_embedd)),axis=1)
feed_embedding.drop(['feed_embedding'],axis=1,inplace=True)
feed_embedding.columns=['feedid']+['feed_embed_'+str(i) for i in range(args.comp)]
feed_embedding.to_csv(FEATURE_PATH + '/feed_embeddings_PCA{}.csv'.format(args.comp),index=False)