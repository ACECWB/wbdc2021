import pandas as pd
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../config'))
from config import *

r1 = pd.read_csv(SUBMIT_DIR + '/b_10fold.csv')
r2 = pd.read_csv(SUBMIT_DIR + '/b_5fold.csv')

nn1 = pd.read_csv(SUBMIT_DIR + '/difm_2021.csv')
nn2 = pd.read_csv(SUBMIT_DIR + '/difm_1024.csv')
nn3 = pd.read_csv(SUBMIT_DIR + '/difm_121.csv')

r0 = r1.copy()
r1_rate = 0.9
r2_rate = 0.1
for action in r1.columns[2:]:
    r0[action] = r1_rate * r1[action] + r2_rate * r2[action]

nn0 = nn1.copy()
nn0['read_comment'] = 0.4 * nn1['read_comment'] + 0.3 * nn2['read_comment'] + 0.3 * nn3['read_comment']
r0['read_comment'] = r0['read_comment'] * 0.2 + nn0['read_comment'] * 0.8
r0.to_csv(SUBMIT_DIR + '/result.csv', index=False)



