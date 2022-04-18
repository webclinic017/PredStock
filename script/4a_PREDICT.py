# %%
import pandas as pd
import numpy as np
import random
import scipy
import math
import os
# import seaborn as sns
import matplotlib.pyplot as plt
import talib
from tqdm import tqdm
import gc
import slackweb

import pickle
# from prophet import Prophet
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import pyarrow as pa
import pyarrow.parquet as pq

# %%
os.chdir('/home/toshi/PROJECTS/PredStock')
from  common import train

# %%
def slack(txt):
    print(txt)
    try:
        slack = slackweb.Slack(url = pd.read_csv("slackkey.csv")["0"].item())
        slack.notify(text = txt)
    except:
        print("slack_error")

# %%
test = pq.read_table("01_PROC/test1.parquet").to_pandas().dropna()
test["Date"] = pd.to_datetime(test["Date"])

# %%
test = test[test["Date"] == test["Date"].max()]

# %%
test

# %%
pre = pickle.load(open('oc3_best.mdl', 'rb'))

# %%
dffuture2 = pd.concat([test,pd.DataFrame(pre.predict(test)).rename(columns = {'oc3r': 'pred'})], axis = 1)

# %%
dffuture2

# %%
dffuture2["pred"].hist()

# %%
dffuture3 = dffuture2.sort_values("pred")[["code", "pred"]].reset_index(drop=True).head(2)
date = test['Date'].tail(1).item()

# %%
if len(dffuture3) == 0:
    print('NO BEST')
    subject_a = str(date) + ' の予報 ベストナシ' 
    body_a = 'NO BEST'
else:
    print('BEST')
    subject_a = str(date) + ' の予報 ' + str(dffuture3.reset_index().loc[:,'code'].iloc[0])
    body_a = 'BEST\n'
    for i  in range(10):
        if len(dffuture3) == i:
            break
        body_a = body_a + '#' + str(i) + ' Code=' + str(dffuture3.reset_index().loc[:,'code'].iloc[i])
        body_a = body_a + '/Score=' + str(dffuture3.loc[:,'pred'].iloc[i]) + '\n'
    body_a = body_a + ' Length = ' + str(len(dffuture3))

# %%
slacker = subject_a + '\n' + body_a

# %%
slack(slacker)

# %%



