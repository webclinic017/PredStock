# %%
import pandas as pd
import numpy as np
import random
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
from sklearnex import patch_sklearn
patch_sklearn()

# %%
os.chdir('/home/toshi/PROJECTS/PredStock')

# %%
def slack(txt):
    print(txt)
    try:
        slack = slackweb.Slack(url = pd.read_csv("slackkey.csv")["0"].item())
        slack.notify(text = txt)
    except:
        print("slack_error")

# %%
test = pd.read_csv("test.csv").drop(["RATE", "RATE2", "Unnamed: 0"], axis = 1)

# %%
# 今日の日付のものだけ
test = test[test['DATE'] ==test['DATE'].max()].reset_index(drop = True)

# %%
import pickle
pre = pickle.load(open('ag_model_best.mdl', 'rb'))

# %%
try:
    dffuture2 = pd.concat([test,pre.predict_proba(test).rename(columns = {1: 'pred'})['pred']], axis = 1)
except:
    dffuture2 = pd.concat([test,pd.DataFrame(pre.predict(test)).rename(columns = {'RATE': 'pred'})], axis = 1)

# %%
dffuture2["pred"].hist()

# %%
ag_result = pd.read_csv('ag_result.csv')

# %%
thre = ag_result.tail(1)['thre'].item()
print(thre)

# %%
dffuture3 = dffuture2[dffuture2['pred'] >= thre].sort_values('pred', ascending = False).set_index('code')
date = test['DATE'].tail(1).item()
print(date)
print(dffuture3['pred'])
print('全行程完了')

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



