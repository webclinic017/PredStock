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
os.chdir('/home/toshi/PROJECTS/PredStock')

# %%
test = pd.read_csv("test.csv")

# %%
new = pickle.load(open('ag_model_new.mdl', 'rb'))

# %%
# testで予測
gc.collect()
try:
    df_te_new = pd.concat([test,new.predict_proba(test).rename(columns = {1: 'pred'})['pred']], axis = 1)
except:
    df_te_new = pd.concat([test,pd.DataFrame(new.predict(test)).rename(columns = {'RATE': 'pred'})], axis = 1)

# %%
#bestモデルを読んで予測。読めない、あるいは説明データに対応できない場合は、新モデルで上書きする。
try:
    pre = pickle.load(open('ag_model_best.mdl', 'rb'))
    try:
        df_te_pre = pd.concat([test,pre.predict_proba(test).rename(columns = {1: 'pred'})['pred']], axis = 1)
    except:
        df_te_pre = pd.concat([test,pd.DataFrame(pre.predict(test)).rename(columns = {'RATE': 'pred'})], axis = 1)
    slack('旧モデル予測完了')
except:
    slack('旧モデル読み込みおよび予測失敗、新モデルで上書き')
    pre = new
    df_te_pre = df_te_new
    pickle.dump(new, open('ag_model_best.mdl', 'wb'))

# %%
plt.scatter(df_te_new["RATE2"], df_te_new["pred"])

# %%
plt.scatter(df_te_pre["RATE2"],df_te_pre["pred"])

# %%
n = holdout * 2

def evalu_profit(df, thre, n):
    score = (df[df['pred'] >= thre]['RATE2'].mean()) * min(n, df[df['pred'] >= thre]['RATE2'].count())
    return score

def optima(df, n):
    score = -math.inf
    mmax = 1000
    thre_max = df['pred'].max()
    thre_min = df['pred'].min()
    for m in range(mmax):

        thre = thre_min + (thre_max - thre_min) * m/mmax
        raw_score = evalu_profit(df, thre, n)
        if raw_score > score:
            score = raw_score
            thre_out = thre
    score = math.exp(score / n) - 1
    return score, thre_out

score_new, thre_new = optima(df_te_new, n)
score_pre, thre_pre = optima(df_te_pre, n)

# %%
slack('score_new =' + str(score_new) + ', thre_new =' + str(thre_new))
slack('score_pre =' + str(score_pre) + ', thre_pre =' + str(thre_pre))

# %%
df_te_new[df_te_new['pred'] >= thre_new].plot.scatter(x = 'RATE2', y = 'pred', xlim = (-0.2, 0.2))

# %%
df_te_pre[df_te_pre['pred'] >= thre_pre].plot.scatter(x = 'RATE2', y = 'pred', xlim = (-0.2, 0.2))

# %%
#新旧モデル可視化
fig = plt.figure(figsize = (10,6), facecolor="white")
plt.hist(df_te_new[df_te_new['pred'] >= thre_new]['RATE2'], bins=60, range = (-0.3,0.3), alpha = 0.5)
plt.hist(df_te_pre[df_te_pre['pred'] >= thre_pre]['RATE2'], bins=60, range = (-0.3,0.3), alpha = 0.5)
plt.title('new=' + str(score_new) +' / pre='+ str(score_pre))
plt.grid()
fig.savefig("hist_positive_ag.png")

if score_new >= score_pre:
    pickle.dump(new, open('ag_model_best.mdl', 'wb'))
    print("モデル更新完了")
    score_best = score_new
    thre_best = thre_new
    slacker = "モデル更新完了" + str(score_best)
else:
    score_best = score_pre
    thre_best = thre_pre
    slacker = "モデルそのまま" + str(score_best)

# %%
ag_result = pd.DataFrame([[score_new, thre_new], [score_pre, thre_pre], [score_best, thre_best]], columns = ['score', 'thre'], index = ['new', 'pre', 'best'])
print(ag_result)
ag_result.to_csv('ag_result.csv')

# %%
slack(slacker)

# %%



