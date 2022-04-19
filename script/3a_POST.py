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
new = pickle.load(open('oc3_new.mdl', 'rb'))

# %%
holdout = len(test["Date"].unique())

# %%
# testで予測
gc.collect()
df_te_new = pd.concat([test,pd.DataFrame(new.predict(test)).rename(columns = {'oc3r': 'pred'})], axis = 1)

# %%
#bestモデルを読んで予測。読めない、あるいは説明データに対応できない場合は、新モデルで上書きする。
try:
    pre = pickle.load(open('oc3_best.mdl', 'rb'))
    df_te_pre = pd.concat([test,pd.DataFrame(pre.predict(test)).rename(columns = {'oc3r': 'pred'})], axis = 1)
    slack('旧モデル予測完了')
except:
    slack('旧モデル読み込みおよび予測失敗、新モデルで上書き')
    pre = new
    df_te_pre = df_te_new
    pickle.dump(new, open('oc3_best.mdl', 'wb'))

# %%
def get_best(df, n):
    return df.sort_values("pred", ascending=False).head(n)
def get_worst(df, n):
    return df.sort_values("pred", ascending=False).tail(n)

# %%
score_new = df_te_new.groupby("Date").apply(get_best, 1)["oc3"].mean()
score_pre = df_te_pre.groupby("Date").apply(get_best, 1)["oc3"].mean() 

if score_new >= score_pre:
    pickle.dump(new, open('ag_model_best.mdl', 'wb'))
    print("モデル更新完了")
    score_best = score_new
    slacker = "モデル更新完了" + str(score_best)
else:
    score_best = score_pre
    slacker = "モデルそのまま" + str(score_best)
# score_new = r2_score(df_te_new["RATE"], df_te_new["pred"])
# score_pre = r2_score(df_te_pre["RATE"], df_te_pre["pred"])

slack("score_new = " + str(score_new))
slack("score_pre = " + str(score_pre))

# %%
def visu(df, name, score):
    fig = plt.figure(figsize = (15,7), facecolor="white")
    ilist = [1, 2, 5, 10, 20, 50, 100, 200]
    for i in ilist:
        lista = []
        for j in df["Date"].unique():
            dfx = df[df["Date"] == j]
            pb = get_best(dfx, i)[["Date", "oc3"]].groupby("Date").mean().reset_index()
            lista.append(pb)
        pb = pd.concat(lista)
        pb["oc3"] = pb["oc3"].cumsum()
        plt.plot(pb["Date"], pb["oc3"])
    for i in ilist:
        lista = []
        for j in df["Date"].unique():
            dfx = df[df["Date"] == j]
            pb = get_worst(dfx, i)[["Date", "oc3"]].groupby("Date").mean().reset_index()
            lista.append(pb)
        pb = pd.concat(lista)
        pb["oc3"] = pb["oc3"].cumsum()
        plt.scatter(pb["Date"], pb["oc3"], s = 5)
    plt.legend(ilist + ilist)
    plt.title("score =" + str(score))
    fig.savefig(name + ".png")

# %%
visu(df_te_new, "pn_new", score_new)

# %%
visu(df_te_pre, "pn_pre", score_pre)

# %%



