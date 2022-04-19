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
import datetime
import pickle
# from prophet import Prophet
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import timedelta

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
test = pq.read_table("01_PROC/tsfeature.parquet").to_pandas().dropna()
test["Date"] = pd.to_datetime(test["Date"])

# %%
date = test["Date"].drop_duplicates()

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
dffuture3 = dffuture2.sort_values("pred")[["code", "pred", "Date"]].reset_index(drop=True).head(2)
date = test['Date'].tail(1).item()
strdate = str(datetime.date(date.year, date.month, date.day))

# %%
try:
    # dfown = pd.read_csv("dfown.csv")
    own = pq.read_table("01_PROC/own.parquet").to_pandas()
except:
    pq.write_table(pa.Table.from_pandas(dffuture3), "01_PROC/own.parquet")
    own = dffuture3


# %%
own = pd.concat([own, dffuture3]).drop_duplicates()
pq.write_table(pa.Table.from_pandas(own), "01_PROC/own.parquet")

# %%
own

# %%
dffuture3

# %%
strdate = str(datetime.date(date.year, date.month, date.day))
s = strdate + "\n"
for i in range(len(dffuture3)):
    s += "Buy "+ dffuture3["code"][i] + "\n"

# %%
slack(s)

# %%



