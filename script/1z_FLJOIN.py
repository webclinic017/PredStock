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
import datetime
from datetime import timedelta

import gc
import slackweb
import pickle
# from prophet import Prophet
# from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import pyarrow as pa
import pyarrow.parquet as pq

# %%
os.chdir('/home/toshi/PROJECTS/PredStock')
from  common import  dfall

# %%

ngroup = 5
path = "0a_HISTORY/"

# %%
feature = pq.read_table("01_PROC/tsfeature.parquet").to_pandas()
feature["Date"] = pd.to_datetime(feature["Date"])

# %%
feature.head()

# %%
label = pq.read_table("01_PROC/tslabel.parquet").to_pandas()
label["Date"] = pd.to_datetime(label["Date"])

# %%
label

# %%
join = pd.merge(feature, label, on = ["Date", "code"])

# %%
join

# %%
def crate(df, uname):
    df = df[df[uname] < 0.5]
    df = df[df[uname] > -0.5]
    return df

# %%
join = crate(join, "oc3")

# %%
def ranker(df):
    # df["oc3r"] = (df["oc3"].rank() - 1) / (df["oc3"].count() - 1)
    df["oc3r"] = df["oc3"].clip(lower = 0)
    # df["oc3r"] = df["oc3"]
    return df

# %%
join = join.groupby("Date").apply(ranker).reset_index()

# %%
test = join[join["Date"] > join["Date"].max() - timedelta(days = 365)]
train = join[join["Date"] <= join["Date"].max() - timedelta(days = 365)]

# %%
dgroup = train.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
dgroup["group"] = (ngroup * dgroup.index / (dgroup.index.max() + 1)).astype(int)
dgroup = dgroup[["Date", "group"]]
train = pd.merge(train, dgroup)

# %%
pq.write_table(pa.Table.from_pandas(test), "01_PROC/test1.parquet")
pq.write_table(pa.Table.from_pandas(train), "01_PROC/train1.parquet")

# %%
train.shape

# %%
train.columns

# %%
train.describe()

# %%



