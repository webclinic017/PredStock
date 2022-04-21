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
from  common import get_data_j, reader, mmt, beta, dfall

# %%
path = "0a_HISTORY/"

# %%
ndays = 1460
ldays = 5
ndays = ndays + ldays + 5

# %%
df = pq.read_table("01_PROC/cannik.parquet").to_pandas()
df["Date"] = pd.to_datetime(df["Date"])
df["pClose_nd"] = df["pClose"] - df["npClose"]

# %%
df

# %%
dfxday = pq.read_table("01_PROC/xday.parquet").to_pandas()
dfxday["Date"] = pd.to_datetime(dfxday["Date"])

# %%
dfxday

# %%
df = pd.merge(df, dfxday, on = "Date")

# %%
dfyd = reader("JPY=X", ndays, path)[["Date", "pClose"]].rename(columns={"pClose": "ypClose"})
dfyd["Date"] = pd.to_datetime(dfyd["Date"])
df = pd.merge(df, dfyd, on = "Date")

# %%
df = df.drop(["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "npClose"], axis = 1)

# %%
df[["pClose",  "rOpen", "rHigh", "rLow", "pClose_n", "pClose_nd"]] *= 10
df["pxVolume"] *= 0.4
df["ypClose"] *= 50

# %%
# clipname = ["pClose","pxVolume","rOpen","rHigh","rLow","beta","pClose_n","daymm","daybeta","ypClose"]
# df[clipname] = df[clipname].clip(lower=-2, upper=2)

# %%
df.describe()

# %%
df.hist(figsize = (20, 20), bins = 50, range=(-1, 1))

# %%
def tsen(df):
    list_2 = []
    for n in range(ldays):
        list_2.append(df.drop(["Date", "code"], axis = 1).add_prefix(str(n + 1) + '_').shift(n))
    df2 = pd.concat(list_2, axis = 1)
    df2["Date"] = df["Date"]
    df2["code"] = df["code"]
    return df2

# %%
dfts = df.groupby("code").apply(tsen).reset_index().dropna().drop("index", axis = 1)

# %%
dfts

# %%
data_j = get_data_j()[["code", "indus"]]
data_j["code"] = data_j["code"] + ".T"

# %%
data_j

# %%
dfts = pd.merge(dfts, data_j, on = "code")

# %%
mm = pq.read_table("01_PROC/mm.parquet").to_pandas()

# %%
mm

# %%
dfts = pd.merge(dfts, mm, on = "code")

# %%
dfts

# %%
scale = pq.read_table("01_PROC/scale.parquet").to_pandas()
scale["scale"] = np.log(scale["scale"]) * 0.05

# %%
scale.hist()

# %%
scale

# %%
scale = scale[scale["scale"] > scale["scale"].quantile(0.25)]

# %%
dfts = pd.merge(dfts, scale, on = "code")
# dfts.drop("scale", inplace = True)

# %%
dfts['day']= dfts["Date"].dt.weekday
dfts = pd.get_dummies(dfts, columns=['day'])
dfts = pd.get_dummies(dfts, columns=['indus'])

# %%
dfts

# %%
pq.write_table(pa.Table.from_pandas(dfts), "01_PROC/tsfeature.parquet")

# %%
dfts


