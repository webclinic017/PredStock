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
from  common import  dfall, reader


# %%
path = "0a_HISTORY/"

# %%
ndays = 1460

# %%
df = dfall(ndays, path)
df["Date"] = pd.to_datetime(df["Date"])

# %%
dfn = reader("^N225", ndays, path).add_prefix("n").rename(columns={"nDate": "Date"})
dfn["Date"] = pd.to_datetime(dfn["Date"])

# %%
dfn

# %%
df = pd.merge(df, dfn, on = "Date")

# %%
df

# %%
# OP_CP
def oc3(df):
    nday = 3
    df["oc" + str(nday)] = np.log(df['Close'].shift(-nday) / df['Open'].shift(-1))
    return df[["oc" + str(nday), "Date", "code"]]

# # CP_CP
# def cc1(df):
#     nday = 1
#     df["cc" + str(nday)] = np.log(df['Close'].shift(-nday) / df['Close'])
#     return df[["cc" + str(nday), "Date", "code"]]

# %%
df2 = df.groupby("code").apply(oc3)
# df3 = df.groupby("code").apply(cc1)
# df_out = pd.merge(df2, df3, on = ["Date", "code"])
df_out = df2

# %%
pq.write_table(pa.Table.from_pandas(df_out), "01_PROC/tslabel.parquet")

# %%
df_out

# %%
# fig = plt.figure(figsize = (8,8), facecolor="white")
# # plt.plot(df_out["oc3"], df_out["cc1"], alpha = 0.1)
# plt.scatter(df_out["oc3"], df_out["cc1"], s = 2, alpha = 0.1)
# plt.xlim(-0.5, 0.5)
# plt.ylim(-0.5, 0.5)

# %%



