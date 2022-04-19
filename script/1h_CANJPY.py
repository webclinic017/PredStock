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
ndays = 1825

# %%
dfn = reader("JPY=X", ndays, path)[["Date", "pClose"]].rename(columns={"pClose": "jpypClose"})

# %%
dfn

# %%
# df = dfall(ndays, path)
# df["pxVolume"] = df["pxVolume"].replace(np.inf, 0).replace(-np.inf, 0)

# %%
cannik = pq.read_table("01_PROC/cannik.parquet").to_pandas()

# %%
cannik

# %%
df = pd.merge(cannik, dfn, on = "Date")

# %%
beta = pq.read_table("01_PROC/betajpy.parquet").to_pandas()

# %%
beta

# %%
df = pd.merge(df, beta, on = "code")

# %%
df

# %%
df["pClose_y"] = df["pClose"] - (df["betajpy"] * df["jpypClose"])

# %%
pq.write_table(pa.Table.from_pandas(df), "01_PROC/cannikjpy.parquet")

# %%
df

# %%
df[["pClose", "npClose", "pClose_n", "pClose_y", "jpypClose"]].hist(figsize=(15, 12), range=(-0.1, 0.1), bins = 50)

# %%



