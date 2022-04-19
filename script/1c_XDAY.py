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
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from sklearnex import patch_sklearn
patch_sklearn()

from prophet import Prophet

# %%
os.chdir('/home/toshi/PROJECTS/PredStock')
from  common import get_data_j, mmt, days, dfall, reader

# %%
path = "0a_HISTORY/"

# %%
# data_j = get_data_j()

# %%
ndays = 1825

# %%
dfall = dfall(ndays, path)
dfall["Date"] = pd.to_datetime(dfall["Date"])

# %%
dfall

# %%
mm = pq.read_table("01_PROC/mm.parquet").to_pandas()
mm["Date"] = pd.to_datetime(mm["Date"])

# %%
mm

# %%
beta = pq.read_table("01_PROC/beta.parquet").to_pandas()

# %%
beta

# %%
dfall = pd.merge(dfall, mm, on = ["code", "Date"])
dfall = pd.merge(dfall, beta, on = "code")

# %%
dfall

# %%
nlist = [s for s in dfall.columns if 'mm' in s]
nlist.append("beta")

# %%
nlist

# %%
lista = []
for j in dfall["Date"].unique():
    listb = [j]
    for i in nlist:
        df_cut = dfall[dfall["Date"] == j]
        resu = days(df_cut, i)
        listb.append(resu)
    lista.append(listb)

# %%
xday = pd.DataFrame(lista, columns=["Date"] + nlist).set_index("Date")
xday = xday.fillna(0)
xday.index = pd.to_datetime(xday.index)
xday = xday.add_prefix("day").reset_index()

# %%
pq.write_table(pa.Table.from_pandas(xday), "01_PROC/xday.parquet")

# %%
xday

# %%
xday.corr()

# %%
fig = plt.figure(figsize = (7,7), facecolor="white")
# plt.plot(xday["dayvcoef"], xday["daymm"], alpha = 0.2)
plt.scatter(xday["daybeta"], xday["daymm90"], s = 2)

# %%



