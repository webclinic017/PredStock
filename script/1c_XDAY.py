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
ndays = 1460

# %%
dfall = dfall(ndays, path)

# %%
dfall

# %%
mm = pq.read_table("01_PROC/mm.parquet").to_pandas()

# %%
mm

# %%
beta = pq.read_table("01_PROC/beta.parquet").to_pandas()

# %%
beta

# %%
dfall = pd.merge(dfall, mm, on = "code")
dfall = pd.merge(dfall, beta, on = "code")

# %%
dfall

# %%
df2 = dfall[dfall["Date"] == datetime.date(2022, 4, 11)]

# %%
df2

# %%
plt.scatter(df2["mm"], df2["pClose"], alpha = 0.5)

# %%
plt.scatter(df2["beta"], df2["pClose"], alpha = 0.5)

# %%
nlist = ["mm", "beta"]
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
xday.index = pd.to_datetime(xday.index)
xday = xday.add_prefix("day").reset_index()
xday["daymm"].fillna(0, inplace=True)
xday["daybeta"].fillna(0, inplace=True)

# %%
pq.write_table(pa.Table.from_pandas(xday), "01_PROC/xday.parquet")

# %%
xday

# %%
xday.corr()

# %%
fig = plt.figure(figsize = (5,5), facecolor="white")
# plt.plot(xday["dayvcoef"], xday["daymm"], alpha = 0.2)
plt.scatter(xday["daybeta"], xday["daymm"], s = 3)

# %%
fig = plt.figure(figsize = (18,5), facecolor="white")
plt.plot(xday["Date"], xday["daymm"])
plt.scatter(xday["Date"], xday["daymm"], s = 3)

# %%
fig = plt.figure(figsize = (18,5), facecolor="white")
plt.plot(xday["Date"], xday["daybeta"])
plt.scatter(xday["Date"], xday["daybeta"], s = 3)

# %%
xday

# %%
df = xday.copy()
df = df.rename(columns={"Date": "ds", "daymm": "y"})

# %%
m = Prophet()
m.fit(df)

# %%
future = m.make_future_dataframe(periods=1)

# %%
forecast = m.predict(future)

# %%
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# %%

fig1 = m.plot(forecast)


# %%



