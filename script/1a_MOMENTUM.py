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
from  common import get_data_j, reader, mmt, dfall

# %%
path = "0a_HISTORY/"

# %%
nlist = [365, 90, 30]

# %%
df = dfall(2190, path)
df["Date"] = pd.to_datetime(df["Date"])

# %%
lista = []
for i in df["code"].unique():
    df_cut = df[df["code"] == i]
    lista.append(mmt(df_cut, nlist))

# %%
df2 = pd.concat(lista).dropna()

# %%
df2

# %%
df2.describe()

# %%
pq.write_table(pa.Table.from_pandas(df2), "01_PROC/mm.parquet")

# %%



