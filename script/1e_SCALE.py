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
ndays = 365

# %%
df = dfall(ndays, path)

# %%
df

# %%
def scale(df):
    return (df["Close"] * df["Volume"]).sum()

# %%
df2 = df.groupby("code").apply(scale).reset_index().rename(columns={0: "scale"})

# %%
df2

# %%
pq.write_table(pa.Table.from_pandas(df2), "01_PROC/scale.parquet")

# %%



