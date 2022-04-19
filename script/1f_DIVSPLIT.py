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
from  common import get_data_j, reader, mmt, beta, dfall, divsplit

# %%
path = "0a_HISTORY/"

# %%
ndays = 1460

# %%
df = dfall(ndays, path)

# %%
df

# %%
df2 = df.groupby("code").apply(divsplit).reset_index()

# %%
df2 = df2.drop("index", axis = 1)

# %%
pq.write_table(pa.Table.from_pandas(df2), "01_PROC/divsplit.parquet")

# %%
df2

# %%



