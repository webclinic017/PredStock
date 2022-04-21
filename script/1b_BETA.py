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

# %%
os.chdir('/home/toshi/PROJECTS/PredStock')
from  common import get_data_j, reader, mmt, beta

# %%
path = "0a_HISTORY/"

# %%
data_j = get_data_j()

# %%
ndays = 730

# %%
dfn = reader("^N225", ndays, path).add_prefix("n")

# %%
dfn

# %%
listx = []
for i in data_j["code"]:
    i = i + ".T"
    df = reader(i,ndays, path)
    if len(df) > 1:
        df = pd.concat([df, dfn], axis = 1).dropna()
        out = [beta(df["npClose"], df["pClose"])]
        out.append(i)
        # print(out)
        listx.append(out)

# %%
beta = pd.DataFrame(listx, columns=["beta", "code"])

# %%
pq.write_table(pa.Table.from_pandas(beta), "01_PROC/beta.parquet")

# %%
beta

# %%
beta["beta"].hist()

# %%



