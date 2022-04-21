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
from  common import get_data_j, reader, mmt

# %%
path = "0a_HISTORY/"

# %%
data_j = get_data_j()

# %%
ndays = 730

# %%
mmlist = [ndays]

# %%
lista = []
for i in data_j["code"]:
   i = str(i) + ".T"
   df = reader(i,ndays, path)
   if len(df) > 1:
      listb = []
      listb.append(mmt(df,ndays))
      dfmm = pd.DataFrame(listb, index = ["mm"]).T
      dfmm["code"] = [i]
      lista.append(dfmm)
dfmm = pd.concat(lista).reset_index(drop = True)

# %%
pq.write_table(pa.Table.from_pandas(dfmm), "01_PROC/mm.parquet")

# %%
dfmm

# %%



