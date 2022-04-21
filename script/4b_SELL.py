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
import gc
import slackweb
import datetime
import pickle
# from prophet import Prophet
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import timedelta
import yfinance as yf

# %%
os.chdir('/home/toshi/PROJECTS/PredStock')
from  common import train

# %%
def slack(txt):
    print(txt)
    try:
        slack = slackweb.Slack(url = pd.read_csv("slackkey.csv")["0"].item())
        slack.notify(text = txt)
    except:
        print("slack_error")

# %%
own = pq.read_table("01_PROC/own.parquet").to_pandas()

# %%
own

# %%
dlimit = yf.Ticker("^N225").history(period="5d", auto_adjust=True).reset_index().tail(4).head(1)["Date"].item()

# %%
to_sell = own[own["Date"] <= dlimit]
own = own[own["Date"] > dlimit]

# %%
pq.write_table(pa.Table.from_pandas(own), "01_PROC/own.parquet")

# %%
to_sell

# %%
# strdate = str(datetime.date(date.year, date.month, date.day))
s = ""
for i in range(len(to_sell)):
    s += "Sell "+ to_sell["code"][i] + "\n"

# %%
slack(s)

# %%



