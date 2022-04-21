# %%
import pandas as pd
import numpy as np
import os
import pickle
# import xlrd
# pip install yfinance
import datetime
import yfinance as yf

import timeout_decorator
from retry import retry
import os
import pathlib
import datetime
import time
import platform

# %%
os.chdir('../')
from  common import get_data_j, history

# %%
code = pd.DataFrame(get_data_j()["code"])

# %%
code2 = list(code["code"] + ".T")
# code2 = []

# %%
aux = pd.read_csv("aux.csv")

# %%
list1 = aux["ticker"]

# %%
list1 = ["^N225", "^DJI", "^IXIC", "JPY=X", "EURUSD=X", "GBPUSD=X", "CNYJPY=X"]
for i in list1:
    code2.append(i)

# %%
@retry(tries=10)
@timeout_decorator.timeout(30)
def ydl(i):
    print(i)
    history(i)


# %%
path = "0a_HISTORY/"

# %%
for i in code2:
    ydl(i)

# %%



