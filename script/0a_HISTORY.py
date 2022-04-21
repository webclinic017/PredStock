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
os.chdir('/home/toshi/PROJECTS/PredStock')
from  common import get_data_j, history

# %%
code = pd.DataFrame(get_data_j()["code"])

# %%
code2 = list(code["code"] + ".T")

# %%
code2.append("^N225")
code2.append("JPY=X")
code2.append("EURUSD=X")
code2.append("GBPUSD=X")

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



