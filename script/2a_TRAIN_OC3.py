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
import pickle

# %%
os.chdir('/home/toshi/PROJECTS/PredStock')
from  common import train


# %%
path = "01_PROC/train1.parquet"

# %%
new = train(path, "oc3r", ["oc3", "code", "index", "Date"])
pickle.dump(new, open('oc3_new.mdl', 'wb'))

# %%



