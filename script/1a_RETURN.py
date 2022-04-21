# %%
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import timedelta

# %%
os.chdir('../')
from  common import  readall, momentum

# %%
path = "0a_HISTORY/"

# %%
days = 3650
mmlist = [90, 60, 30]

# %%
df = readall(days, path)

# %%
df2 = momentum(df, mmlist)

# %%
df2

# %%
pq.write_table(pa.Table.from_pandas(df2), "01_PROC/momentum.parquet")

# %%



