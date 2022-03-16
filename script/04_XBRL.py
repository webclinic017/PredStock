import glob
import os
import shutil
import time

import numpy as np
import pandas as pd
import timeout_decorator
from tqdm import tqdm

os.chdir("/home/toshi/PROJECTS/PredStock/XBRL/")

data_j = pd.read_excel("../data_j.xls")[["コード", "銘柄名"]]

data_j["後株"] = data_j["銘柄名"] + "株式会社"

data_j["前株"] = "株式会社" + data_j["銘柄名"]

# zip から読み込む (データフレームのリストが返る)
allfiles = glob.glob("./ZIP/*")
# allfiles.append(glob.glob('C:/Users/toshi/Documents/STOCK/XBRL/ZIP_NOT_CSV/*'))

# print(allfiles)

from xbrl_proc import read_xbrl_from_zip


@timeout_decorator.timeout(15, use_signals=False)
def process(i):
    fname = i.rsplit("/", 1)[1]
    try:
        zip_file = i
        df_list = read_xbrl_from_zip(zip_file)
        df = df_list[0]
        name = df[df["tag"] == "FilerNameInJapaneseDEI"]["値"]
        code = pd.concat(
            [data_j[data_j["後株"] == name.iloc[0]], data_j[data_j["前株"] == name.iloc[0]]]
        )["コード"].iloc[0]
        date = df["提出日"][0].date()
        df.to_csv("./CSV/i{}_{}.csv".format(code, str(df["提出日"][0].date())))
        shutil.move(zip_file, "./ZIP_GET_CSV/" + fname)
        # print(date, code)
        return date, code
    except:
        shutil.move(zip_file, "./ZIP_NOT_CSV/" + fname)
        return 0, 0


try:
    df_old = pd.read_csv("XBRL_list.csv").drop_duplicates()
    print("get_list")
except:
    df_old = pd.DataFrame()


list_ = []
if len(allfiles) > 0:
    for i in allfiles:
        DATE, code = process(i)
        code = int(code)
        # print(DATE, code)
        list_.append(pd.DataFrame([DATE, code], index=["DATE", "code"]).T)
    df_new = pd.concat(list_)
else:
    df_new = pd.DataFrame()

if len(df_new) == 0:
    if len(df_old) == 0:
        print("no_data")
    else:
        df_old.drop_duplicates().sort_values("code").to_csv(
            "XBRL_list.csv", index=False
        )
else:
    if len(df_old) == 0:
        df_new.drop_duplicates().sort_values("code").to_csv(
            "XBRL_list.csv", index=False
        )
    else:
        pd.concat([df_new, df_old]).drop_duplicates().sort_values("code").to_csv(
            "XBRL_list.csv", index=False
        )

