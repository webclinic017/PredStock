import datetime
import gc

# import scipy
import math
import os

# import slackweb
import pickle
import random
from datetime import timedelta

# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import talib
import pyarrow as pa
import pyarrow.parquet as pq
import xlrd
import yfinance as yf
from autogluon.core.dataset import TabularDataset
from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.knn.knn_rapids_model import KNNRapidsModel
from sklearn.linear_model import LinearRegression
from sklearnex import patch_sklearn
from tqdm import tqdm

patch_sklearn()


# ダウンロードする株価の種別を決める
def get_data_j():
    os.chdir("/home/toshi/PROJECTS/PredStock")
    data_j = pd.read_excel("data_j.xls")
    data1 = data_j[data_j["市場・商品区分"].isin(["プライム（内国株式）"])]
    data2 = data_j[data_j["市場・商品区分"].isin(["市場第一部（内国株）"])]
    if len(data1) > len(data2):
        data_j = data1
    else:
        data_j = data2

    data_j = data_j.rename(columns={"コード": "code", "33業種区分": "indus", "規模コード": "scale"})
    data_j.index = data_j.index.astype(int)
    data_j = data_j[data_j["code"] < 10000]
    data_j["code"] = data_j["code"].astype("str")
    return data_j[["code", "indus", "scale"]]


def history(i: str):
    flag = 0
    path = "0a_HISTORY/"
    try:
        df = pq.read_table(path + i.replace("^", "-") + "_price.parquet").to_pandas()
        try:
            df2 = yf.Ticker(i).history(period="1mo", auto_adjust=True).reset_index()
            df2["Date"] = pd.to_datetime(df2["Date"]).dt.date
            df = pd.concat([df, df2])
        except:
            flag = 1

        if df.duplicated().sum() <= 10:
            flag = 1
    except:
        flag = 1
    if flag == 0:
        print("add_to_old")
        df = df.drop_duplicates(subset=["Date"], keep="last")
    else:
        print("rewrite")
        df = yf.Ticker(i).history(period="max", auto_adjust=True).reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Dividends"] = df["Dividends"].astype("float")
    df["Stock Splits"] = df["Stock Splits"].astype("float")
    df = df.sort_values("Date")
    pq.write_table(
        pa.Table.from_pandas(df), path + i.replace("^", "-") + "_price.parquet"
    )


def reader(n: str, ndays: int, path: str):
    try:
        df = pq.read_table(path + n.replace("^", "-") + "_price.parquet").to_pandas()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.set_index("Date")
        df["xVolume"] = df["Volume"] * (df["Close"] + df["Open"]) * 0.5
        df = pd.concat([df, pct(df)], axis=1).drop("xVolume", axis = 1)
        df = df[df.index >= (df.index.max() - timedelta(days=ndays))]
        nlist = ["Open", "High", "Low"]
        for m in nlist:
            df["r" + m] = np.log(df[m] / df["Close"])
        df = df.reset_index()
        df["code"] = n
    except:
        df = pd.DataFrame()
    return df


def pct(df):
    df = df.sort_index().add_prefix("p")
    return np.log(df.pct_change() + 1)[["pClose", "pxVolume"]]


# モーメンタム計算
def mmt(df, ndays):
    df = df[df["Date"] >= df["Date"].max() - timedelta(days=ndays)]
    p1 = df.head(1)["Close"].item()
    p2 = df.tail(1)["Close"].item()
    rate = p2 / p1
    ratio = (df["Date"].max() - df["Date"].min()) / datetime.timedelta(days=ndays)
    return np.log(rate / ratio)


def slack(txt):
    print(txt)
    try:
        slack = slackweb.Slack(url=pd.read_csv("slackkey.csv")["0"].item())
        slack.notify(text=txt)
    except:
        print("slack_error")


# ベータ
def beta(x, y):
    x = np.array(x)
    lr = LinearRegression()
    X = x.reshape((len(x), 1))
    lr.fit(X, y)
    coef = lr.coef_[0]
    return coef


# 日の性質係数
def days(df, name):
    r = np.corrcoef(df[name], df["pClose"])[0, 1]
    return r


def dfall(ndays, path):
    listx = []
    data_j = get_data_j()
    for i in data_j["code"]:
        i = i + ".T"
        df = reader(i, ndays, path)
        df["code"] = i
        if len(df) > 1:
            listx.append(df)
    dfout = pd.concat(listx)
    dfout["Date"] = pd.to_datetime(dfout["Date"]).dt.date
    dfout = dfout[dfout["Date"] > (dfout["Date"].max() - timedelta(days=ndays))]
    return dfout


def train(path, name, dropname):
    train_b = TabularDataset(path)

    save_path = None
    label_column = name
    metric = "r2"

    hyperparameters = {
        KNNRapidsModel: {
            "ag_args_fit": {"num_gpus": 1},
            "ag_args_ensemble": {"num_folds_parallel": 1},
        },
        "LR": {"ag_args_ensemble": {"num_folds_parallel": 4}},
        "XGB": {
            "ag_args_fit": {"num_gpus": 1},
            "ag_args_ensemble": {"num_folds_parallel": 1},
        },
        # "CAT": {
        #     "ag_args_fit": {"num_gpus": 1},
        #     "ag_args_ensemble": {"num_folds_parallel": 1},
        # },
        # "GBM": [
        #     # {
        #     #     "ag_args_fit": {"num_gpus": 1},
        #     #     "ag_args_ensemble": {"num_folds_parallel": 3},
        #     #     "extra_trees": True,
        #     #     "ag_args": {"name_suffix": "XT"},
        #     # },
        #     {
        #         "ag_args_fit": {"num_gpus": 1},
        #         "ag_args_ensemble": {"num_folds_parallel": 3},
        #     },
        #     # 'GBMLarge',
        # ],
        # "XT": {"ag_args_ensemble": {"num_folds_parallel": 1}},
        # 'NN_TORCH': {'ag_args_fit': {'num_gpus': 1}, "ag_args_ensemble": {"num_folds_parallel": 3}},
        # 'FASTAI': {'ag_args_fit': {'num_gpus': 1}, "ag_args_ensemble": {"num_folds_parallel": 3}},
        # 'TRANSF': {
        #     'ag_args_fit': {'num_gpus': 1},
        #     "ag_args_ensemble": {"num_folds_parallel": 1},
        #     "batch_size": 4096,
        #     "d_size": 16,
        #     },
    }

    hyperparameter_tune_kwargs = {
        "searcher": "auto",
        "scheduler": "local",
        "num_trials": 10,
    }
    
    new = TabularPredictor(
        label=label_column, eval_metric=metric, path=save_path, groups="group"
    )

    new.fit(
        train_data=train_b.drop(dropname, axis = 1),
        num_bag_folds=10,
        num_bag_sets=1,
        num_stack_levels=1,
        hyperparameters=hyperparameters,
        # hyperparameter_tune_kwargs = hyperparameter_tune_kwargs,
        save_space=True,
    )
    return new

def divsplit(df):
    df = df[["Date", "Dividends", "Stock Splits", "code"]]
    df["Stock Splits"] = (df["Stock Splits"] > 0) * 1
    for i in range(len(df)):
        df.iat[i,1] = max(df.iat[i,1], df.iat[i-1,1] * 0.7)
        df.iat[i,2] = max(df.iat[i,2], df.iat[i-1,2] * 0.7)
    return df

if __name__ == '__main__':
    print("これは自作モジュールです")