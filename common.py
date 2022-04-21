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

def slack(txt):
    print(txt)
    try:
        slack = slackweb.Slack(url=pd.read_csv("slackkey.csv")["0"].item())
        slack.notify(text=txt)
    except:
        print("slack_error")

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

    data_j = data_j.rename(columns={"コード": "code", "33業種コード": "indus", "規模コード": "scale"})
    data_j.index = data_j.index.astype(int)
    data_j = data_j[data_j["code"] < 10000]
    data_j["code"] = data_j["code"].astype("str")
    return data_j[["code", "indus", "scale", "33業種区分"]]





def pct(df):
    dfout = np.log(df[["xVolume", "Close"]].pct_change() + 1)
    dfout = dfout.rename(columns={"xVolume": "rxVolume", "Close": "ret"})

    return dfout


def reader(n: str, path: str):
    try:
        df = pq.read_table(path + n.replace("^", "-") + "_price.parquet").to_pandas()
        df["code"] = n
    except:
        df = pd.DataFrame()
    return df


# def readall(ndays, path):
#     listx = []
#     data_j = get_data_j()
#     for i in data_j["code"]:
#         i = i + ".T"
#         df = reader(i, ndays, path).drop(["Open", "High", "Low", "Close"], axis=1)
#         df["code"] = i
#         if len(df) > 1:
#             listx.append(df)
#     dfout = pd.concat(listx)
#     dfout["Date"] = pd.to_datetime(dfout["Date"])
#     dfout = dfout[dfout["Date"] > (dfout["Date"].max() - timedelta(days=ndays))]
#     return dfout


# モーメンタム計算
def momentum(dfin, mmlist):
    listc = []
    for cname in dfin["code"].unique():
        df = dfin[dfin["code"]==cname].reset_index(drop = True)
        listb=[]
        for lenmm in mmlist:
            lista =[]
            for i in range(len(df)):
                moment =  df[df["Date"] > df["Date"].iloc[i : i + 1].item() - timedelta(days=lenmm)]["ret"].sum()
                lista.append(moment)
            listb.append(pd.DataFrame(lista, columns=["momentum_"+str(lenmm)]))
        dfb = pd.concat(listb, axis = 1)
        dfb["code"] = cname
        dfb["Date"] = df["Date"]
        listc.append(dfb)
    return pd.concat(listc)

# ベータ
def beta(x, y):
    x = np.array(x)
    lr = LinearRegression()
    X = x.reshape((len(x), len(x.T)))
    lr.fit(X, y)
    return lr.coef_


# 日の性質係数
def days(df, name):
    r = np.corrcoef(df[name], df["pClose"])[0, 1]
    return r


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
        "GBM": [
            # {
            #     "ag_args_fit": {"num_gpus": 1},
            #     "ag_args_ensemble": {"num_folds_parallel": 3},
            #     "extra_trees": True,
            #     "ag_args": {"name_suffix": "XT"},
            # },
            {
                "ag_args_fit": {"num_gpus": 1},
                "ag_args_ensemble": {"num_folds_parallel": 3},
            },
            # 'GBMLarge',
        ],
        # "XT": {"ag_args_ensemble": {"num_folds_parallel": 1}},
        "NN_TORCH": {
            "ag_args_fit": {"num_gpus": 1},
            "ag_args_ensemble": {"num_folds_parallel": 3},
        },
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
        train_data=train_b.drop(dropname, axis=1),
        num_bag_folds=10,
        num_bag_sets=1,
        num_stack_levels=2,
        hyperparameters=hyperparameters,
        # hyperparameter_tune_kwargs = hyperparameter_tune_kwargs,
        save_space=True,
    )
    return new


def divsplit(df):
    df = df[["Date", "Dividends", "Stock Splits", "code"]]
    df["Stock Splits"] = (df["Stock Splits"] > 0) * 1
    for i in range(len(df)):
        df.iat[i, 1] = max(df.iat[i, 1], df.iat[i - 1, 1] * 0.7)
        df.iat[i, 2] = max(df.iat[i, 2], df.iat[i - 1, 2] * 0.7)
    return df


if __name__ == "__main__":
    print("これは自作モジュールです")
