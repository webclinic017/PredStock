import gc
import math
import os
import pickle
import random

# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import slackweb
import talib
# from prophet import Prophet
from sklearnex import patch_sklearn
from tqdm import tqdm

patch_sklearn()


os.chdir("/home/toshi/PROJECTS/PredStock")


def index_dtime(df):
    df.index = pd.to_datetime(df.index)
    return df


# ダウンロードする株価の種別を決める
def get_data_j():
    data_j = pd.read_excel("data_j.xls")[["コード", "市場・商品区分", "17業種コード", "規模コード"]]

    #     削除するものは有効にする
    data_j = data_j[data_j["市場・商品区分"] != "ETF・ETN"]
    data_j = data_j[data_j["市場・商品区分"] != "JASDAQ(グロース・内国株）"]
    data_j = data_j[data_j["市場・商品区分"] != "JASDAQ(スタンダード・内国株）"]
    data_j = data_j[data_j["市場・商品区分"] != "JASDAQ(スタンダード・外国株）"]
    data_j = data_j[data_j["市場・商品区分"] != "PRO Market"]
    data_j = data_j[data_j["市場・商品区分"] != "REIT・ベンチャーファンド・カントリーファンド・インフラファンド"]
    data_j = data_j[data_j["市場・商品区分"] != "マザーズ（内国株）"]
    data_j = data_j[data_j["市場・商品区分"] != "マザーズ（外国株）"]
    data_j = data_j[data_j["市場・商品区分"] != "出資証券"]
    #     data_j = data_j[data_j['市場・商品区分'] != '市場第一部（内国株）']
    data_j = data_j[data_j["市場・商品区分"] != "市場第一部（外国株）"]
    data_j = data_j[data_j["市場・商品区分"] != "市場第二部（内国株）"]
    data_j = data_j[data_j["市場・商品区分"] != "市場第二部（外国株）"]

    data_j = data_j.drop("市場・商品区分", axis=1)
    data_j = data_j.rename(
        columns={"コード": "code", "17業種コード": "indus", "規模コード": "scale"}
    ).sort_values("code")
    data_j.index = data_j.index.astype(int)
    data_j = data_j[data_j["code"] < 10000]
    return data_j


data_j = get_data_j()
data_j["code"] = data_j["code"].astype("str")
data_j = (
    data_j.append({"code": "0000", "indus": np.nan, "scale": np.nan}, ignore_index=True)
    .sort_values("code")
    .reset_index(drop=True)
)
code_list = data_j.drop(["indus", "scale"], axis=1).reset_index(drop=True)

# 産業コードをしぼる
# data_j = data_j[data_j['indus'] == 1]


def feature(df, lday):
    #     time series
    list_ = []

    n_diff = ["xCP", "VOL"]
    df_diff = df[n_diff]
    list_.append(np.log(df_diff.pct_change() + 1))

    n_ratio = [
        "OP",
        "HP",
        "LP",
        #  "BBUP", "BBLO",
    ]
    for name in n_ratio:
        list_.append(pd.DataFrame(np.log(df[name] / df["CP"]), columns={name}))

    # n_ratio2 = [
    #     'NOP',
    #     # 'NHP',
    #     # 'NLP'
    #     ]
    # for name in n_ratio2:
    #     list_.append(pd.DataFrame(np.log(df[name]/df['NCP']), columns = {name}))

    # list_.append(df[[
    #     "MACDHIS",
    #      ]])

    dffeat = pd.concat(list_, axis=1).replace([np.inf, -np.inf], np.nan)
    # dffeat[['xCP', 'OP', 'HP', 'LP']] *= 30
    # dffeat[[
    #     'NOP',
    #     # 'NHP',
    #     # 'NLP'
    #     ]] *= 60
    # dffeat[["MACDHIS"]] *= 0.015
    # df[["BBMID", "BBUP", "BBLO", "SMA5", "SMA25"]] *= 6
    # df[["RSI9", "RSI14"]] *= 0.03

    list_2 = []
    for n in range(lday):
        list_2.append(dffeat.add_prefix(str(n + 1) + "_").shift(n))
    out = pd.concat(list_2, axis=1)

    out = pd.concat(
        [
            out,
            df[
                [
                    "LVOL",
                    # "RSI9", "RSI14",
                    # "BBMID", "BBUP", "BBLO",
                    # "SMA5", "SMA25",
                    "CP",
                ]
            ],
        ],
        axis=1,
    )

    return out


def add_talib(df):
    df = df.ffill()
    df["SMA5"] = talib.SMA(df["CP"], timeperiod=5)
    df["SMA25"] = talib.SMA(df["CP"], timeperiod=25)
    df["SMA5"] = np.log(df["SMA5"] / df["CP"])
    df["SMA25"] = np.log(df["SMA25"] / df["CP"])
    df["BBUP"], df["BBMID"], df["BBLO"] = talib.BBANDS(
        df["CP"], timeperiod=25, nbdevup=2, nbdevdn=2, matype=0
    )
    df["BBUP"] = np.log(df["BBUP"] / df["CP"])
    df["BBMID"] = np.log(df["BBMID"] / df["CP"])
    df["BBLO"] = np.log(df["BBLO"] / df["CP"])
    df["MACD"], df["MACDSIG"], df["MACDHIS"] = talib.MACD(
        df["CP"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["RSI9"] = talib.RSI(df["CP"], timeperiod=9) - 50
    df["RSI14"] = talib.RSI(df["CP"], timeperiod=14) - 50
    return df[
        [
            "SMA5",
            "SMA25",
            "BBUP",
            "BBMID",
            "BBLO",
            "MACD",
            "MACDSIG",
            "MACDHIS",
            "RSI9",
            "RSI14",
        ]
    ]


XBRL_list = pd.read_csv("XBRL/XBRL_list.csv").drop_duplicates(subset=["DATE", "code"])
XBRL_list["code"] = XBRL_list["code"].astype(str)
data_j = pd.merge(
    pd.DataFrame(XBRL_list["code"].unique().astype("str"), columns=["code"]), data_j
)


lday = 10
holdout = 62
day_sample = 440 + int(200 * random.random())

path = "./00-JPRAW/"
df_date = pd.read_csv(path + "0000" + ".csv", index_col=0, parse_dates=True)
df_date = df_date.add_prefix("N")
df_date.drop("NVOL", axis=1, inplace=True)

n = 0
list_tr = []
list_te = []
for i, j, scale in zip(data_j["code"], data_j["indus"], data_j["scale"]):
    k = str(i)
    l = str(j)
    scale2 = str(scale)
    if os.path.exists(path + k + ".csv") == 1:
        df = pd.read_csv(path + k + ".csv", index_col=0, parse_dates=True)
        df["VOL"] *= (df["OP"] + df["CP"]) / 2
        df["LCP"] = np.log(df["CP"])
        df["LVOL"] = (np.log(df["VOL"].rolling(lday).mean()) - 10) / 10
        df_plus = pd.concat([df_date, df], axis=1)
        df_plus = pd.concat([df_plus, add_talib(df_plus)], axis=1)
        df_plus["xOP"] = df_plus["OP"] / df_plus["NOP"]
        df_plus["xHP"] = df_plus["HP"] / df_plus["NHP"]
        df_plus["xLP"] = df_plus["LP"] / df_plus["NLP"]
        df_plus["xCP"] = df_plus["CP"] / df_plus["NCP"]
        df_uni = feature(df_plus, lday)  # for predict
        # df_uni = pd.concat([feature(df_plus, lday), label(df_plus, nday)], axis = 1) #for train
        df_uni = df_uni.replace(np.inf, np.nan).replace(-np.inf, np.nan)
        df_uni["code"] = k
        df_uni["indus"] = l
        df_uni["scale"] = scale2
        df_uni["sinday"] = np.sin(df_uni.index.weekday.astype("float") * 2 * np.pi / 5)
        df_uni["cosday"] = np.cos(df_uni.index.weekday.astype("float") * 2 * np.pi / 5)
        # df_uni['day']= df_uni.index.weekday.astype("float")
        xbrl_cut = XBRL_list[XBRL_list["code"] == i]
        xbrl_cut["DATE"] = pd.to_datetime(xbrl_cut["DATE"])
        xbrl_cut["DATEX"] = xbrl_cut["DATE"]
        xbrl_cut = xbrl_cut.set_index("DATE")
        xbrl_cut.drop("code", axis=1, inplace=True)
        df_uni = pd.concat([df_uni, xbrl_cut], axis=1)
        df_uni["DATEX"] = df_uni["DATEX"].fillna(method="ffill")
        df_uni["dura"] = (df_uni["DATEX"] - df_uni.index) * (-1)
        df_uni["dura"] = df_uni["dura"].dt.total_seconds() / 86400
        df_uni.drop("DATEX", axis=1, inplace=True)

        list_te.append(df_uni.tail(holdout))
        list_tr.append(df_uni.shift(holdout).tail(day_sample))
train = pd.concat(list_tr).reset_index()
test = pd.concat(list_te).reset_index()
del list_te, list_tr


train


p_rate = 0.8


def prep(df):
    df = df[df["CP"] < 3000]
    df.drop("CP", axis=1, inplace=True)

    # 予測のときはLABELを処理しない
    # df['LABEL'] = df['RATE'] > np.log(drate)
    # df['LABEL'] = df['LABEL'].astype('int') * 1

    df.drop("dura", axis=1, inplace=True)
    # df = df[df['dura'] <= 40]
    # df['dura'] /= 5

    # df.drop("day", axis = 1, inplace = True)
    # df = pd.get_dummies(df, columns=['day'])

    df.drop("scale", axis=1, inplace=True)
    # df = pd.get_dummies(df, columns=['scale'])

    # df.drop("indus", axis = 1, inplace = True)
    df = pd.get_dummies(df, columns=["indus"])

    # 予測のときはdropnaしない
    # df.dropna(inplace = True, subset = ["1_xCP"])
    # df.dropna(inplace = True)

    df = df.reset_index(drop=True)

    # df["RATE2"] = df["RATE"]

    df = df.groupby("DATE").apply(ranker).reset_index(drop=True)

    # df["RATE"] = (df["RATE"] > p_rate) * 1

    return df


def ranker(df):
    ignore_list = [
        "DATE",
        "sinday",
        "cosday",
        "indus_1",
        "indus_2",
        "indus_3",
        "indus_4",
        "indus_5",
        "indus_6",
        "indus_7",
        "indus_8",
        "indus_9",
        "indus_10",
        "indus_11",
        "indus_12",
        "indus_13",
        "indus_14",
        "indus_15",
        "indus_16",
        "indus_17",
        # "day_0.0", "day_1.0", "day_2.0",
        # "day_3.0", "day_4.0" ,
        "code",
        "LVOL",
        # "RATE2",
    ]
    # 昇順でランクづけされる、つまり最も値上がりしたものは１、最も値下がりしたものは０
    df1 = df.drop(ignore_list, axis=1).rank() - 1
    df1 /= df1.max()
    df2 = df[ignore_list]
    return pd.concat([df1, df2], axis=1)


train = prep(train)
test = prep(test)


# 今日の日付のものだけ
test = test[test["DATE"] == test["DATE"].max()].reset_index(drop=True)


import pickle

pre = pickle.load(open("ag_model_flat_best", "rb"))


# X_test = test.drop(['DATE', 'code'], axis = 1)


try:
    dffuture2 = pd.concat(
        [test, pre.predict_proba(test).rename(columns={1: "pred"})["pred"]], axis=1
    )
except:
    dffuture2 = pd.concat(
        [test, pd.DataFrame(pre.predict(test)).rename(columns={"RATE": "pred"})], axis=1
    )


dffuture2["pred"].hist()


dffuture2.sort_values("pred", ascending=False)[["code", "pred"]].head(1)


dffuture3 = dffuture2.sort_values("pred", ascending=False)[["code", "pred"]]
date = test["DATE"].tail(1).item()
print(date)
print(dffuture3)
print("全行程完了")


if len(dffuture3) == 0:
    print("NO BEST")
    subject_a = str(date) + " の予報 ベストナシ"
    body_a = "NO BEST"
else:
    print("BEST")
    subject_a = (
        str(date) + " の予報 " + str(dffuture3.reset_index().loc[:, "code"].iloc[0])
    )
    body_a = "BEST\n"
    for i in range(1):
        if len(dffuture3) == i:
            break
        body_a = (
            body_a
            + "#"
            + str(i)
            + " Code="
            + str(dffuture3.reset_index().loc[:, "code"].iloc[i])
        )
        body_a = body_a + "/Score=" + str(dffuture3.loc[:, "pred"].iloc[i]) + "\n"
    # body_a = body_a + ' Length = ' + str(len(dffuture3))


slacker = subject_a + "\n" + body_a


print(slacker)


try:
    slack = slackweb.Slack(url=pd.read_csv("slackkey.csv")["0"].item())
    slack.notify(text=slacker)
except:
    print(1)

