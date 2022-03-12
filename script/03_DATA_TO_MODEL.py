import gc
import math
import os
import pickle
import random

# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import slackweb
import talib
from joblib import Parallel, delayed
# from prophet import Prophet
from sklearn.metrics import r2_score
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


os.chdir("/home/toshi/PROJECTS/PredStock")


# nday = int(input('先読み日数'))
nday = int(1)
slack(nday)


# drate = 1 + 0.01 * float(input('変動閾値 %'))
drate = 1 + 0.01 * float(1.6 + 0.1 * (1 - 2 * random.random()))
slack(drate)


# # CP_CP
# def label(df, nday, drate):
#     df_label = pd.DataFrame(np.log(df['CP'].shift(-nday) / df['CP'])).rename(columns = {'CP': 'RATE'})

#     # df_label['LABEL'] = df_label['RATE'] > np.log(drate)

#     df_label['LABEL1'] = df_label['RATE'] > np.log(drate)
#     df_label['LABEL2'] = df['LP'].shift(-1) <= df['CP']
#     df_label['LABEL'] = df_label['LABEL1'] & df_label['LABEL2']

#     df_label['LABEL'] = df_label['LABEL'] * 1
#     df_label['LABEL'] = df_label['LABEL'].astype('int')
#     return df_label[['RATE', 'LABEL']]

# OP_CP
def label(df, nday):
    df_label = pd.DataFrame(
        np.log(df["CP"].shift(-nday) / df["OP"].shift(-1)), columns=["RATE"]
    )
    return df_label["RATE"]


# def index_dtime(df):
#     df.index = pd.to_datetime(df.index)
#     return df


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


XBRL_list = pd.read_csv("XBRL/XBRL_list.csv").drop_duplicates()
XBRL_list["code"] = XBRL_list["code"].astype(str)
data_j = pd.merge(
    pd.DataFrame(XBRL_list["code"].unique().astype("str"), columns=["code"]), data_j
)


# indusx = 1
# data_j = data_j[data_j["indus"] == indusx]


data_j


def feature(df, lday):
    #     time series
    list_ = []

    n_diff = [
        "xCP",
        "VOL",
        # "NOP"
    ]
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
                    # "BBMID",
                    #  "BBUP", "BBLO",
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
        # df_uni = feature(df_plus, lday) #for predict
        df_uni = pd.concat(
            [feature(df_plus, lday), label(df_plus, nday)], axis=1
        )  # for train
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


train.shape


n_aug = 1


def augment(df):
    ignore_list = [
        "DATE",
        "sinday",
        "cosday",
        "CP",
        "code",
        "indus",
        "scale",
        "dura",
        "LVOL",
    ]
    list_augment = []
    r_aug = 1
    for n in range(n_aug):
        print(r_aug)
        df1 = df.drop(ignore_list, axis=1)
        df2 = df[ignore_list]
        list_augment.append(pd.concat([df1 * r_aug, df2], axis=1))
        r_aug = 1.1 ** (1 - 2 * random.random())
    return pd.concat(list_augment).reset_index(drop=True)


train = augment(train)
gc.collect()


train.shape


def prep(df):
    df = df[df["CP"] < 3000]
    df.drop("CP", axis=1, inplace=True)

    df.drop("dura", axis=1, inplace=True)
    # df = df[df['dura'] <= 40]
    # df['dura'] /= 5

    # df.drop("day", axis = 1, inplace = True)
    # df = pd.get_dummies(df, columns=['day'])

    df.drop("scale", axis=1, inplace=True)
    # df = pd.get_dummies(df, columns=['scale'])

    df.drop("indus", axis=1, inplace=True)
    # df = pd.get_dummies(df, columns=['indus'])

    # 予測のときはdropnaしない
    df.dropna(inplace=True)
    df["RATE2"] = df["RATE"]
    df["RATE"] = (df["RATE"] > np.log(drate)) * 1

    df = df.reset_index(drop=True)

    return df


train = prep(train)
test = prep(test)
gc.collect()


train.shape


slack("train = " + str(len(train)))


slack("test = " + str(len(test)))


train.hist(figsize=(30, 30), bins=20)


# さらに間引く
n_sample = 10000000
if len(train) > n_sample:
    train_b = train.sample(n_sample)
else:
    train_b = train
gc.collect()


from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.knn.knn_rapids_model import KNNRapidsModel
from autogluon.tabular.models.lr.lr_rapids_model import LinearRapidsModel

save_path = None
label_column = "RATE"
# metric = 'r2'
# metric = 'f1'
metric = "roc_auc"
# metric = 'log_loss'

gbm_options = [
    {"ag_args_fit": {"num_gpus": 1}, "num_leaves": 20, "ag_args": {"name_suffix": "A"}},
    {"ag_args_fit": {"num_gpus": 1}, "num_leaves": 40, "ag_args": {"name_suffix": "B"}},
    {"ag_args_fit": {"num_gpus": 1}, "num_leaves": 80, "ag_args": {"name_suffix": "C"}},
    {
        "ag_args_fit": {"num_gpus": 1},
        "num_leaves": 160,
        "ag_args": {"name_suffix": "D"},
    },
    {
        "ag_args_fit": {"num_gpus": 1},
        "extra_trees": True,
        "num_leaves": 20,
        "ag_args": {"name_suffix": "XTA"},
    },
    {
        "ag_args_fit": {"num_gpus": 1},
        "extra_trees": True,
        "num_leaves": 40,
        "ag_args": {"name_suffix": "XTB"},
    },
    {
        "ag_args_fit": {"num_gpus": 1},
        "extra_trees": True,
        "num_leaves": 80,
        "ag_args": {"name_suffix": "XTC"},
    },
    {
        "ag_args_fit": {"num_gpus": 1},
        "extra_trees": True,
        "num_leaves": 160,
        "ag_args": {"name_suffix": "XTD"},
    },
    "GBMLarge",
]

xgb_options = [
    {"ag_args_fit": {"num_gpus": 1}, "tree_method": "gpu_hist"},
    {
        "max_depth": 7,
        "ag_args_fit": {"num_gpus": 1},
        "tree_method": "gpu_hist",
        "ag_args": {"name_suffix": "A"},
    },
    {
        "max_depth": 9,
        "ag_args_fit": {"num_gpus": 1},
        "tree_method": "gpu_hist",
        "ag_args": {"name_suffix": "B"},
    },
]

cat_options = [
    {"ag_args_fit": {"num_gpus": 1}},
    {"max_depth": 8, "ag_args_fit": {"num_gpus": 1}, "ag_args": {"name_suffix": "A"}},
    {"max_depth": 10, "ag_args_fit": {"num_gpus": 1}, "ag_args": {"name_suffix": "B"}},
]

xt_options = [
    {"n_estimators": 20, "n_jobs": -1, "ag_args": {"name_suffix": "A"}},
    {"n_estimators": 30, "n_jobs": -1, "ag_args": {"name_suffix": "B"}},
    {"n_estimators": 40, "n_jobs": -1, "ag_args": {"name_suffix": "C"}},
    {"n_estimators": 50, "n_jobs": -1, "ag_args": {"name_suffix": "D"}},
    {"n_estimators": 70, "n_jobs": -1, "ag_args": {"name_suffix": "E"}},
]

hyperparameters = {
    KNNRapidsModel: {"ag_args_ensemble": {"num_folds_parallel": 1}},
    LinearRapidsModel: {"ag_args_ensemble": {"num_folds_parallel": 1}},
    # "LR": {},
    # 'XGB': xgb_options,
    # 'CAT': cat_options,
    # 'GBM': gbm_options,
    # 'XT': xt_options,
    "XGB": {
        "ag_args_fit": {"num_gpus": 1},
        "ag_args_ensemble": {"num_folds_parallel": 2},
    },
    "CAT": {
        "ag_args_fit": {"num_gpus": 1},
        "ag_args_ensemble": {"num_folds_parallel": 1},
    },
    # 'GBM': [
    #     {'ag_args_fit': {'num_gpus': 1}, 'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
    #     # {},
    #     # 'GBMLarge',
    # ],
    # 'XT': {},
    # 'NN_TORCH': {'ag_args_fit': {'num_gpus': 1}, "MXNET_CUDNN_LIB_CHECKING" : 0},
    # 'FASTAI': {'ag_args_fit': {'num_gpus': 1}},
    # 'TRANSF': {
    #     'ag_args_fit': {'num_gpus': 1},
    #     "batch_size": 4096,
    #     "d_size": 16,
    #     },
}

hyperparameter_tune_kwargs = {
    "searcher": "auto",
    "scheduler": "local",
    "num_trials": 10,
}

new = TabularPredictor(label=label_column, eval_metric=metric, path=save_path)

new.fit(
    train_data=train_b.drop(["DATE", "code", "RATE2"], axis=1),
    num_bag_folds=10,
    num_bag_sets=3,
    num_stack_levels=1,
    hyperparameters=hyperparameters,
    # hyperparameter_tune_kwargs = hyperparameter_tune_kwargs,
    save_space=True,
)


test.to_csv("test.csv")


test = pd.read_csv("test.csv")


pickle.dump(new, open("ag_model_flat_new", "wb"))


new = pickle.load(open("ag_model_flat_new", "rb"))


# testで予測
gc.collect()
try:
    df_te_new = pd.concat(
        [test, new.predict_proba(test).rename(columns={1: "pred"})["pred"]], axis=1
    )
except:
    df_te_new = pd.concat(
        [test, pd.DataFrame(new.predict(test)).rename(columns={"RATE": "pred"})], axis=1
    )


# bestモデルを読んで予測。読めない、あるいは説明データに対応できない場合は、新モデルで上書きする。
try:
    pre = pickle.load(open("ag_model_flat_best", "rb"))
    try:
        df_te_pre = pd.concat(
            [test, pre.predict_proba(test).rename(columns={1: "pred"})["pred"]], axis=1
        )
    except:
        df_te_pre = pd.concat(
            [test, pd.DataFrame(pre.predict(test)).rename(columns={"RATE": "pred"})],
            axis=1,
        )
    slack("旧モデル予測完了")
except:
    slack("旧モデル読み込みおよび予測失敗、新モデルで上書き")
    pre = new
    df_te_pre = df_te_new
    pickle.dump(new, open("ag_model_flat_best", "wb"))


def get_best(df):
    return df.sort_values("pred", ascending=False).head(1)


score_new = df_te_new.groupby("DATE").apply(get_best)["RATE2"].mean()
score_pre = df_te_pre.groupby("DATE").apply(get_best)["RATE2"].mean()
# score_new = r2_score(df_te_new["RATE"], df_te_new["pred"])
# score_pre = r2_score(df_te_pre["RATE"], df_te_pre["pred"])

slack("score_new = " + str(score_new))
slack("score_pre = " + str(score_pre))


# 新旧モデル可視化
fig = plt.figure(figsize=(10, 6), facecolor="white")
plt.hist(
    df_te_new.groupby("DATE").apply(get_best)["RATE2"],
    bins=20,
    alpha=0.5,
    range=(-0.3, 0.3),
)
plt.hist(
    df_te_pre.groupby("DATE").apply(get_best)["RATE2"],
    bins=20,
    alpha=0.5,
    range=(-0.3, 0.3),
)
plt.title("new=" + str(score_new) + " / pre=" + str(score_pre))
plt.grid()
fig.savefig("hist_positive_ag.png")

if score_new >= score_pre:
    pickle.dump(new, open("ag_model_flat_best", "wb"))
    score_best = score_new
    slacker = "モデル更新完了 " + str(score_best)
else:
    pickle.dump(pre, open("ag_model_flat_best", "wb"))
    score_best = score_pre
    slacker = "モデルそのまま " + str(score_best)


slack(slacker)

