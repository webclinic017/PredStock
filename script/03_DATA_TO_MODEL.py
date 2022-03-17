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
nday = int(3)
slack(nday)

# drate = 1 + 0.01 * float(input('変動閾値 %'))
drate = 1 + 0.01 * float(2 + 0.1 * (1 - 2 * random.random()))
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


def feature(df, lday):
    #     time series
    n_diff = ["CP", "NCP", "VOL"]
    df_diff = df[n_diff].add_prefix("p")
    df_diff = np.log(df_diff.pct_change() + 1)
    # df_diff['xCP'] = df_diff['xCP'].clip(lower = -0.3, upper = 0.3)

    n_ratio = ["OP", "HP", "LP"]
    list_ = []
    for name in n_ratio:
        list_.append(pd.DataFrame(np.log(df[name] / df["CP"]), columns={name}))

    n_ratio2 = ["NOP", "NHP", "NLP"]
    for name in n_ratio2:
        list_.append(pd.DataFrame(np.log(df[name] / df["NCP"]), columns={name}))

    df_ratio = pd.concat(list_, axis=1)
    dffeat = pd.concat([df_diff, df_ratio], axis=1).replace([np.inf, -np.inf], np.nan)
    dffeat[["pCP", "pNCP", "OP", "HP", "LP"]] *= 30
    # dffeat[['xOP', 'xHP', 'xLP']] *= 30
    dffeat[["NOP", "NHP", "NLP"]] *= 60

    list_2 = []
    for n in range(lday):
        list_2.append(dffeat.add_prefix(str(n + 1) + "_").shift(n))
    out = pd.concat(list_2, axis=1)

    out = pd.concat([out, df[["LVOL", "CP"]]], axis=1)
    return out


lday = 10
holdout = 62
day_sample = 520
day_behind = 0

path = "./00-JPRAW/"
df_date = pd.read_csv(path + "0000" + ".csv")
if day_behind > 0:
    df_date = df_date[: (-1 * day_behind)]
df_date["DATE"] = pd.to_datetime(df_date["DATE"])
df_date = df_date.sort_values("DATE").set_index("DATE")
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
        df = pd.read_csv(path + k + ".csv")
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE").set_index("DATE")
        if day_behind > 0:
            df = df[: (-1 * day_behind)]
        df["VOL"] *= (df["OP"] + df["CP"]) / 2
        df["LVOL"] = (np.log(df["VOL"].rolling(lday).mean()) - 10) / 10
        df_plus = pd.concat([df_date, df], axis=1)
        df_plus["xOP"] = df_plus["OP"] / df_plus["NOP"]
        df_plus["xHP"] = df_plus["HP"] / df_plus["NHP"]
        df_plus["xLP"] = df_plus["LP"] / df_plus["NLP"]
        df_plus["xCP"] = df_plus["CP"] / df_plus["NCP"]
        # df_uni = feature(df_plus, lday) #for predict
        df_uni = pd.concat(
            [feature(df_plus, lday), label(df_plus, nday)], axis=1
        )  # for train
        df_uni = df_uni.replace(np.inf, np.nan).replace(-np.inf, np.nan)
        df_uni["day"] = df_uni.index.weekday
        df_uni["code"] = k
        df_uni["indus"] = l
        df_uni["scale"] = scale2

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

n_aug = 2


def augment(df):
    ignore_list = ["DATE", "day", "CP", "code", "indus", "scale", "dura", "LVOL"]
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

train.columns

train.shape


def prep(df):

    df = df[df["CP"] < 2000]
    df.drop("CP", axis=1, inplace=True)

    # df.drop('indus', axis = 1, inplace = True)

    # df = pd.get_dummies(df, columns=['day'])

    # df = df[df['dura'] <= 20]
    df.drop(["dura"], axis=1, inplace=True)
    # df['dura'] /= 5

    # df.drop("day", axis = 1, inplace = True)
    df = pd.get_dummies(df, columns=["day"])

    # df.drop("scale", axis=1, inplace=True)
    df = pd.get_dummies(df, columns=["scale"])

    # df.drop("indus", axis=1, inplace=True)
    df = pd.get_dummies(df, columns=["indus"])

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

n = 5
# さらに間引く
n_sample = 5000000
if len(train) > n_sample:
    train_b = train.sample(n_sample)
else:
    train_b = train

dgroup = train_b.drop_duplicates("DATE").sort_values("DATE").reset_index(drop=True)
dgroup["group"] = (n * dgroup.index / (dgroup.index.max() + 1)).astype(int)
dgroup = dgroup[["DATE", "group"]]
train_b = pd.merge(train_b, dgroup)
del train
gc.collect()

from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.knn.knn_rapids_model import KNNRapidsModel

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
        "max_depth": 5,
        "ag_args_fit": {"num_gpus": 1},
        "tree_method": "gpu_hist",
        "ag_args": {"name_suffix": "A"},
    },
    {
        "max_depth": 3,
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
    KNNRapidsModel: {
        "ag_args_fit": {"num_gpus": 1},
        "ag_args_ensemble": {"num_folds_parallel": 1},
    },
    "LR": {"ag_args_ensemble": {"num_folds_parallel": 12}},
    "XGB": xgb_options,
    # 'CAT': cat_options,
    # 'GBM': gbm_options,
    "XT": xt_options,
    # 'XGB': {'ag_args_fit': {'num_gpus': 1}, "ag_args_ensemble": {"num_folds_parallel": 1}},
    "CAT": {
        "ag_args_fit": {"num_gpus": 1},
        "ag_args_ensemble": {"num_folds_parallel": 1},
    },
    "GBM": [
        {
            "ag_args_fit": {"num_gpus": 1},
            "ag_args_ensemble": {"num_folds_parallel": 1},
            "extra_trees": True,
            "ag_args": {"name_suffix": "XT"},
        },
        {"ag_args_fit": {"num_gpus": 1}, "ag_args_ensemble": {"num_folds_parallel": 1}},
        # 'GBMLarge',
    ],
    # 'XT': {},
    # 'NN_TORCH': {'ag_args_fit': {'num_gpus': 1}, "ag_args_ensemble": {"num_folds_parallel": 2}},
    # 'FASTAI': {'ag_args_fit': {'num_gpus': 1}, "ag_args_ensemble": {"num_folds_parallel": 2}},
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
    train_data=train_b.drop(["DATE", "code", "RATE2"], axis=1),
    num_bag_folds=10,
    num_bag_sets=1,
    num_stack_levels=2,
    hyperparameters=hyperparameters,
    # hyperparameter_tune_kwargs = hyperparameter_tune_kwargs,
    save_space=True,
)

test.to_csv("test.csv")

test = pd.read_csv("test.csv")

pickle.dump(new, open("ag_model_new.mdl", "wb"))

new = pickle.load(open("ag_model_new.mdl", "rb"))

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
    pre = pickle.load(open("ag_model_best.mdl", "rb"))
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
    pickle.dump(new, open("ag_model_best.mdl", "wb"))

plt.scatter(df_te_new["RATE2"], df_te_new["pred"])

plt.scatter(df_te_pre["RATE2"], df_te_pre["pred"])

n = holdout * 2


def evalu_profit(df, thre, n):
    score = (df[df["pred"] >= thre]["RATE2"].mean()) * min(
        n, df[df["pred"] >= thre]["RATE2"].count()
    )
    return score


def optima(df, n):
    score = -math.inf
    mmax = 1000
    thre_max = df["pred"].max()
    thre_min = df["pred"].min()
    for m in range(mmax):

        thre = thre_min + (thre_max - thre_min) * m / mmax
        raw_score = evalu_profit(df, thre, n)
        if raw_score > score:
            score = raw_score
            thre_out = thre
    score = math.exp(score / n) - 1
    return score, thre_out


score_new, thre_new = optima(df_te_new, n)
score_pre, thre_pre = optima(df_te_pre, n)

slack("score_new =" + str(score_new) + ", thre_new =" + str(thre_new))
slack("score_pre =" + str(score_pre) + ", thre_pre =" + str(thre_pre))

df_te_new[df_te_new["pred"] >= thre_new].plot.scatter(
    x="RATE2", y="pred", xlim=(-0.2, 0.2)
)

df_te_pre[df_te_pre["pred"] >= thre_pre].plot.scatter(
    x="RATE2", y="pred", xlim=(-0.2, 0.2)
)

# 新旧モデル可視化
fig = plt.figure(figsize=(10, 6), facecolor="white")
plt.hist(
    df_te_new[df_te_new["pred"] >= thre_new]["RATE2"],
    bins=60,
    range=(-0.3, 0.3),
    alpha=0.5,
)
plt.hist(
    df_te_pre[df_te_pre["pred"] >= thre_pre]["RATE2"],
    bins=60,
    range=(-0.3, 0.3),
    alpha=0.5,
)
plt.title("new=" + str(score_new) + " / pre=" + str(score_pre))
plt.grid()
fig.savefig("hist_positive_ag.png")

if score_new >= score_pre:
    pickle.dump(new, open("ag_model_best.mdl", "wb"))
    print("モデル更新完了")
    score_best = score_new
    thre_best = thre_new
    slacker = "モデル更新完了" + str(score_best)
else:
    score_best = score_pre
    thre_best = thre_pre
    slacker = "モデルそのまま" + str(score_best)

ag_result = pd.DataFrame(
    [[score_new, thre_new], [score_pre, thre_pre], [score_best, thre_best]],
    columns=["score", "thre"],
    index=["new", "pre", "best"],
)
print(ag_result)
ag_result.to_csv("ag_result.csv")

slack(slacker)

