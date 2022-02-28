# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import random
import math
import os
# import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import slackweb
from sklearnex import patch_sklearn
patch_sklearn()


# %%
os.chdir('/home/toshi/STOCK')


# %%
#nday = int(input('先読み日数'))
nday = int(1)
print(nday)


# %%
#drate = 1 + 0.01 * float(input('変動閾値 %'))
drate = 1 + 0.01 * float(1.5 + 0.1* (1  - 2 * random.random()))
print(drate)


# %%
def index_dtime(df):
    df.index = pd.to_datetime(df.index)
    return df


# %%
def get_data_j():
    data_j = pd.read_excel('data_j.xls')[['コード','市場・商品区分', '17業種コード', '規模コード']]
    data_j = data_j[data_j['コード'] < 10000]
    
    #削除するものは有効にする
    data_j = data_j[data_j['市場・商品区分'] != 'ETF・ETN']
    data_j = data_j[data_j['市場・商品区分'] != 'JASDAQ(グロース・内国株）']
    data_j = data_j[data_j['市場・商品区分'] != 'JASDAQ(スタンダード・内国株）']
    data_j = data_j[data_j['市場・商品区分'] != 'JASDAQ(スタンダード・外国株）']
    data_j = data_j[data_j['市場・商品区分'] != 'PRO Market']
    data_j = data_j[data_j['市場・商品区分'] != 'REIT・ベンチャーファンド・カントリーファンド・インフラファンド']
    data_j = data_j[data_j['市場・商品区分'] != 'マザーズ（内国株）']
    data_j = data_j[data_j['市場・商品区分'] != 'マザーズ（外国株）']
    data_j = data_j[data_j['市場・商品区分'] != '出資証券']
#     data_j = data_j[data_j['市場・商品区分'] != '市場第一部（内国株）']
    data_j = data_j[data_j['市場・商品区分'] != '市場第一部（外国株）']
    data_j = data_j[data_j['市場・商品区分'] != '市場第二部（内国株）']
    data_j = data_j[data_j['市場・商品区分'] != '市場第二部（外国株）']
    data_j = data_j.drop("市場・商品区分", axis=1)
    data_j = data_j.rename(columns={'コード': 'code', '17業種コード': 'indus', '規模コード': 'scale'}).sort_values('code')
    data_j.index = data_j.index.astype(int) 
    return data_j


# %%
data_j = get_data_j()


# %%
def feature(df, lday):
#     time series
    n_diff = ['CP', 'VOL']
    df_diff = df[n_diff]
    df_diff['VOL'] *= df_diff['CP']
    df_diff= np.log(df_diff.pct_change() + 1)
    
    n_ratio = ['OP', 'HP', 'LP']
    list_ = []
    for name, n in zip(n_ratio, n_ratio):
        list_.append(pd.DataFrame(np.log(df[name].clip(lower = 1)/df['CP'].clip(lower = 1)), columns = {name}))
    df_ratio = pd.concat(list_, axis = 1)
    dffeat = pd.concat([df_diff, df_ratio], axis = 1).replace([np.inf, -np.inf], np.nan)
    
    list_2 = []
    for n in range(lday):
        list_2.append(dffeat.add_suffix('_' + str(n + 1)).shift(n))
    out = pd.concat(list_2, axis = 1)
    return out


# %%
# # CP_CP
# def label(df, nday, drate):
#     df_label = pd.DataFrame(np.log(df['CP'].clip(lower = 1).shift(-nday) / df['CP'].clip(lower = 1))).rename(columns = {'CP': 'RATE'})
#     # df_label['LABEL'] = df_label['RATE'] > np.log(drate)
#     df_label['LABEL1'] = df_label['RATE'] > np.log(drate)
#     df_label['LABEL2'] = df['LP'].shift(-1) <= df['CP']
#     df_label['LABEL'] = df_label['LABEL1'] & df_label['LABEL2']
#     df_label['LABEL'] = df_label['LABEL'] * 1 
#     df_label['LABEL'] = df_label['LABEL'].astype('int')
#     return df_label[['RATE', 'LABEL']]

# OP_CP
def label(df, nday, drate):
    df_label = pd.DataFrame(np.log(df['CP'].clip(lower = 1).shift(-nday) / df['OP'].clip(lower = 1).shift(-1)), columns = ['RATE'])
    df_label['LABEL1'] = df_label['RATE'] > np.log(drate)
    df_label['LABEL'] = df_label['LABEL1'].astype('int') * 1 
    return df_label[['RATE', 'LABEL']]


# %%
path ='./00-JPRAW/'
list_date = []
for i in data_j['code']:
    k = str(i)
    if os.path.exists(path + k +'.csv') == 1:
        df = pd.read_csv(path + k +'.csv')
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.sort_values('DATE').set_index('DATE')
        list_date.append(df)
df_date = pd.concat(list_date, axis = 1).sort_values('DATE')
df_date['dum'] = 1
df_date = df_date['dum']


# %%
lday = 10
holdout = 60
day_sample = 1000

n = 0
list_tr = []
list_te = []
for i, j in zip(data_j['code'], data_j['indus']):
    k = str(i)
    l = float(j)
    if os.path.exists(path + k +'.csv') == 1:
        df = pd.read_csv(path + k +'.csv')
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.sort_values('DATE').tail(lday + holdout + day_sample + nday + 1).set_index('DATE')
        df = df.sort_values('DATE').tail(1).set_index('DATE')
        df = pd.concat([df_date, df], axis = 1)
        # df.fillna(method = 'ffill', limit = 1, inplace = True)
        df_uni = pd.concat([feature(df, lday), label(df, nday, drate)], axis = 1).dropna()
        df_uni['day']= df_uni.index.weekday.astype('float')
        df_uni['code'] = k
        df_uni['indus'] = l
        list_te.append(df_uni.tail(holdout))
        list_tr.append(df_uni.shift(holdout).tail(day_sample))
test = pd.concat(list_te).reset_index()
train = pd.concat(list_tr).reset_index()
del list_te, list_tr


# %%
def prep(df):
    df = pd.get_dummies(df, columns=['indus', 'day'])
    return df


# %%
# train.dropna(inplace = True)
# test.dropna(inplace = True)
train = prep(train)
test = prep(test)


# %%
test = test[test['DATE'] ==test['DATE'].max()].reset_index(drop = True)


# %%
test


# %%
import pickle
pre = pickle.load(open('ag_model_best', 'rb'))


# %%
X_test = test.drop(['RATE', 'LABEL', 'DATE', 'code'], axis = 1)


# %%
predict = pre.predict_proba(X_test)


# %%
predict.rename(columns = {1: 'pred'})['pred']


# %%
dffuture2 = pd.concat([test,predict.rename(columns = {1: 'pred'})['pred']], axis = 1)


# %%
dffuture3 = dffuture2[dffuture2['pred'] > 0.5].sort_values('pred', ascending = False).set_index('code')
date = test['DATE'].tail(1).item()
# dffuture3.Score.to_csv('./02-JPLAB/LIST_'+str(date)+'.csv')
# dffuture3.Score.to_csv('./code_list_power.csv')
print(nday)
print(date)
print(dffuture3['pred'])
print('全行程完了')


# %%
dffuture3


# %%
if len(dffuture3) == 0:
    print('NO BEST')
    subject_a = str(date) + ' の予報 ベストナシ' 
    body_a = 'NO BEST'
else:
    print('BEST')
    subject_a = str(date) + ' の予報 ' + str(dffuture3.reset_index().loc[:,'code'].iloc[0])
    body_a = 'BEST'
    for i  in range(10):
        if len(dffuture3) == i:
            break
        body_a = body_a + ' Code=' + str(dffuture3.reset_index().loc[:,'code'].iloc[i])
        body_a = body_a + '/Score=' + str(dffuture3.loc[:,'pred'].iloc[i]) + ' :'
    body_a = body_a + ' Length = ' + str(len(dffuture3))


# %%
print(subject_a)


# %%
print(body_a)


# %%
try:
    slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
    slack.notify(text=body_a)
except:
    print(1)


# %%




