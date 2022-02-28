#!/usr/bin/env python
# coding: utf-8

# In[1]:


# tp3
import pandas as pd
import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import slackweb
from sklearnex import patch_sklearn
patch_sklearn()


# In[2]:


os.chdir('/home/toshi/STOCK')


# In[3]:


#nday = int(input('先読み日数'))
nday = int(1)
print(nday)


# In[4]:


#drate = 1 + 0.01 * float(input('変動閾値 %'))
drate = 1 + 0.01 * float(1.0 + 0.1* (1  - 2 * random.random()))
print(drate)


# In[5]:


def index_dtime(df):
    df.index = pd.to_datetime(df.index)
    return df


# In[6]:


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


# In[7]:


data_j = get_data_j()


# In[8]:


def feature(df, lday):
#     time series
    n_diff = ['CP', 'VOL']
    df_diff = df[n_diff]
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


# In[9]:


def label(df, nday, drate):
    df_label = pd.DataFrame(np.log(df['CP'].clip(lower = 1).shift(-nday) / df['CP'].clip(lower = 1))).rename(columns = {'CP': 'RATE'})
    df_label['LABEL1'] = df_label['RATE'] > np.log(drate)
    df_label['LABEL2'] = df['LP'].shift(-1) <= df['CP']
    df_label['LABEL'] = df_label['LABEL1'] & df_label['LABEL2']
    df_label['LABEL'] = df_label['LABEL'].astype('int') * 1 
    return df_label[['RATE', 'LABEL']]


# In[10]:


path ='./00-JPRAW/'
list_date = []
for i in data_j['code']:
    k = str(i)
    if os.path.exists(path + k +'.csv') == 1:
        df = pd.read_csv(path + k +'.csv').set_index('DATE')
        list_date.append(df)
df_date = pd.concat(list_date, axis = 1).sort_values('DATE')
df_date['dum'] = 1
df_date = df_date['dum']


# In[11]:


lday = 10
holdout = 60
day_sample = 100

n = 0
list_tr = []
list_te = []
for i, j in zip(data_j['code'], data_j['indus']):
    k = str(i)
    l = float(j)
    if os.path.exists(path + k +'.csv') == 1:
        df = pd.read_csv(path + k +'.csv').sort_values('DATE').set_index('DATE')
        df = pd.concat([df_date, df], axis = 1)
        # df.fillna(method = 'ffill', limit = 1, inplace = True)
        df_uni = pd.concat([feature(df, lday), label(df, nday, drate)], axis = 1)
        df_uni['code'] = k
        df_uni['indus'] = l
        list_te.append(df_uni.tail(holdout))
        list_tr.append(df_uni.shift(holdout).tail(day_sample))
test = pd.concat(list_te).reset_index()
train = pd.concat(list_tr).reset_index()
del list_te, list_tr


# In[12]:


def prep(df):
    df = pd.get_dummies(df, columns=['indus'])
    return df


# In[13]:


train.dropna(inplace = True)
test.dropna(inplace = True)
train = prep(train)
test = prep(test)


# In[14]:


# 外れ値を除去する
train = train[train['RATE'] > -0.485]
train = train[train['RATE'] < 0.485]


# In[15]:


print(len(train[train['LABEL'] == 1]) / len(train),'/', len(train))


# In[16]:


# さらに間引く
n_sample = 1000000
if len(train) > n_sample:
    train_b = train.sample(n_sample).reset_index(drop = True)
else :
    train_b = train


# In[17]:


print(len(train_b[train_b.LABEL== 1]) / len(train_b),'/', len(train_b))


# In[18]:


#     代入する
X_train = train_b.drop(['RATE', 'LABEL', 'DATE', 'code'], axis = 1)
y_train = train_b['LABEL']
X_test = test.drop(['RATE', 'LABEL', 'DATE', 'code'], axis = 1)
y_test = test['LABEL']
gc.collect()


# In[21]:


from tpot import TPOTClassifier
tpot = TPOTClassifier(generations = 100,
                    early_stop = 10,
                      population_size = 50,
                      # offspring_size = 10,
                      # cv=10,
                      # config_dict='TPOT NN',
                      # config_dict='TPOT light',
                      config_dict = 'TPOT cuML',
                      scoring = 'accuracy',
                      memory = 'auto',
                      # warm_start = True,
                      max_time_mins = 60 * 24,
                      # max_eval_time_mins = 7,
                      # use_dask = True,
                     verbosity = 2,
                      # n_jobs = 6,
                      template = 'Selector-Transformer-Classifier',
                     )


# In[22]:


#     学習
tpot.fit(X_train, y_train)


# In[ ]:


# 新モデルを保存
pd.to_pickle(obj=tpot.fitted_pipeline_, filepath_or_buffer='tpot_model_new.pkl')


# In[ ]:


# 新モデルで予測
df_te_new = pd.concat([test,pd.DataFrame(tpot.predict(X_test),columns = ['pred'])], axis = 1).dropna()


# In[ ]:


#旧モデルを読んで予測。読めない、あるいは説明データに対応できない場合は、新モデルで上書きする。
try:
    tpot_pre = pd.read_pickle('tpot_model_best.pkl')
    df_te_pre = pd.concat([test,pd.DataFrame(tpot_pre.predict(X_test),columns = ['pred'])], axis = 1).dropna()
    print('旧モデル予測完了')
except:
    tpot_pre = tpot
    df_te_pre = pd.concat([test,pd.DataFrame(tpot_pre.predict(X_test),columns = ['pred'])], axis = 1).dropna()
    pd.to_pickle(obj=tpot.fitted_pipeline_, filepath_or_buffer='tpot_model_best.pkl')
    print('旧モデル読み込みおよび予測失敗、新モデルで上書き')


# In[ ]:


n = holdout * 2
#ヒット数と質
subject = "モデル学習完了"
score_new = (df_te_new[df_te_new['pred'] == 1]['RATE'].mean() 
             -  df_te_new[df_te_new['pred'] == 0]['RATE'].mean()) *  min(n, df_te_new[df_te_new['pred'] == 1]['RATE'].count())
score_pre = (df_te_pre[df_te_pre['pred'] == 1]['RATE'].mean() 
             -  df_te_pre[df_te_pre['pred'] == 0]['RATE'].mean()) *  min(n, df_te_pre[df_te_pre['pred'] == 1]['RATE'].count())
score_new = math.exp(score_new / n) - 1
score_pre = math.exp(score_pre / n) - 1

#新旧モデル可視化
fig = plt.figure(figsize = (10,6), facecolor="white")
plt.hist(df_te_new[df_te_new['pred'] == 1]['RATE'], bins=60, range = (-0.3,0.3), alpha = 0.5)
plt.hist(df_te_pre[df_te_pre['pred'] == 1]['RATE'], bins=60, range = (-0.3,0.3), alpha = 0.5)
plt.title('new=' + str(score_new) +' / pre='+ str(score_pre))
plt.grid()
# plt.show()
fig.savefig("hist_positive_tpot.png")

if score_new >= score_pre or math.isnan(score_pre):
    pd.to_pickle(obj=tpot.fitted_pipeline_, filepath_or_buffer='tpot_model_best.pkl')
    subject = "モデル更新完了"
print(subject)
score = max(score_new,score_pre)


# In[ ]:


slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
slack.notify(text="TP学習完了_score=" + str(score))


# In[ ]:


# import time
# time.sleep(60)
# import os
# os.system('systemctl poweroff') 


# In[ ]:




