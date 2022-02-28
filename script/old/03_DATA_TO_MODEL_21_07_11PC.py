#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 曜日情報を追加
import pandas as pd
import datetime
import glob
import numpy as np
import random
import os
import time
import math
from pycaret.datasets import get_data
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# from scipy.stats import kendalltau 
# from scipy.stats import gaussian_kde
from tqdm import tqdm
import gc
import smtplib, ssl
from email.mime.text import MIMEText
import slackweb
# from sklearnex import patch_sklearn
# patch_sklearn()


# In[2]:


os.chdir('/home/toshi/STOCK')


# In[3]:


#nday = int(input('先読み日数'))
nday = int(1)
print(nday)
#drate = 1 + 0.01 * float(input('変動閾値 %'))
drate = 1 + 0.01 * float(1.9 + 0.3* (1  - 2 * random.random()))
print(drate)


# In[4]:


def evalu(model_new):
    n = 120
    # 新モデルを保存
    save_model(model_new, 'pycaret_model_new')
    # 新モデルで予測
    df_te_new = predict_model(model_new, data=df_te).dropna()
    #旧モデルを読んで予測。読めない、あるいは説明データに対応できない場合は、新モデルで上書きする。
    try:
        model_pre = load_model('pycaret_model_best')
        df_te_pre = predict_model(model_pre, data=df_te).dropna()
        print('旧モデル予測完了')
    except:
        model_pre = model_new
        df_te_pre = predict_model(model_pre, data=df_te).dropna()
        save_model(model_new, 'pycaret_model_best')
        print('旧モデル読み込みおよび予測失敗、新モデルで上書き')

    #ヒット数と質
    subject = "モデル学習完了"
    score_new = (df_te_new[df_te_new['Label'] == 1]['RATE'].mean() 
                 -  df_te_new[df_te_new['Label'] == 0]['RATE'].mean()) *  min(n, df_te_new[df_te_new['Label'] == 1]['RATE'].count())
    score_pre = (df_te_pre[df_te_pre['Label'] == 1]['RATE'].mean() 
                 -  df_te_pre[df_te_pre['Label'] == 0]['RATE'].mean()) *  min(n, df_te_pre[df_te_pre['Label'] == 1]['RATE'].count())
    score_new = math.exp(score_new / n) - 1
    score_pre = math.exp(score_pre / n) - 1

    #新旧モデル可視化
    fig = plt.figure(figsize = (10,6), facecolor="white")
    plt.hist(df_te_new[df_te_new['Label'] == 1]['RATE'], bins=60, range = (-0.3,0.3), alpha = 0.5)
    plt.hist(df_te_pre[df_te_pre['Label'] == 1]['RATE'], bins=60, range = (-0.3,0.3), alpha = 0.5)
    plt.title('new=' + str(score_new) +' / pre='+ str(score_pre))
    plt.grid()
    # plt.show()
    fig.savefig("hist_positive.png")

    if score_new >= score_pre:
        save_model(model_new, 'pycaret_model_best')
        subject = "モデル更新完了"
    print(subject)
    return score_new, max(score_new, score_pre)


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


code_list  = data_j.drop(["indus", "scale"], axis=1).reset_index(drop = True)


# In[9]:


dplus = data_j.set_index('code', inplace = False).drop('scale', axis = 1)


# In[10]:


#モデルを作るための元データを読み込む
path ='../STOCK/00-JPRAW/'
dfop = pd.DataFrame()
dfhp = pd.DataFrame()
dflp = pd.DataFrame()
dfcp = pd.DataFrame()
dfvol = pd.DataFrame()

list_op = []
list_hp = []
list_lp = []
list_cp = []
list_vol = []

n = 0
for i in tqdm(range(len(code_list))):
    k = str(code_list.loc[i,'code'])
    if os.path.exists(path + k +'.csv') == 1:
        df = pd.read_csv(path + k +'.csv').set_index('DATE')
        
        dfop = pd.DataFrame(df['OP']).rename(columns = {'OP': k}).groupby(level = 0).last()
        dfhp = pd.DataFrame(df['HP']).rename(columns = {'HP': k}).groupby(level = 0).last()
        dflp = pd.DataFrame(df['LP']).rename(columns = {'LP': k}).groupby(level = 0).last()
        dfcp = pd.DataFrame(df['CP']).rename(columns = {'CP': k}).groupby(level = 0).last()
        dfvol = pd.DataFrame(df['CP'] * df['VOL']).rename(columns = {0: k}).groupby(level = 0).last()
        
        list_op.append(dfop)
        list_hp.append(dfhp)
        list_lp.append(dflp)
        list_cp.append(dfcp)
        list_vol.append(dfvol)

dfop = pd.concat(list_op ,join='outer', axis = 1)
dfhp = pd.concat(list_hp, join='outer', axis = 1)
dflp = pd.concat(list_lp, join='outer', axis = 1)
dfcp = pd.concat(list_cp, join='outer', axis = 1)
dfvol = pd.concat(list_vol, join='outer', axis = 1)
del list_op
del list_hp
del list_lp
del list_cp
del list_vol

dfop = index_dtime(dfop.sort_index().fillna(method='ffill', limit = 1))
dfhp = index_dtime(dfhp.sort_index().fillna(method='ffill', limit = 1))
dflp = index_dtime(dflp.sort_index().fillna(method='ffill', limit = 1))
dfcp = index_dtime(dfcp.sort_index().fillna(method='ffill', limit = 1))
dfvol = index_dtime(dfvol.sort_index().fillna(method='ffill', limit = 1))


# In[11]:


dfop


# In[12]:


lday = 10 #サンプル長
lday2 = int(lday * 0.5)
holdout = 60 #ホールドアウト検定に使う日数
# day_sample = 200 + int(1600 * random.random()) #さかのぼる日数
day_sample = 1000
code_sample = 3000 #選択株数


# In[13]:


list_te = []
list_tr = []
m = 0
for n in range(1):

    #ラベルデータ
    dflab = np.log(dfcp.pct_change() + 1).rolling(nday).sum().shift(-nday)[:-nday]
    #dflab = (dflab > np.log(drate)) * 1 
    #説明データ
    dfdlog = np.log(dfcp.pct_change() + 1)[:-nday]
    dfphp = np.log(dfhp/dfcp)[:-nday]
    dfplp = np.log(dflp/dfcp)[:-nday]
    dfpop = np.log(dfop/dfcp)[:-nday]
    dfcplog = np.log(dfcp)[:-nday]
    dfvlog = np.log(dfvol.pct_change() + 1)[:-nday]
    date_list = dflab.sort_index(ascending=False).index.values
    
    raf = 2 ** m
    dflab = dflab * raf
    dfdlog = dfdlog * raf
    dfphp =  dfphp * raf
    dfplp = dfplp * raf
    dfpop = dfpop * raf
    dfcplog = dfcplog * raf
    dfvlog = dfvlog * raf

    #計算に使うデータフレームを用意する
    i = 0
    for idate in tqdm(date_list):
        #標本長よりデータフレームが短くなったら不完全データになるから中止。
        if len(dfdlog) < lday :
            break
        list2_ = []

        #説明変数と、ラベルをセットにしてデータフレームを組む

        dfpro = dfdlog.tail(lday).reset_index(drop=True)
        dfpro['lab1'] = 'dlog'
        dfpro['lab2'] = pd.RangeIndex(start=0, stop=len(dfpro.index) , step=1) 
        dfpro['lab3'] = dfpro['lab1'].astype(str).str.cat(dfpro['lab2'].astype(str))
        dfpro = dfpro.drop(['lab1','lab2'], axis = 1)
        dfpro = dfpro.set_index('lab3', drop=True).T
        dfpro = dfpro.reset_index().drop("index", axis=1)
        list2_.append(dfpro)    

        dfpro = dfphp.tail(lday2).reset_index(drop=True)
        dfpro['lab1'] = 'hp'
        dfpro['lab2'] = pd.RangeIndex(start=0, stop=len(dfpro.index) , step=1) 
        dfpro['lab3'] = dfpro['lab1'].astype(str).str.cat(dfpro['lab2'].astype(str))
        dfpro = dfpro.drop(['lab1','lab2'], axis = 1)
        dfpro = dfpro.set_index('lab3', drop=True).T
        dfpro = dfpro.reset_index().drop("index", axis=1)
        list2_.append(dfpro)

        dfpro = dfplp.tail(lday2).reset_index(drop=True)
        dfpro['lab1'] = 'lp'
        dfpro['lab2'] = pd.RangeIndex(start=0, stop=len(dfpro.index) , step=1) 
        dfpro['lab3'] = dfpro['lab1'].astype(str).str.cat(dfpro['lab2'].astype(str))
        dfpro = dfpro.drop(['lab1','lab2'], axis = 1)
        dfpro = dfpro.set_index('lab3', drop=True).T
        dfpro = dfpro.reset_index().drop("index", axis=1)
        list2_.append(dfpro)

        dfpro = dfpop.tail(lday2).reset_index(drop=True)
    #     -infは削除する
        dfpro = dfpro.mask(dfpro == float('-inf'), np.nan)
        dfpro['lab1'] = 'op'
        dfpro['lab2'] = pd.RangeIndex(start=0, stop=len(dfpro.index) , step=1) 
        dfpro['lab3'] = dfpro['lab1'].astype(str).str.cat(dfpro['lab2'].astype(str))
        dfpro = dfpro.drop(['lab1','lab2'], axis = 1)
        dfpro = dfpro.set_index('lab3', drop=True).T
        #予測の場合はdropを削除する
        dfpro = dfpro.reset_index().drop("index", axis=1)
        list2_.append(dfpro)

    #     dfpro = dfcplog.tail(lday).reset_index(drop=True)
    #     dfpro['lab1'] = 'cplog'
    #     dfpro['lab2'] = pd.RangeIndex(start=0, stop=len(dfpro.index) , step=1) 
    #     dfpro['lab3'] = dfpro['lab1'].astype(str).str.cat(dfpro['lab2'].astype(str))
    #     dfpro = dfpro.drop(['lab1','lab2'], axis = 1)
    #     dfpro = dfpro.set_index('lab3', drop=True).T
    #     dfpro = dfpro.reset_index().drop("index", axis=1)
    #     dfx = dfpro.iloc[:,lday-1]
    #     dfpro = (dfpro.T - np.array(dfx)).T
    #     list2_.append(dfpro)

        dfpro = dfvlog.tail(lday2).reset_index(drop=True)
        dfpro['lab1'] = 'vlog'
        dfpro['lab2'] = pd.RangeIndex(start=0, stop=len(dfpro.index) , step=1) 
        dfpro['lab3'] = dfpro['lab1'].astype(str).str.cat(dfpro['lab2'].astype(str))
        dfpro = dfpro.drop(['lab1','lab2'], axis = 1)
        dfpro = dfpro.set_index('lab3', drop=True).T
        dfpro = dfpro.reset_index().drop("index", axis=1)
        list2_.append(dfpro)    

        #追加情報
        dfpro2 = dplus
        dfpro2 = dfpro2.reset_index()
        list2_.append(dfpro2)

        #最後にラベルを付加
        dfpro3 = dflab.tail(1).reset_index(drop=True).rename(index={ 0 : 'RATE'}).T
        dfpro3 = dfpro3.reset_index().drop("index", axis=1)
        list2_.append(dfpro3)

    #     曜日を取得
        iday = dfdlog.tail(1).index.weekday.item()
        date = dfdlog.tail(1).index.date.item()

        #結合する
        df2 = pd.concat(list2_, axis = 1)
        #インデクスリセットと不要な列の削除
        df2 = df2.reset_index(drop=True)
        #df2 = df2.drop("index", axis=1).drop("code", axis=1)
        #不要な行の削除
        df2 = df2.dropna(how = 'any')
        #ラベル付与
        df2['LABEL'] = df2['RATE'] > np.log(drate) * 1 
        df2['LABEL'] = df2['LABEL'].astype('int')
    #     曜日付与
        df2['day'] = iday  
        df2['date'] = date

        #検証用
        if i < holdout:
            if n == 0:
                list_te.append(df2)
        #学習用
        else :
            if len(df2) >code_sample:
                df2 = df2.sample(code_sample)
            list_tr.append(df2)
        i = i + 1

        if i > day_sample + holdout:
            break
        #元のデータフレームの下一行をカット繰り返し。これにより1つづつ古いデータから生成するようになる。
        dflab = dflab[:-1]  
        dfcplog = dfcplog[:-1]
        dfvlog = dfvlog[:-1]
        dfdlog = dfdlog[:-1]
        dfphp = dfphp[:-1]
        dfplp = dfplp[:-1]
        dfpop = dfpop[:-1]
        #指定した回数に達したら中止。長すぎると終わらないから。
    m = (random.random() - 0.5) * 1.5

df_te = pd.concat(list_te).drop_duplicates().sort_values('date').reset_index(drop = True)
del list_te
df_te = df_te.drop_duplicates().sort_values('date').reset_index(drop = True)
df_tr = pd.concat(list_tr).drop_duplicates().sort_values('date').reset_index(drop = True)
del list_tr
df_tr = df_tr.drop_duplicates().sort_values('date').reset_index(drop = True)
gc.collect()


# In[14]:


# 外れ値を除去する
# df_tr = df_tr[df_tr['RATE'] > -0.485]
df_tr = df_tr[df_tr['RATE'] < 0.485]


# In[15]:


print(len(df_tr[df_tr['LABEL'] == 1]) / len(df_tr),'/', len(df_tr))


# In[16]:


# さらに間引く
n_sample = 10000000
if len(df_tr) > n_sample:
    df_trb = df_tr.sample(n_sample).reset_index(drop = True)
else :
    df_trb = df_tr


# In[17]:


print(len(df_trb[df_trb.LABEL== 1]) / len(df_trb))


# In[18]:


print(len(df_trb))
gc.collect()


# In[19]:


df_trb


# 218608samples

# In[20]:


from pycaret.classification import *
exp1 = setup(df_trb,
             target = 'LABEL',
             ignore_features = ['RATE', 'date', 'code'],
             train_size = 0.8,
            normalize = True, 
            transformation = True,
             data_split_shuffle = False,
             fold_strategy = 'timeseries',
             remove_multicollinearity = True,
             multicollinearity_threshold = 0.98,
             use_gpu = True,
            silent = True,
            )


# In[21]:


model = create_model('xgboost')


# In[22]:


score, score_best = evalu(model)
print('score =', score)


# In[23]:


model_e = ensemble_model(model, n_estimators = 30)


# In[24]:


score_e, score_best = evalu(model_e)
print('score_e =', score_e)


# In[25]:


try: 
    slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
    slack.notify(text="学習完了_score=" + str(score_best))
except:
    print(1)


# In[26]:


# import time
# time.sleep(60)
# import os
# os.system('systemctl poweroff') 


# In[ ]:





# In[ ]:




