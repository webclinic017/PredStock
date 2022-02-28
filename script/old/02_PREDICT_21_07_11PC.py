#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import glob
import numpy as np
import random
import os
from urllib.request import Request, urlopen
# from bs4 import BeautifulSoup
import requests
from datetime import datetime
import time
# from pycaret.datasets import get_data
# from pycaret.classification import *
import smtplib, ssl
from email.mime.text import MIMEText
from tqdm import tqdm
import slackweb


# In[2]:


os.chdir('/home/toshi/STOCK')


# In[3]:


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


# In[4]:


data_j = get_data_j()
code_list  = data_j.drop(["indus", "scale"], axis=1).reset_index(drop = True)


# In[5]:


dplus = data_j.set_index('code', inplace = False).drop('scale', axis = 1)
dplus


# In[6]:


def index_dtime(df):
    df.index = pd.to_datetime(df.index)
    return df


# In[7]:


#モデルを作るための元データを読み込む
path ='./00-JPRAW/NEW300/'
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

dfop = index_dtime(dfop.sort_index().fillna(method='ffill', limit = 2))
dfhp = index_dtime(dfhp.sort_index().fillna(method='ffill', limit = 2))
dflp = index_dtime(dflp.sort_index().fillna(method='ffill', limit = 2))
dfcp = index_dtime(dfcp.sort_index().fillna(method='ffill', limit = 2))
dfvol = index_dtime(dfvol.sort_index().fillna(method='ffill', limit = 2))


# In[8]:


from pycaret.classification import *
model = load_model('./pycaret_model_best')


# In[9]:


lday = 10 #サンプル長
lday2 = int(lday * 0.5)

# holdout = 20 #ホールドアウト検定に使う日数
day_sample = 2000 #さかのぼる日数
code_sample = 3000 #選択株数


# In[10]:


nday = 0

#計算に使うデータフレームを用意する

#ラベルデータ
dflab = np.log(dfcp.pct_change() + 1).rolling(nday).sum().shift(-nday)
#dflab = (dflab > np.log(drate)) * 1 
#説明データ
dfdlog = np.log(dfcp.pct_change() + 1)
dfphp = np.log(dfhp/dfcp)
dfplp = np.log(dflp/dfcp)
dfpop = np.log(dfop/dfcp)
dfcplog = np.log(dfcp)
dfvlog = np.log(dfvol.pct_change() + 1)
date_list = dflab.sort_index(ascending=False).index.values
#検証のための日数、以下のブロックは予測処理の時だけ使う

if nday >= 1:
    dfcplog = dfcplog[:-nday]
    dfvlog = dfvlog[:-nday]
    dfdlog = dfdlog[:-nday]
    dfphp = dfphp[:-nday]
    dfplp = dfplp[:-nday]
    dfpop = dfpop[:-nday]

list2_ = []
list_te = []

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
dfpro = dfpro.reset_index()
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
# df2 = df2.dropna(how = 'any')
#ラベル付与
# df2['LABEL'] = df2['RATE'] > np.log(drate) * 1 
# df2['LABEL'] = df2['LABEL'].astype('int')
#     曜日付与
df2['day'] = iday  
df2['date'] = date
list_te.append(df2)
df_te = pd.concat(list_te)


# In[11]:


dffuture2 = predict_model(model, data=df_te, raw_score=True)
dffuture3 = dffuture2[dffuture2['Label'] == 1].sort_values('Score_1', ascending = False).set_index('index')
date = dfpop.reset_index().tail(1).iloc[0, 0]
# dffuture3.Score.to_csv('./02-JPLAB/LIST_'+str(date)+'.csv')
# dffuture3.Score.to_csv('./code_list_power.csv')
print(nday)
print(date)
print(dffuture3['Score_1'])
print('全行程完了')


# In[12]:


dffuture2


# In[13]:


df2[['dlog0', 'index']].sort_values('dlog0')


# In[14]:


# # dffuture2 = predict_model(loadmodel, data=df2.reset_index())
# dffuture3 = dffuture2[dffuture2['Label'] > 0.5].set_index('index')
# date = dfpop.reset_index().tail(1).iloc[0, 0]
# dffuture3.Label.to_csv('./02-JPLAB/LIST_'+str(date)+'.csv')
# dffuture3.Label.to_csv('./code_list_power.csv')


# In[15]:


date


# In[16]:


if len(dffuture3) == 0:
    print('NO BEST')
    subject_a = str(date) + ' の予報 ベストナシ' 
    body_a = 'NO BEST'
else:
    print('BEST')
    subject_a = str(date) + ' の予報 ' + str(dffuture3.reset_index().loc[:,'index'].iloc[0])
    body_a = 'BEST'
    for i  in range(10):
        if len(dffuture3) == i:
            break
        body_a = body_a + ' Index=' + str(dffuture3.reset_index().loc[:,'index'].iloc[i])
        body_a = body_a + '/Score=' + str(dffuture3.loc[:,'Score_1'].iloc[i]) + ' :'
    body_a = body_a + ' Length = ' + str(len(dffuture3))


# In[17]:


subject_a


# In[18]:


body_a


# In[19]:


try:
    slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
    slack.notify(text=body_a)
except:
    print(1)


# In[20]:


# import os
# os.system('systemctl poweroff') 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




