#!/usr/bin/env python
# coding: utf-8

# In[1]:


# tp2
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


#nday = int(input('先読み日数'))
nday = int(2)
print(nday)
#drate = 1 + 0.01 * float(input('変動閾値 %'))
drate = 1 + 0.01 * float(2.3 + 0.2 * random.random())
print(drate)


# In[3]:


np.log(drate) 


# In[4]:


os.chdir('/home/toshi/STOCK')


# In[5]:


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


# In[6]:


data_j = get_data_j()
code_list  = data_j.drop(["indus", "scale"], axis=1).reset_index(drop = True)


# In[7]:


code_list


# In[8]:


dplus = data_j.set_index('code', inplace = False).drop('scale', axis = 1)
dplus


# In[9]:


dplus.astype(int).describe()


# In[10]:


def index_dtime(df):
    df.index = pd.to_datetime(df.index)
    return df


# In[11]:


#モデルを作るための元データを読み込む
path ='./00-JPRAW/'
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


# In[12]:


list_op = []
list_hp = []
list_lp = []
list_cp = []
list_vol = []
gc.collect()


# In[13]:


lday = 10 #サンプル長
lday2 = int(lday * 0.5)
holdout = 60 #ホールドアウト検定に使う日数
day_sample = 1000 #さかのぼる日数
code_sample = 3000 #選択株数


# In[14]:


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
        iday_sin = math.sin(iday / 5 * 2 * math.pi)
        iday_cos = math.cos(iday / 5 * 2 * math.pi)

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
        df2['day_sin'] = iday_sin  
        df2['day_cos'] = iday_cos  
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


# In[15]:


# df_tr = df_tr[df_tr['RATE'] > -0.485]
df_tr = df_tr[df_tr['RATE'] < 0.485]


# In[16]:


df_tr = pd.get_dummies(df_tr, columns=['indus'])
df_te = pd.get_dummies(df_te, columns=['indus'])


# In[17]:


print(len(df_te), len(df_tr))


# In[30]:


# さらに間引く
n_sample = 1000000
if len(df_tr) > n_sample:
    df_trb = df_tr.sample(n_sample).reset_index(drop = True)
else :
    df_trb = df_tr


# In[31]:


len(df_trb[df_trb.LABEL== 1]) / len(df_trb)


# In[33]:


len(df_trb)


# In[34]:


df_trb


# In[35]:


#     代入する
X_train = df_trb.drop(['RATE', 'LABEL', 'date', 'code'], axis = 1)
y_train = df_trb['LABEL']
X_test = df_te.drop(['RATE', 'LABEL', 'date', 'code'], axis = 1)
y_test = df_te['LABEL']
gc.collect()


# In[36]:


from tpot import TPOTClassifier
tpot = TPOTClassifier(generations = 100,
                      early_stop = 10,
                      population_size = 50,
                      # offspring_size = 10,
                      # cv=10,
                      # config_dict='TPOT NN',
                      # config_dict='TPOT light',
#                       config_dict = 'TPOT cuML',
                      # scoring = 'precision',
                      # memory = 'auto',
                      # warm_start = True,
                      max_eval_time_mins = 30,
#                      use_dask = True,
                      verbosity = 3,
                      n_jobs = 4,
                      # template = 'Selector-Transformer-Classifier',
                      )


# In[37]:


#     学習
tpot.fit(X_train, y_train)


# In[39]:


# 新モデルを保存
pd.to_pickle(obj=tpot.fitted_pipeline_, filepath_or_buffer='tpot_model_new.pkl')


# In[40]:


# 新モデルで予測
df_te_new = pd.concat([df_te,pd.DataFrame(tpot.predict(X_test),columns = ['Label'])], axis = 1).dropna()


# In[54]:


#旧モデルを読んで予測。読めない、あるいは説明データに対応できない場合は、新モデルで上書きする。
try:
    tpot_pre = pd.read_pickle('tpot_model_best.pkl')
    df_te_pre = pd.concat([df_te,pd.DataFrame(tpot_pre.predict(X_test),columns = ['Label'])], axis = 1).dropna()
    print('旧モデル予測完了')
except:
    tpot_pre = tpot
    df_te_pre = pd.concat([df_te,pd.DataFrame(tpot_pre.predict(X_test),columns = ['Label'])], axis = 1).dropna()
    pd.to_pickle(obj=tpot.fitted_pipeline_, filepath_or_buffer='tpot_model_best.pkl')
    print('旧モデル読み込みおよび予測失敗、新モデルで上書き')


# In[55]:


n = holdout * 2
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
fig.savefig("hist_positive_tpot.png")

if score_new >= score_pre or math.isnan(score_pre):
    pd.to_pickle(obj=tpot.fitted_pipeline_, filepath_or_buffer='tpot_model_best.pkl')
    subject = "モデル更新完了"
print(subject)
score = max(score_new,score_pre)


# In[43]:


slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
slack.notify(text="TP学習完了_score=" + str(score))


# In[ ]:


# import os
# import time
# time.sleep(10)
# os.system('systemctl poweroff') 


# In[ ]:




