# %%
import os
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pandas as pd
import requests
import datetime
import time
import xlrd
import smtplib, ssl
from email.mime.text import MIMEText
import html5lib
import slackweb
import numpy as np

# %%
os.chdir('/home/toshi/PROJECTS/PredStock')

# %%
def slack(txt):
    print(txt)
    try:
        slack = slackweb.Slack(url = pd.read_csv("slackkey.csv")["0"].item())
        slack.notify(text = txt)
    except:
        print("slack_error")

# %%
slack("scraper開始")

# %%
# try:
#     slack = slackweb.Slack(url = pd.read_csv("slackkey.csv")["0"].item())
#     slack.notify(text='scraper開始')
# except:
#     print('slack_not_available')

# %%
#ダウンロードする株価の種別を決める
def get_data_j():
    data_j = pd.read_excel('data_j.xls')[['コード','市場・商品区分', '17業種コード', '規模コード']]
    
#     削除するものは有効にする
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
    data_j = data_j[data_j['code'] < 10000]
    return data_j

data_j = get_data_j()
data_j['code'] = data_j['code'].astype('str')
data_j = data_j.append({'code': '0000', 'indus': np.nan, 'scale': np.nan}, ignore_index=True).sort_values('code').reset_index(drop = True)
code_list  = data_j.drop(["indus", "scale"], axis=1).reset_index(drop = True)

# %%
data_j.head()

# %%
# code_list = code_list.head(1)

# %%
code_list

# %%
def get_dfs_new(code):
    time.sleep(1)
#     try:
    url = 'https://kabutan.jp/stock/kabuka?code=' + str(code)
    data = pd.read_html(url)
    df = pd.concat([data[4].rename(columns={'本日': '日付'}), data[5]])
    df.drop(['前日比', '前日比％'], axis = 1,inplace = True)
    df.drop(df[df['始値'].astype(str) == '－'].index, inplace = True)
    df = df.rename(columns={'日付': 'DATE'})
    df = df.rename(columns={'始値': 'OP'})
    df = df.rename(columns={'高値': 'HP'})
    df = df.rename(columns={'安値': 'LP'})
    df = df.rename(columns={'終値': 'CP'})
    df = df.rename(columns={'売買高(株)': 'VOL'})
    df['DATE'] = pd.to_datetime(df['DATE'], format='%y/%m/%d')
    df = df.sort_values('DATE')
    df = df.set_index('DATE', drop = True)
    df = df.astype(float)
    return df
#     except:
#         return pd.DataFrame()

# %%
get_dfs_new(1301)

# %%
def get_dfs3(stock_number):

    dfs = []
    for y in range(2, 10):
        print(y)
        time.sleep(1)
        url = 'https://kabutan.jp/stock/kabuka?code=' + str(stock_number) + '&ashi=day&page=' + str(y)
        try:
            df = pd.read_html(url)[5]
            df.drop(['前日比', '前日比％'], axis = 1,inplace = True)
            df.drop(df[df['始値'].astype(str) == '－'].index, inplace = True)
    #         print(df)
            df = df.rename(columns={'日付': 'DATE'})
            df = df.rename(columns={'始値': 'OP'})
            df = df.rename(columns={'高値': 'HP'})
            df = df.rename(columns={'安値': 'LP'})
            df = df.rename(columns={'終値': 'CP'})
            df = df.rename(columns={'売買高(株)': 'VOL'})
            df['DATE'] = pd.to_datetime(df['DATE'], format='%y/%m/%d')
            df = df.sort_values('DATE')
            df = df.set_index('DATE', drop = True)
            df = df.astype(float)
            dfs.append(df)
        except:
            print('Cannot get')
    return pd.concat(dfs).sort_index()

# %%
#複数のデータフレームをcsvで保存
for i, k in enumerate(code_list['code']):
    print(k)
#     データが今日のものならば、すでに取得済みとしてスキップ
    today = datetime.datetime.now().date()
    try:
        ddate = pd.to_datetime(pd.read_csv('./00-JPRAW/NEW300/{}.csv'.format(k))['DATE']).tail(1).item().date()
    except:
        ddate = 1
    if today == ddate:
        continue
    try:
        data = get_dfs_new(k)
    except:
        print('skip')
        continue
    if len(data) == 0:
        continue
    if len(data) > 0:
        #最新*日のデータは逆順になっているが、ソートすればOK
        data.to_csv('./00-JPRAW/NEW300/{}.csv'.format(k))
    try :
        data_old = pd.read_csv('./00-JPRAW/{}.csv'.format(k))
        try:
            data_old['CPA'] is None
            data_old = data_old.drop('CPA', axis = 1)
            print('drop CPA')
        except:
            a=1
        data_old['DATE'] = pd.to_datetime(data_old['DATE'])
        data_old.set_index('DATE', inplace=True)
        #x = len(pd.concat([data,data_old])) - pd.concat([data,data_old]).duplicated().value_counts()[False]
        x = (pd.concat([data_old,data]).duplicated() * 1).sum()
    except :
        x = -1
    print('重複データ数 ' + str(x))
    #重複データ数が*を下回るなら、内容は変わっているから、とりなおす
    if x < 2 :
        print(data_old)
        print('取り直し')
        data_old = pd.concat([data_old, get_dfs3(k)])
    data = pd.concat([data_old, data]).drop_duplicates().rename(columns = {0: int(k)}).groupby(level = 0).last().sort_index()
    data = data.sort_index()
    data.to_csv('./00-JPRAW/{}.csv'.format(k))

# %%
print('株価更新完了')

# %%
slack("kabuka_DL_done")

# %%



