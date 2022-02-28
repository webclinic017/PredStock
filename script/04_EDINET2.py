# %%
from datetime import datetime, timedelta
import datetime
import requests
import json
import zipfile
import glob
import pandas as pd
import glob
import os
import shutil
from tqdm import tqdm

# %%
os.chdir('/home/toshi/STOCK/XBRL/')

# %%
def fetch_xbrl(date):
    print(date)
    # 書類一覧APIのエンドポイント
    url = "https://disclosure.edinet-fsa.go.jp/api/v1/documents.json"
    # 書類一覧APIのリクエストパラメータ
    params = {
      "date" : date,
      "type" : 2
    }

    # 書類一覧APIの呼び出し
    res = requests.get(url, params=params, verify=False)

    # resultデータ取得
    res_text = json.loads(res.text)
    results= res_text["results"]

    # 決算データに絞る
    kessan = []
    for result in results:
        if result['docDescription'] is not None:
            if '四半期' in result['docDescription']:
                kessan.append(result)
#     print(kessan)
    if len(kessan) > 0:
        print(len(kessan))
    # zipファイル取得
    for i in range(len(kessan)):
        docid = kessan[i]['docID']
        if not glob.glob('./ZIP_GET_CSV/' + docid + '.zip') :
            # if not glob.glob('./ZIP_NOT_CSV/' + docid + '.zip') :
            url = 'https://disclosure.edinet-fsa.go.jp/api/v1/documents/' + docid
            params = {
                "type" : 1
            }
            res = requests.get(url, params = params, verify=False)

            # ファイルへ出力
            filename = "./ZIP/" + docid + ".zip"
            print(filename)
            if res.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=1024):
                        f.write(chunk)

# %%
# 日付のリスト生成()
date_list = [datetime.datetime.now() + timedelta(days=-i)  for i in range(0,30)]
# 文字列に変換
date_str_list = [d.strftime("%Y-%m-%d") for d in date_list]
print(date_str_list[0])

# %%
from joblib import Parallel, delayed

# %%
r = Parallel(n_jobs = 20)( [delayed(fetch_xbrl)(i) for i in date_str_list] )

# %%



