#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import accuracy_score,roc_auc_score, classification_report
from sklearn import metrics
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import time
from functools import reduce
import pyarrow
from pyarrow import fs
import pyarrow.parquet as pq
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
#%%
depth = pd.DataFrame()
symbol = 'btcusdt'
year = 2022
for month in tqdm(range(9, 10)):
    # 拿orderbook+trade的数据
    # data_type = ''
    filters = [('symbol', '=', symbol), ('year', '=', year), ('month','=',month)]
    dataset_base = pq.ParquetDataset('datafile/tick/order_book_100ms/gate_spot', filters=filters, filesystem=minio)
    trade_base = dataset_base.read_pandas().to_pandas()
    trade_base = trade_base.iloc[:, :-6]

    depth = depth.append(trade_base)
    # print(all_data)

#%%
del trade_base, dataset_base
#%%
data_1 = pd.read_csv('/songhe/BTCUSDT/btcusdt_20221010_1011_200ms_success_1_ST5.0_20221015.csv')
data_0 = pd.read_csv('/songhe/BTCUSDT/btcusdt_20221010_1011_200ms_success_0_ST5.0_20221015.csv')
data_singal = pd.read_csv('/songhe/BTCUSDT/btcusdt_20221010_1011_200ms_wap1_ST5.0_20221015.csv')
#%%
data_1 = data_1.loc[:,['closetime','predict','target']]
data_0 = data_0.loc[:,['closetime','predict','target']]
data_singal = data_singal.loc[:,['closetime', 'predict', 'target']]
#%%
data = pd.merge(data_1, data_0, on='closetime', how='outer')
# data = pd.merge(data, data_singal, on='closetime', how='outer')
#%%
data['predict_x'] = np.where(data['predict_x']>0.618670,1,0)
data['predict_y'] = np.where(data['predict_y']>0.570367,1,0)
#%%
data['success'] = np.where((data['predict_x']==1)&(data['predict_y']==1),1,0)
#%%
test = data[data['success'].isin([1])]
#%%
signal_long = 0.509873
signal_short = 0.509873
df_1 = test.loc[test['predict']>signal_long]
df_0 = test.loc[test['predict']<signal_short]
#%%
test_1 = len(df_1[(df_1.predict>0.509873)&(df_1.target>0)])/len(df_1)
df_0['target'] = np.where(df_0['target']>0,1,-1)
test_0 = len(df_0[(df_0.predict<0.509873)&(df_0.target<0)])/len(df_0)
print(test_1)
print(test_0)
#%%
test['target'] = np.where((test['target_x']==1)&(test['target_y']==1),1,0)
#%%
len(df_0[(df_0.predict<0.509873)&(df_0.target<0)])/len(df_0)