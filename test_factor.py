import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce
import pyarrow
from pyarrow import fs
import pyarrow.parquet as pq
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
# month = 1
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
#%%
# all_data = pd.DataFrame()
symbol = 'btcusdt'
platform = 'gate_swap_u'
year = 2022
for month in tqdm(range(8, 9)):
    # 拿orderbook+trade的数据
    # data_type = ''
    filters = [('symbol', '=', symbol), ('year', '=', year), ('month','=',month)]
    dataset = pq.ParquetDataset('datafile/tick/tick_1s/gate_swap_u', filters=filters, filesystem=minio)
    tick_1s = dataset.read_pandas().to_pandas()
    tick_1s = tick_1s.iloc[:, :-3]
#%%
tick_1s['datetime'] = pd.to_datetime(tick_1s['closetime']+28800, unit='s')
tick_1s = tick_1s.sort_values(by='closetime', ascending=True)
tick_1s = tick_1s.reset_index(drop=True)
#%%
tick_1s_first = tick_1s.iloc[:10000,:]
#%%
trade_data = trade_preprocessor(tick_1s_first, rolling=60)
trade_data['datetime'] = pd.to_datetime(trade_data['closetime']+28800, unit='s')
#%%
a = trade_data.set_index('datetime')
print(a[a.index>='2022-08-01 02:30:00']['log_return'])