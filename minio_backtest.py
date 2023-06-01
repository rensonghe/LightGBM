from pyarrow import Table
from pyarrow import fs
import pyarrow.parquet as pq
import os
import pandas as pd
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
#%%
data = pd.read_csv('/songhe/BTCUSDT/btcusdt_20221010_1011_1s_success_ST6.0_20221017.csv')
data = data.iloc[:,1:]
print(data['predict'].describe())
#%%
data['symbol'] = 'btcusdt'
data['platform'] = 'gate_spot'
data['starttime'] = data['closetime']
data['endtime'] = data['closetime']+1500
data['side'] = 'buy_sell'
#%%
signal_long = 0.55
# signal_short = 0.417546
df_1 = data.loc[data['predict']>signal_long]
# df_0 = data.loc[data['predict']<signal_short]
#%% update backtest signal

dir_name = 'datafile/bt_record/songhe/{}'.format("btid=hft100ms_20221019_btcusdt")
pq.write_to_dataset(Table.from_pandas(df=df_1),
                    root_path=dir_name,
                    filesystem=minio, basename_template="part-{i}.parquet",
                    existing_data_behavior="overwrite_or_ignore")
#%% updated backtest needed data

symbol = 'btcusdt'
platform = 'gate_swap_u'


for month in range(10, 11):
    for day in range(10, 12):
        file_name = '{}_{}_{}_{}_{}depth.csv'.format(platform, symbol, year, month, day)
        test = pd.read_csv(file_name)
        pq.write_to_dataset(Table.from_pandas(df=test),
                            root_path='datafile/tick/order_book_100ms/gate_spot',
                            partition_cols=['symbol', 'year', 'month', 'day'],
                            filesystem=minio,
                            basename_template="part-{i}.parquet",
                            existing_data_behavior="overwrite_or_ignore")
        os.remove(file_name)

test = trade_info[trade_info['begin_id'] - trade_info['begin_id'].shift(1) - trade_info['merged_count'] != 0]