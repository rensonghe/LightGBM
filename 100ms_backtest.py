#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from functools import reduce
import pyarrow
from pyarrow import fs
import pyarrow.parquet as pq
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
#%%
all_data = pd.DataFrame()
symbol = 'btcusdt'
platform = 'gate_swap_u'
year = 2022
for month in tqdm(range(6, 7)):
    # 拿orderbook+trade的数据
    # data_type = ''
    filters = [('symbol', '=', symbol), ('year', '=', year), ('month','=',month)]
    dataset_base = pq.ParquetDataset('datafile/feat/trade_1s_base/gate_swap_u', filters=filters, filesystem=minio)
    trade_base = dataset_base.read_pandas().to_pandas()
    trade_base = trade_base.iloc[:, :-3]

#%%
all_data['datetime'] = pd.to_datetime(all_data['closetime']+28800000, unit='ms')
all_data = all_data.sort_values(by='datetime', ascending=True)
#%%
# all_data['last_price_vwap'] = (all_data['last_price']*all_data['size']).rolling(120).sum()/all_data['size'].rolling(120).sum()
#%%
test_high_low = all_data.set_index('datetime').groupby(pd.Grouper(freq='1s')).agg({'price':['min','max']})
test_high_low.columns = ['_'.join(col) for col in test_high_low.columns]
test_high_low['diff'] = test_high_low['price_max']-test_high_low['price_min']
df1 = test_high_low[~test_high_low['diff'].isin([0])]
#%%
df2 = all_data.set_index('datetime').groupby(pd.Grouper(freq='1s')).agg('last')
# df2_1 = all_data.set_index('datetime').groupby(pd.Grouper(freq='1s')).agg({'price_vwap': ['mean']})
# df2_1.columns = ['_'.join(col) for col in df2_1.columns]
# df2 = pd.merge(df2, df2_1, on='datetime')
#%%
df3 = pd.merge(df1, df2, on='datetime', how='left')
#%%
df3 = df3.loc[:,['closetime', 'price_min', 'price_max', 'wap1']]
#%%
df3 = df3.dropna(axis=0)
#%%
all_data = all_data.loc[:,['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1',
       'ask_price2', 'ask_size2', 'bid_price2', 'bid_size2', 'ask_price3',
       'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4',
       'bid_price4', 'bid_size4','last_price','size','datetime']]
all_data['closetime'] = all_data['closetime'].astype('int')
#%%
df3['mid'] = df3['wap1'].shift(1)
df3['bid'] = np.where(df3['price_min']<df3['wap1']*0.99995,1,0)
df3['ask'] = np.where(df3['price_max']>df3['wap1']*1.00005,1,0)
#%%
df3_1 = df3[~df3['bid'].isin([0])]
df3_0 = df3[~df3['ask'].isin([0])]
#%%
df3_1['closetime'] = df3_1['closetime'] - 1000
df3_0['closetime'] = df3_0['closetime'] - 1000
#%%
data = pd.read_csv('/songhe/BTCUSDT/btcusdt_20221010_1011_1s_success_ST6.0_20221017.csv')
data = data.iloc[:,1:]
print(data['predict'].describe())
#%%
signal_long = 0.55
# signal_short = 0.417546
df_1 = data.loc[data['predict']>signal_long]
# df_0 = data.loc[data['predict']<signal_short]
#%%
df3_1['closetime'] = df3_1['closetime'].astype(int)
df_1_df3_1 = pd.merge(df_1, df3_1, on='closetime')
df3_0['closetime'] = df3_0['closetime'].astype(int)
df_0_df3_0 = pd.merge(df_0, df3_0, on='closetime')

#%%
time_1 = []
# time_ask = []
# wap1_time_1 = []
num = 0
for i, time in enumerate(df_1_df3_1['closetime']):
    # print(time)
    dt_1 = all_data[all_data['closetime'] == time + 5000].values
    dt_ask = all_data[all_data['closetime'] == time].values
    if len(dt_1) > 0:
        # time_1.append(dt_1[0][5])
        time_ask.append(dt_ask[0][11])
        # wap1_time_1.append(dt_1[0][23])
    else:
        time_1.append(0)
        # time_ask.append(0)
        # wap1_time_1.append(0)
# df_1_df3_1['buy_price'] = time_ask
df_1_df3_1['sell_price'] = time_1
# df_1_df3_1['wap1_time_1'] = wap1_time_1
print(df_1_df3_1)

time_0 = []
# time_bid = []
# wap1_time_0 = []
num = 0
for i, time in enumerate(df_0_df3_0['closetime']):
    # print(time)
    dt_0 = all_data[all_data['closetime'] == time + 1000].values
    # dt_bid = all_data[all_data['closetime'] == time].values
    if len(dt_0) > 0:
        time_0.append(dt_0[0][3])
        # time_bid.append(dt_bid[0][13])
        # wap1_time_0.append(dt_0[0][23])
    else:
        time_0.append(0)
        # time_bid.append(0)
        # wap1_time_0.append(0)
# df_0_df3_0['sell_price'] = time_bid
df_0_df3_0['buy_price'] = time_0
# df_0_df3_0['wap1_time_0'] = wap1_time_0
print(df_0_df3_0)
#%%
df_1_df3_1['wap1_return'] = np.log((df_1_df3_1['sell_price'])/(df_1_df3_1['wap1_x']))
df_1_df3_1.replace(-np.inf, 0, inplace=True)
# print(np.sum(df_1_df3_1['wap1_return']))
df_0_df3_0['wap1_return'] = np.log(df_0_df3_0['buy_price']/df_0_df3_0['wap1_x'])
df_0_df3_0.replace(-np.inf, 0, inplace=True)
# print(np.sum(df_0_df3_0['wap1_return']))

print('阈值:{} 挂过价多单粗略成交率: {:.2%}'.format(signal_long, len(df_1_df3_1)/len(df3_1)))
print('阈值:{} 粗略多单成交总利润: {:.6f}'.format(signal_long, df_1_df3_1['wap1_return'].sum()))
print('阈值:{} 挂过价空单粗略成交率: {:.2%}'.format(signal_short, len(df_0_df3_0)/len(df3_0)))
print('阈值:{} 粗略空单成交总利润均值: {:.6f}'.format(signal_short, (-1)*(df_0_df3_0['wap1_return'].sum())))

#%%
time_ask_1s = []
time_bid_1s = []
time_ask = []
time_bid = []
num = 0
for i, time in enumerate(df_1['closetime']):
    # print(time)
    dt_1 = all_data[all_data['closetime'] == time + 5000].values
    dt_ask = all_data[all_data['closetime'] == time].values
    if len(dt_1) > 0:
        time_ask_1s.append(dt_1[0][5])
        time_bid_1s.append(dt_1[0][7])
        time_ask.append(dt_ask[0][5])
        time_bid.append(dt_ask[0][7])
    else:
        time_ask_1s.append(0)
        time_bid_1s.append(0)
        time_ask.append(0)
        time_bid.append(0)
df_1['ask_price1_1s'] = time_ask_1s
df_1['bid_price1_1s'] = time_bid_1s
df_1['ask_price1'] = time_ask
df_1['bid_price1'] = time_bid

# time_0 = []
# time_bid = []
# num = 0
# for i, time in enumerate(df_0['closetime']):
#     # print(time)
#     dt_0 = all_data[all_data['closetime'] == time + 5000].values
#     # dt_bid = all_data[all_data['closetime'] == time].values
#     if len(dt_0) > 0:
#         time_0.append(dt_0[0][3])
#         # time_bid.append(dt_bid[0][-1])
#     else:
#         time_0.append(0)
#         # time_bid.append(0)
# df_0['buy_price'] = time_0
# df_0['sell_price'] = time_bid
#%%
df_1['return'] = np.where(df_1['target']>0,np.log((df_1['sell_price'])/(df_1['buy_price'])), np.log((df_1['buy_price_1s'])/(df_1['buy_price'])))
df_1.replace(-np.inf,0,inplace=True)
# df_0['wap1_return'] = np.log((df_0['mid_price3'])/(df_0['sell_price']))
# df_0.replace(np.inf,0,inplace=True)

# print(df_1['return'].mean())
# print((-1)*df_0['wap1_return'].mean())
#%% 双边策略
all_data['datetime'] = pd.to_datetime(all_data['closetime']+28800000, unit='ms')
df = all_data.set_index('datetime').groupby(pd.Grouper(freq='5000ms')).apply('last')
df = pd.merge(df, price_max_min, on='closetime', how='inner')
df = df.sort_values(by='closetime', ascending=True)
# df['wap1'] = calc_wap1(df)
df['buy_success'] = (df['bid_price1'].shift(1))-df['price_min']
df['sell_success'] = df['price_max']-(df['ask_price1'].shift(1))
df['buy_success'] = np.where(df['buy_success']>0, 1, df['buy_success'])
df['sell_success'] = np.where(df['sell_success']>0, 1, df['sell_success'])
df['buy_success'] = df['buy_success'].shift(-1)
df['sell_success'] = df['sell_success'].shift(-1)
df = df.loc[:,['closetime','sell_success','buy_success']]
test = pd.merge(df_1, df, on='closetime', how='inner')
test['sell_fail'] = np.where((test['buy_success']==1)&(test['sell_success']!=1),np.log((test['bid_price1_1s'])/(test['bid_price1'])),0)
test['buy_fail'] = np.where((test['sell_success']==1)&(test['buy_success']!=1),np.log((test['ask_price1_1s'])/(test['ask_price1'])),0)
test['fail'] = test['sell_fail']+test['buy_fail']
test['return'] = np.where(test['target']>0,np.log((test['ask_price1'])/(test['bid_price1'])), test['fail'])
test['profit'] = np.cumsum(test['return'])
test[['profit']].plot()
plt.show()
#%% 盈亏比策略
def ask_level_price(df):
    wap = (df['ask_price1']*df['ask_size2']+df['ask_price2']*df['ask_size1'])/(df['ask_size1']+df['ask_size2'])
    return wap
all_data['datetime'] = pd.to_datetime(all_data['closetime']+28800000, unit='ms')
all_data['wap'] = ask_level_price(all_data)
df = all_data.set_index('datetime').groupby(pd.Grouper(freq='5000ms')).apply('last')
df = pd.merge(df, price_max_min, on='closetime', how='inner')
df = df.sort_values(by='closetime', ascending=True)
df['sell_success'] = df['price_max']-(df['wap'].shift(1))
df['sell_success'] = np.where(df['sell_success']>0, 1, df['sell_success'])
df['sell_success'] = df['sell_success'].shift(-1)
df = df.loc[:,['closetime','sell_success']]
test = pd.merge(df_1, df, on='closetime', how='inner')
test['sell_fail'] = np.where(test['sell_success']!=1, np.log((test['bid_price1_1s'])/(test['ask_price1'])),0)
test['return'] = np.where(test['target']>0,np.log((test['wap'])/(test['ask_price1'])), test['sell_fail'])
test['profit'] = np.cumsum(test['return'])
test[['profit']].plot()
plt.show()

