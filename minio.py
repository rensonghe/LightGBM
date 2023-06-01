import time
from pyarrow import fs
import pyarrow.parquet as pq
from pyarrow import Table
import pandas as pd

minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="opRIlgkBJxXxB1wo",
                        secret_key="MY1NWIZtlPbcYMBoyRDW4RrI73cM4J3d", scheme="http")


# 从minio中读数据
def get_data_from_minio(platform, symbol, dir_name, index_name='closetime', year=None, month=None, day=None,
                        start_time=None, end_time=None, multiplication_factor=1000):
    '''

    :param platform: str 传入平台
    :param symbol: str 'btcusdt'
    :param dir_name: str minio的目录名
    :param index_name: 下标名 一般为 'closetime' 币安现货Kline 为 'opentime'
    :param year: int 2022
    :param month: int 8
    :param day: int 8
    :param start_time: '2022-06-20-0'
    :param end_time:'2022-06-29-0'
    :param multiplication_factor: 用于兼容13位时间戳  币安的时间戳是13位的 multiplication_factor=1000 否则为1
    :return:
    '''
    # 注意要看minio中的数据来做过滤
    filters = []
    if symbol:
        filters.append(("symbol", "=", symbol))
    if year:
        filters.append(('year', '=', year))
    if month:
        filters.append(('month', '=', month))
    if day:
        filters.append(('day', '=', day))
    if start_time:
        if isinstance(start_time, str):
            start_time = int(time.mktime(time.strptime(start_time, "%Y-%m-%d-%H"))) * multiplication_factor
        filters.append((index_name, '>=', start_time))
    if end_time:
        if isinstance(end_time, str):
            end_time = int(time.mktime(time.strptime(end_time, "%Y-%m-%d-%H"))) * multiplication_factor
        filters.append((index_name, '<=', end_time))
    if filters:
        dataset = pq.ParquetDataset('{}'.format(dir_name), filters=filters, filesystem=minio)
    else:
        dataset = pq.ParquetDataset('{}'.format(dir_name), filesystem=minio)

    dataset = dataset.read_pandas().to_pandas()

    if multiplication_factor != 1000:
        dataset[index_name] *= 1000
        dataset[index_name] = dataset[index_name].astype('int64')
    return dataset


# 写数据
# pq.write_to_dataset(Table.from_pandas(agg_data), 'datafile/tick/trade/gate_swap_u',
#                     partition_cols=['symbol', 'year', 'month']
#                     , filesystem=minio, basename_template="part-{i}.parquet",
#                     existing_data_behavior="overwrite_or_ignore")

if __name__ == '__main__':
    pass
    # data = get_data_from_minio('gate_swap_u', 'btcusdt', 2022, 8, 'datafile/tick/tick_1s/gate_swap_u',
    #                            start_time='2022-8-31-0', end_time='2022-8-31-12')
    # print(data['dealid'].values[-1] - data['dealid'].values[0] + 1)
    # data = get_data_from_minio('gate_swap_u', 'btcusdt', 2022, 8, 'datafile/tick/tick_1s/gate_swap_u')
    # print(data['closetime'].values[-1] - data['closetime'].values[0])

    # kline = get_data_from_minio('binance_spot', 'btcusdt', 'datafile/kline/1m/binance_spot',
    #                             index_name='opentime', start_time='2022-7-30-0', end_time='2022-7-31-12',
    #                             multiplication_factor=1000)
    # print(kline)

    # feat_kline = get_data_from_minio('binance_spot', 'btcusdt', 'datafile/feat/kline_1m/binance_spot',
    #                                  start_time='2022-7-30-0', end_time='2022-7-31-12')
    # print(feat_kline)
    # trade_info = get_data_from_minio('gate_spot', 'btcusdt', 'datafile/tick/order_book_100ms/gate_spot',
    #                                  start_time='2022-7-1-0', end_time='2022-7-29-12')
    # trade_info.sort_values(by='begin_id', ascending=True, inplace=True)
    # test = trade_info[trade_info['begin_id'] - trade_info['begin_id'].shift(1) - trade_info['merged_count'] != 0]
    # print(test)
    trade_info = get_data_from_minio(platform=None, symbol=None,
                                     dir_name='datafile/bt_record/btid=kline1m_long_1011_ethusdt')
    print(
        trade_info
    )
    trade_info.sort_values(by='begin_id', ascending=True, inplace=True)
    test = trade_info[trade_info['begin_id'] - trade_info['begin_id'].shift(1) - trade_info['merged_count'] != 0]
