from pyarrow import fs, Table
from enum import Enum
import pyarrow.parquet as pq
import gc
import pandas as pd
import time
import threading
#%%
"""
数据管理员
"""
MINIO_ENV = {
    "endpoint_override": "192.168.34.57:9000",
    "access_key": "zVGhI7gEzJtcY5ph",
    "secret_key": "9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx",
    "scheme": "http",
}
# 回测订单流
DATAPATH_BACKTEST_PATH = "datafile/bt_record/songhe"
DATAPATH_DCD_TRADE = "datafile/tick/trade/"
DATAPATH_DCD_EDPTH = "datafile/tick/order_book_100ms/"


class DataType(Enum):
    FT = 'FT',  # FT 期货数据
    DCK = 'DCK',  # DCK 数字货币k线
    DCD = 'DCD'  # DCD 数字货币深度,一般带有成交记录


class MinioDataPath(Enum):
    FT = "futures/main_eight"
    FTK = "futures/main_eight_kline",
    DCK_1m_bn_spot = "datafile/kline/1m/binance_spot",
    DCK_1m_bn_swapu = "datafile/kline/1m/binance_swap_u",
    DCK_1m_gt_spot = "datafile/kline/1m/gate_spot",
    DCK_1m_gt_swapu = "datafile/kline/1m/gate_swap_u",
    DCD = "datafile/tick/tick_1s/"


class Meta(object):

    def __init__(self):
        self.mfs = fs.S3FileSystem(
            endpoint_override=MINIO_ENV['endpoint_override'],
            access_key=MINIO_ENV['access_key'],
            secret_key=MINIO_ENV['secret_key'],
            scheme=MINIO_ENV['scheme']
        )

    def read_minio_data(self, root_path, symbol=None, btid=None, platform=None, start_time=None, end_time=None):
        """读取minio的数据为pd数据;暂时整体读,大数据读取需要从入口处做分片建议一次30天或50W条"""
        filters = []
        if symbol:
            filters.append(('symbol', '=', symbol))
        if btid:
            filters.append(('btid', '=', btid))
        # if platform:
        #     filters.append(('platform', '=', platform))

        if root_path == DATAPATH_DCD_TRADE:
            if start_time:
                filters.append(('timestamp', '>=', start_time))
            if end_time:
                filters.append(('timestamp', '<=', end_time))
            dataset = pq.ParquetDataset(root_path + platform, filters=filters,
                                        filesystem=self.mfs).read_pandas().to_pandas()
            dataset['closetime'] = dataset['timestamp']
            dataset = dataset.drop(['timestamp'], axis=1)
        elif root_path == DATAPATH_DCD_EDPTH:
            if start_time:
                filters.append(('closetime', '>=', start_time))
            if end_time:
                filters.append(('closetime', '<=', end_time))
            dataset = pq.ParquetDataset(root_path + platform, filters=filters,
                                        filesystem=self.mfs).read_pandas().to_pandas()
        else:
            if start_time:
                filters.append(('closetime', '>=', start_time))
            if end_time:
                filters.append(('closetime', '<=', end_time))
            filters.remove(('btid', '=', btid))
            dataset = pq.ParquetDataset('{}/btid={}'.format(root_path, btid), filters=filters,
                                        filesystem=self.mfs).read_pandas().to_pandas()
        return dataset

    def get_dcd_back_test_data(self, btid, start_time, end_time):
        # ---------------------------- 策略信号 ----------------------------------------
        filters = [('closetime', '>=', start_time), ('closetime', '<=', end_time)]
        userorder = pq.ParquetDataset('{}/btid={}'.format(DATAPATH_BACKTEST_PATH, btid), filters=filters,
                                      filesystem=self.mfs).read_pandas().to_pandas()

        symbol = userorder['symbol'].iloc[0]
        platform = 'gate_swap_u' if userorder['platform'].iloc[0] == 'binance_spot' else userorder['platform'].iloc[0]
        userorder.sort_values(by='closetime', ascending=True, inplace=True)
        userorder = userorder.rename(columns={"price": "s_price", "size": "s_size"})
        userorder = userorder.drop(['platform'], axis=1)

        # trades = self.read_minio_data(DATAPATH_DCD_TRADE, symbol=symbol, platform=platform, start_time=start_time,
        #                               end_time=end_time)
        trade_1 = pd.read_csv(
            '/run/media/ps/data/songhe/crypto/btcusdt/tick_100ms/order_flow/gate_spot_btcusdt_order_flow_2022-10-7-0_2022-10-12-3.csv')
        trade_2 = pd.read_csv(
            '/run/media/ps/data/songhe/crypto/btcusdt/tick_100ms/order_flow/gate_spot_btcusdt_order_flow_2022-10-12-0_2022-10-18-0.csv')
        trades = pd.concat([trade_1, trade_2], axis=0)
        trades = trades.rename({'timestamp': 'closetime'}, axis='columns')
        # trades['closetime'] = (trades['closetime'] / 100).astype(int) * 100 + 99
        trades.sort_values(by='dealid', ascending=True, inplace=True)
        trades = trades.drop(['symbol', 'year', 'month'], axis=1)

        # depth = self.read_minio_data(DATAPATH_DCD_EDPTH, symbol=symbol, platform=platform, start_time=start_time,
        #                              end_time=end_time)
        depth_1 = pd.read_csv(
            '/run/media/ps/data/songhe/crypto/btcusdt/tick_100ms/order_book/gate_spot_btcusdt_2022_10_8depth.csv')
        depth_2 = pd.read_csv(
            '/run/media/ps/data/songhe/crypto/btcusdt/tick_100ms/order_book/gate_spot_btcusdt_2022_10_9depth.csv')
        depth_3 = pd.read_csv(
            '/run/media/ps/data/songhe/crypto/btcusdt/tick_100ms/order_book/gate_spot_btcusdt_2022_10_10depth.csv')
        depth_4 = pd.read_csv(
            '/run/media/ps/data/songhe/crypto/btcusdt/tick_100ms/order_book/gate_spot_btcusdt_2022_10_11depth.csv')
        depth_5 = pd.read_csv(
            '/run/media/ps/data/songhe/crypto/btcusdt/tick_100ms/order_book/gate_spot_btcusdt_2022_10_12depth.csv')
        depth_6 = pd.read_csv(
            '/run/media/ps/data/songhe/crypto/btcusdt/tick_100ms/order_book/gate_spot_btcusdt_2022_10_13depth.csv')
        depth_7 = pd.read_csv(
            '/run/media/ps/data/songhe/crypto/btcusdt/tick_100ms/order_book/gate_spot_btcusdt_2022_10_14depth.csv')
        depth_8 = pd.read_csv(
            '/run/media/ps/data/songhe/crypto/btcusdt/tick_100ms/order_book/gate_spot_btcusdt_2022_10_15depth.csv')
        depth_9 = pd.read_csv(
            '/run/media/ps/data/songhe/crypto/btcusdt/tick_100ms/order_book/gate_spot_btcusdt_2022_10_16depth.csv')
        depth_10 = pd.read_csv(
            '/run/media/ps/data/songhe/crypto/btcusdt/tick_100ms/order_book/gate_spot_btcusdt_2022_10_17depth.csv')
        depth = pd.concat([depth_1, depth_2, depth_3, depth_4, depth_5, depth_6, depth_7, depth_8, depth_9, depth_10], axis=0)
        # del depth_1, depth_2, depth_3, depth_4, depth_5, depth_6, depth_7
        depth = depth.loc[:,
                ['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1', 'ask_price2', 'ask_size2',
                 'bid_price2',
                 'bid_size2', 'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4',
                 'bid_price4',
                 'bid_size4', 'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5']]

        # depth['datetime'] = pd.to_datetime(depth['closetime']+28800000, unit='ms')
        # depth = depth.sort_values(by='closetime', ascending=True)
        depth.reset_index(drop=False, inplace=True)  # 这个只是临时需要加
        depth.sort_values(by='closetime', ascending=True, inplace=True)

        # 盘口深度和订单流取并集
        trades_merge_depth = trades.merge(depth, how='outer', on='closetime')
        # trades_merge_depth = pd.merge(trades, depth, how='outer', on='closetime')
        trades_merge_depth.sort_values(by='closetime', ignore_index=True, ascending=True, inplace=True)
        trades_merge_depth = trades_merge_depth.merge(userorder, how='outer', on='closetime', )
        # colnums = ['ask_price1', 'ask_size1', 'bid_price1', 'bid_size1', 'ask_price2', 'ask_size2',
        #            'bid_price2', 'bid_size2', 'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3',
        #            'ask_price4', 'ask_size4', 'bid_price4', 'bid_size4', 'starttime', 'endtime']
        colnums = ['starttime', 'endtime']

        # 将数据做向下补齐
        for var in colnums:
            trades_merge_depth[var].fillna(method='ffill', inplace=True)
        back_test_data = trades_merge_depth.loc[trades_merge_depth['starttime'] - 2 <= trades_merge_depth['closetime']]
        back_test_data = back_test_data.loc[back_test_data['closetime'] <= back_test_data['endtime']]
        back_test_data.to_csv('/songhe/BTCUSDT/hft100ms_20221015_1017_5s_20221024_btcusdt_backtest_data.csv')
        print(back_test_data)
        pq.write_to_dataset(Table.from_pandas(df=back_test_data),
                            root_path='{}/btid={}_backtest_data'.format(DATAPATH_BACKTEST_PATH, btid),
                            filesystem=self.mfs,
                            basename_template="part-{i}.parquet",
                            existing_data_behavior="overwrite_or_ignore")

        del userorder
        del depth
        del trades
        del trades_merge_depth
        gc.collect()
        return back_test_data

    def run(self):
        # TODO --------------------------- 修改此处参数即可运行数据组装 --------------------------------
        btid = "hft100ms_20221015_1017_5s_20221024_btcusdt"
        start_time = '2022-10-14-0'  # 这个要比回测的开始时间至少提前1小时
        end_time = '2022-10-18-0'  # 这个要比回测的结束时间至少延后1小时

        if isinstance(start_time, str):
            start_time = int(time.mktime(time.strptime(start_time, "%Y-%m-%d-%H"))) * 1000
            end_time = int(time.mktime(time.strptime(end_time, "%Y-%m-%d-%H"))) * 1000
        t = time.time()
        print(t)
        res = self.get_dcd_back_test_data(btid=btid, start_time=start_time, end_time=end_time)
        # res.
        print('组装数据耗时', time.time() - t)


if __name__ == '__main__':
    Meta().run()
# Meta().run()