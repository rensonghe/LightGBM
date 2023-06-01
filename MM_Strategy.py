#%%
import pandas as pd
from websocket import create_connection
import json
import time
import threading
import pyarrow.parquet as pq
from pyarrow import fs
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
DATAPATH_BACKTEST_PATH = "datafile/bt_record/songhe"
btid = "hft100ms_20221015_1017_5s_20221024_btcusdt_backtest_data"

class Execute(object):
    def __init__(self):
        # 从csv中获取信号执行需要的数据 也可以从minio中获取 运行getDCDBTData.py 可以获取该数据
        # self.bt_data = pd.read_csv('datafile_bt_record_songhe_btid=hft100ms_20221020_btcusdt_backtest_data.csv')
        self.bt_data = pq.ParquetDataset('{}/btid={}'.format(DATAPATH_BACKTEST_PATH, btid),
                                      filesystem=minio).read_pandas().to_pandas()
        self.bt_data.sort_values(by='closetime', ascending=True, inplace=True)
        self.symbol = self.bt_data['symbol'].iloc[0]
        self.column_list = self.bt_data.columns.to_list()  # 回测组装数据的表头列表

        self.depth = {'a': [], 'b': [], 'closetime': 1234567890123, 'old_data': []}  # 当前的深度信息
        self.ws = create_connection("ws://192.168.34.9:9001/", timeout=3000)
        self.initial_cash = 1000  # initial cash 初始本金
        self.open_orders = []  # 当前正在挂单的列表
        self.position = {'price': 0, 'amount': 0, 'fee': 0}  # 持仓均价、数量和本次手续费
        self.cash = 0  # 当前本金
        self.last_place_order_time = 0  # 记录上一次下单时间

        self.place_buy_order = 0
        self.place_sell_order = 0

    def _place_order(self, create_timestamp, symbol, price, side, size, order_type='gtc', status='open'):
        place_order = {
            "create_timestamp": create_timestamp,
            "channel": "place_order",
            "symbol": symbol,
            "price": price,
            "side": side,
            "size": size,  # 为正
            "order_type": order_type,
            "o_type": 0,
            "auto_cancle_interval": 0,
            "otype_round": 0,
            "auto_cancel": False,
            "auto_interval": 0,
            "direction": None,
            "status": status,
        }
        print('place_order', place_order, create_timestamp)
        # self.queue.put(place_order)
        self.ws.send(json.dumps(place_order))
        self.ws_receive()

    def _cancel_order(self, cancel_timestamp, symbol, cancel_type="all"):
        cancel_order = {
            "cancel_timestamp": cancel_timestamp,
            "channel": "cancel_order",
            "symbol": symbol,
            "cancel_type": cancel_type,  # 为all则撤销所有订单 为single则order_list会生效
            "order_list": []  # 这个列表里面有的订单都会撤销
        }

        # self.queue.put(place_order)
        self.ws.send(json.dumps(cancel_order))
        self.ws_receive()

    # 根据数据做订单执行
    def execute(self):
        test = {
            "channel": "init",
            "symbol": self.symbol,
            "principal": self.initial_cash,
            "auto_cancel_interval": 0,
            "open_interval": 0,
            "otype": "normal",
            "close_interval": 0,
            "maker_fee": 0,
            "taker_fee": 0,
            "column_list": self.column_list,
        }
        self.ws.send(json.dumps(test))
        print(self.ws.recv())
        # print("test channel Received '%s'" % result)
        sum_num = 0
        for i in self.bt_data.values.tolist():
            # print(i)
            sum_num += 1
            # if sum_num > 100000:
            #     break
            closetime = i[self.column_list.index('closetime')]
            # 此时有成交
            if i[self.column_list.index('dealid')] > 0:
                # 当条成交记录的时间戳
                is_depth_flag = False
                # 此时5s内有深度
                if closetime - self.depth['closetime'] < 5000 and self.depth['a']:
                    for i_ in range(1, 6):
                        i[self.column_list.index('ask_price{}'.format(i_))] = self.depth['a'][i_ - 1][0]
                        i[self.column_list.index('ask_size{}'.format(i_))] = self.depth['a'][i_ - 1][1]
                        i[self.column_list.index('bid_price{}'.format(i_))] = self.depth['b'][i_ - 1][0]
                        i[self.column_list.index('bid_size{}'.format(i_))] = self.depth['b'][i_ - 1][1]
                        is_depth_flag = True
                order_flow = {
                    "closetime": closetime,
                    "channel": "order_flow",
                    'is_depth': is_depth_flag,  # 为True则本次有深度数据 为False 则本次没有深度
                    "result": {
                        "dealid": i[self.column_list.index('dealid')],
                        "price": i[self.column_list.index('price')],
                        "size": i[self.column_list.index('size')],
                    },
                    "row_data": i,
                }
                if self.open_orders:
                    print('order_flow:', order_flow)
                    self.ws.send(json.dumps(order_flow))
                    self.ws_receive()
                # print("order_flow channel Received '%s'" % result)
                # self.queue.put(order_flow)
            # 此时有深度
            if i[self.column_list.index('ask_price1')] > 0:
                ask_info = [[i[self.column_list.index('ask_price{}'.format(i_))],
                             i[self.column_list.index('ask_size{}'.format(i_))]] for i_ in range(1, 6)]
                bid_info = [[i[self.column_list.index('bid_price{}'.format(i_))],
                             i[self.column_list.index('bid_size{}'.format(i_))]] for i_ in range(1, 6)]
                # bid_info = [['bid_price{}'.format(i_), 'bid_size{}'.format(i_)] for i_ in range(1, 6)]

                order_book = {
                    "closetime": closetime,
                    "channel": "order_book",
                    "result": {
                        "a": ask_info,
                        "b": bid_info,
                    },
                    "row_data": i,
                }
                self.depth.update({'a': order_book['result']['a'], 'b': order_book['result']['b'],
                                   'closetime': order_book['closetime'], 'old_data': i})
                self.ws.send(json.dumps(order_book))
                position = self.position['amount']  # 大于0持有多仓 小于0持有空仓
                # 5s内固定价格平仓
                if closetime - self.last_place_order_time < 5000:
                    if self.open_orders:
                        if position > 0:
                            # print('已有限价多单，撤销所有挂单多单')
                            self._cancel_order(cancel_timestamp=closetime, symbol=self.symbol, cancel_type='all')
                            print('已有多单持仓，下空单平仓', closetime)
                            self._place_order(create_timestamp=closetime, symbol=self.symbol, price=self.place_sell_order,
                                              side='sell',size=position, order_type='gtc', status='close')
                        if position < 0:
                            # print('已有限价空单，撤销所有挂单空单')
                            self._cancel_order(cancel_timestamp=closetime, symbol=self.symbol, cancel_type='all')
                            print('已有空单持仓，下多单平仓', closetime)
                            self._place_order(create_timestamp=closetime, symbol=self.symbol, price=self.place_buy_order,
                                              side='buy',size=-position, order_type='gtc', status='close')
                # 超过5s平仓
                if closetime - self.last_place_order_time >= 5000:
                    # 撤销所有订单 下平仓单之前先做撤单
                    if self.open_orders:
                        print('撤销所有订单')
                        self._cancel_order(cancel_timestamp=closetime, symbol=self.symbol, cancel_type='all')
                    if position > 0:  # 持有多仓
                        print('平多仓')
                        bid_price1 = self.depth['b'][0][0]
                        self._place_order(create_timestamp=closetime, symbol=self.symbol, price=bid_price1,
                                          side='sell', size=position, order_type='gtc', status='close')
                    elif position < 0:
                        print('平空仓')
                        ask_price1 = self.depth['a'][0][0]
                        self._place_order(create_timestamp=closetime, symbol=self.symbol, price=ask_price1,
                                          side='buy', size=-position, order_type='gtc', status='close')
                        # print("order_book channel Received '%s'" % result)
                # self.queue.put(order_book)
            # 此时有信号
            if i[self.column_list.index('predict')] > 0:
                # side = i[self.column_list.index('side')]
                # print(s_price, side, s_size, starttime, endtime, symbol)
                # position = self.position['amount']  # 大于0持有多仓 小于0持有空仓
                ask_price1 = self.depth['a'][0][0]
                # print('ask_price1--------------',ask_price1)
                bid_price1 = self.depth['b'][0][0]
                # print('bid_price1--------------',bid_price1)
                buy_size = self.initial_cash / ask_price1
                # buy_size = 0.01
                sell_size = buy_size
                self.last_place_order_time = closetime
                print('下双边订单------------------------------------------------', closetime)
                # if closetime == 1665593754999:
                #     print(self.depth)
                self._place_order(create_timestamp=closetime, symbol=self.symbol, price=bid_price1, side='buy',
                                  size=buy_size, order_type='gtc', status='open')
                self.place_buy_order = bid_price1
                self._place_order(create_timestamp=closetime, symbol=self.symbol, price=ask_price1, side='sell',
                                  size=sell_size, order_type='gtc', status='open')
                self.place_sell_order = ask_price1
                # print(self.depth)


    # 接受server返回过来的信息
    def ws_receive(self):
        res = json.loads(self.ws.recv())
        if res['channel'] == 'open_orders':
            self.open_orders = res['result']  #
            print('当前挂单信息', res['closetime'], self.open_orders)
            self.position = res['position']
            self.cash = res['account']
            print('当前持仓:', self.position, '实时余额', self.cash)
            #
            if res['user_trade']:
                print('用户成交信息:', res['user_trade'])
        else:
            print('接收到其他信息', res)


if __name__ == '__main__':
    Execute().execute()
