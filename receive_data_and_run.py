# %%
import asyncio
import pandas as pd
import numpy as np
import hmac
import hashlib
import json
import time
import threading
import joblib
import websockets as ws
from websockets import ConnectionClosed
from HFT_factor import add_factor_process

# import redis
# pool = redis.ConnectionPool(host='192.168.34.57',port=6379, db=2, decode_responses=True)
# r=redis.Redis(connection_pool=pool)
symbol_list = ['btcusdt']

model_side_0 = joblib.load('10bar_vwap_dogeusdt_lightGBM_20230207_side_0.pkl')
model_side_1 = joblib.load('10bar_vwap_dogeusdt_lightGBM_20230207_side_1.pkl')
model_side_2 = joblib.load('10bar_vwap_dogeusdt_lightGBM_20230207_side_2.pkl')
model_side_3 = joblib.load('10bar_vwap_dogeusdt_lightGBM_20230207_side_3.pkl')
model_side_4 = joblib.load('10bar_vwap_dogeusdt_lightGBM_20230207_side_4.pkl')

model_out_0 = joblib.load('10bar_vwap_dogeusdt_lightGBM_20230207_out_0.pkl')
model_out_1 = joblib.load('10bar_vwap_dogeusdt_lightGBM_20230207_out_1.pkl')
model_out_2 = joblib.load('10bar_vwap_dogeusdt_lightGBM_20230207_out_2.pkl')
model_out_3 = joblib.load('10bar_vwap_dogeusdt_lightGBM_20230207_out_3.pkl')
model_out_4 = joblib.load('10bar_vwap_dogeusdt_lightGBM_20230207_out_4.pkl')

# short_signal = 0.4413212033008063
# long_signal = 0.5889137347246168



class OnlineByKline(object):
    def __init__(self, symbol):
        self.ws = None
        self.symbol = symbol
        # 1m k线信息的存放处 仅用于初始化使用
        self.kline_all = []
        self.depth = []
        self.trade = []
        self.last_time = int(time.time())
        self.open_orders = []
        self.position = {'price': 0, 'amount': 0, 'fee': 0}  # 持仓均价、数量和本次手续费
        self.last_place_order_time = 0  # 记录上一次下单时间
        self.initial_cash = 30
        self.y_pred_list = []
        self.cancel_flag = True
        self.auto_cancel_interval = 0
        self.filled_last_order_time = time.time()*1000
        self.user_trade = []
        self.kill_time = 0

    def _place_order(self, create_timestamp, symbol, price, side, size, order_type='gtc', status='open',
                     auto_cancel_interval=0):
        place_order = {
            "create_timestamp": create_timestamp,
            "channel": "place_order",
            "symbol": symbol,
            "price": price,
            "side": side,
            "size": size,  # 为正
            "order_type": order_type,
            "o_type": 0,
            "auto_cancel_interval": auto_cancel_interval,
            "otype_round": 0,
            "auto_cancel": False,
            "auto_interval": 0,
            "direction": None,
            "status": status,
        }
        print('place_order', place_order, create_timestamp)
        if auto_cancel_interval > 0:
            self.cancel_flag = True
            self.auto_cancel_interval = auto_cancel_interval
        return place_order

    def _cancel_order(self, cancel_timestamp, symbol, cancel_type="all"):
        cancel_order = {
            "cancel_timestamp": cancel_timestamp,
            "channel": "cancel_order",
            "symbol": symbol,
            "cancel_type": cancel_type,  # 为all则撤销所有订单 为single则order_list会生效
            "order_list": []  # 这个列表里面有的订单都会撤销
        }
        return cancel_order

    async def step_one(self, ):
        url = "ws://192.168.34.48:10003"
        old_mim = 0
        while True:
            try:
                async with ws.connect(url) as websocket:

                    init_info = {
                        "channel": "init",
                        # 订阅的列表 形式: ['btcusdt_kline_1m', 'btcusdt_trade', 'btcusdt_depth']
                        "subscribe_list": ['btcusdt_trade', 'btcusdt_depth', 'btcusdt_private'],
                    }
                    time_info = time.localtime(time.time())
                    sign = hmac.new("{}".format(time_info.tm_year * time_info.tm_mon * time_info.tm_hour).encode(),
                                    json.dumps(init_info).encode(),
                                    digestmod=hashlib.sha256).hexdigest()
                    init_info['sign'] = sign
                    await websocket.send(json.dumps(init_info))

                    # asyncio.create_task(ping(websocket))
                    while True:
                        try:
                            res_json = await websocket.recv()
                            res = json.loads(res_json)
                            if 'depth' in res.get('channel'):
                                closetime = res['result']['closetime'] // 100 * 100 + 99
                                depth_dict = {'closetime': closetime,
                                              'ask_price1': res['result']['a'][0][0],
                                              'ask_size1': res['result']['a'][0][1],
                                              'bid_price1': res['result']['b'][0][0],
                                              'bid_size1': res['result']['b'][0][1],
                                              'ask_price2': res['result']['a'][1][0],
                                              'ask_size2': res['result']['a'][1][1],
                                              'bid_price2': res['result']['b'][1][0],
                                              'bid_size2': res['result']['b'][1][1],
                                              'ask_price3': res['result']['a'][2][0],
                                              'ask_size3': res['result']['a'][2][1],
                                              'bid_price3': res['result']['b'][2][0],
                                              'bid_size3': res['result']['b'][2][1],
                                              'ask_price4': res['result']['a'][3][0],
                                              'ask_size4': res['result']['a'][3][1],
                                              'bid_price4': res['result']['b'][3][0],
                                              'bid_size4': res['result']['b'][3][1],
                                              'ask_price5': res['result']['a'][4][0],
                                              'ask_size5': res['result']['a'][4][1],
                                              'bid_price5': res['result']['b'][4][0],
                                              'bid_size5': res['result']['b'][4][1]
                                              }

                                self.depth.append(depth_dict)
                                # time_10 = int(closetime / 1000)
                                # len_depth = int(len(self.depth) * 0.99)
                                # diff_time = self.depth[-1]['closetime'] - self.depth[-len_depth]['closetime']
                                position = self.position['amount']
                                # timestamp = self.position['timestamp']
                                # diff = closetime - self.filled_last_order_time
                                # if position != 0 and diff >= 60000 * 60 * 3:
                                    # print('实时仓位:', position, closetime)

                                if len(self.trade) > 0:
                                    for i in self.trade:
                                        current_price = i['price']
                                else:
                                    current_price = None

                                if current_price is not None and int(closetime/1000) - int(self.kill_time/1000) >= 60:

                                    # 多头止盈止损
                                    if position > 0:
                                        # print('-------------平仓之前撤销所有订单-------------')
                                        cancel_order = self._cancel_order(cancel_timestamp=closetime,
                                                                          symbol=self.symbol, cancel_type='all')
                                        await websocket.send(json.dumps(cancel_order))
                                        pf = float(current_price) / float(self.position['price']) - 1
                                        diff = closetime - self.filled_last_order_time
                                        print('holding hour:',diff/60000/60)
                                        # print(diff,'----------')
                                        bp1 = depth_dict['bid_price1']
                                        con1 = 0
                                        # if diff >= 60000 * 60 * 2:
                                        #     con1 = 1
                                        #     print('-------------多头时间离场-------------')
                                        if pf > 0.05:
                                            con1 = 1
                                            print('-------------多头止盈离场-------------')
                                        elif pf <= -0.003:
                                            con1 = 1
                                            print('-------------多头止损离场-------------')
                                        if con1 == 1:
                                            print('-------------离场时间-----------------',
                                                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(closetime / 1000)))
                                            self.kill_time = closetime
                                            place_order = self._place_order(create_timestamp=closetime,
                                                                            symbol=self.symbol,
                                                                            price=bp1 * 0.9997, side='sell',
                                                                            size=position, order_type='gtc',
                                                                            status='close')
                                            await websocket.send(json.dumps(place_order))

                                    # 空头止盈止损
                                    if position < 0:
                                        # print('-------------平仓之前撤销所有订单-------------')
                                        self._cancel_order(cancel_timestamp=closetime, symbol=self.symbol,
                                                           cancel_type='all')
                                        pf = 1 - float(current_price) / float(self.position['price'])
                                        diff = closetime - self.filled_last_order_time
                                        print('holding hour:', diff / 60000 / 60)
                                        ap1 = depth_dict['ask_price1']
                                        con1 = 0
                                        # if diff >= 60000 * 60 * 2:
                                        #     con1 = 1
                                        #     print('-------------空头时间离场-------------')
                                        if pf > 0.05:
                                            con1 = 1
                                            print('-------------空头止盈离场-------------')
                                        elif pf <= -0.003:
                                            con1 = 1
                                            print('-------------空头止损离场-------------')
                                        if con1 == 1:
                                            print('-------------离场时间-----------------',
                                                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(closetime / 1000)))
                                            self.kill_time = closetime
                                            place_order = self._place_order(create_timestamp=closetime,
                                                                            symbol=self.symbol,
                                                                            price=ap1 * 1.0003, side='buy',
                                                                            size=-position, order_type='gtc',
                                                                            status='close')
                                            await websocket.send(json.dumps(place_order))

                            elif 'user_trades' in res.get('channel'):
                                """{'timestamp': 1670905391798, 
                                    'channel': 'ethusdt_user_trades',
                                    'result': {'price': 1272.15, 'amount': 0.06, 'id': 182627073,
                                                    'fee': 0.0152658, 'role': 'taker'},
                                    'symbol': 'ethusdt'}

                                """
                                print('user_trades', res)
                                # if res['user_trades']:
                                self.filled_last_order_time = res['timestamp']

                            elif 'trade' in res.get('channel'):
                                trade_dict = {'closetime': res['result']['timestamp'], 'price': res['result']['price'],
                                              'size': res['result']['size'], 'cum_size': res['result']['size_sum'],
                                              'turnover': res['result']['volume_sum']}
                                self.trade.append(trade_dict)
                                # print(self.trade)

                            elif 'position' in res.get('channel'):
                                diff = res['timestamp'] - self.filled_last_order_time
                                if diff >= 60000 * 60 * 2:
                                    print('实时仓位:', position, res['timestamp'], diff)
                                # print('position', res)
                                self.position.update(res['result'])
                            if res.get("timestamp"):
                                time_10 = int(res['timestamp'] / 1000)
                                interval_time = 500000
                                # print('diff_time:', diff_time)
                                if self.depth[-1]['closetime'] - self.depth[0]['closetime'] > interval_time\
                                        and time_10 - self.last_time > 0.999:
                                    self.last_time = time_10
                                    len_depth = int(len(self.depth) * 0.99)
                                    diff_time = self.depth[-1]['closetime'] - self.depth[-len_depth]['closetime']
                                    if diff_time > interval_time:
                                        self.depth = self.depth[-len_depth:]
                                    # self.depth = self.depth[-500:]
                                    len_trade = int(len(self.trade) * 0.99)
                                    if self.trade[-1]['closetime'] - self.trade[-len_trade]['closetime'] > interval_time:
                                        self.trade = self.trade[-len_trade:]
                                        # self.trade = self.trade[-500:]
                                    df_depth = pd.DataFrame(self.depth)
                                    df_trade = pd.DataFrame(self.trade)
                                    data_merge = pd.merge(df_depth, df_trade, on='closetime', how='outer')
                                    data_merge = data_merge.sort_values(by='closetime', ascending=True)
                                    data_merge = data_merge.drop_duplicates(subset=['closetime'], keep='last')
                                    data_merge['datetime'] = pd.to_datetime(data_merge['closetime'] + 28800000, unit='ms')
                                    # print(data_merge.iloc[-1:,:])
                                    tick = data_merge.set_index('datetime').groupby(pd.Grouper(freq='1000ms')).apply('last')
                                    tick = tick.reset_index()
                                    tick['min'] = tick['datetime'].dt.minute
                                    # tick['vwap'] = (tick['price'].fillna(0) * abs(tick['size'].fillna(0))).rolling(120).sum() / abs(tick['size'].fillna(0)).rolling(120).sum()

                                    closetime_min = time.localtime(closetime / 1000).tm_min
                                    # print(tick.iloc[-1/:, :]
                                    #       , '-------------------------------')

                                    if closetime_min != old_mim:
                                        # if tick['min'].iloc[-1] != tick['min'].iloc[-2]:
                                        old_mim = closetime_min
                                        # tick1 = tick.iloc[:-1, :]
                                        tick = tick.set_index('datetime')
                                        trade = tick.loc[:,
                                                ['closetime', 'price', 'size', 'cum_size', 'turnover']]
                                        depth = tick.loc[:,
                                                ['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1',
                                                 'ask_price2', 'ask_size2', 'bid_price2', 'bid_size2', 'ask_price3',
                                                 'ask_size3', 'bid_price3', 'bid_size3',
                                                 'ask_price4', 'ask_size4', 'bid_price4', 'bid_size4', 'ask_price5',
                                                 'ask_size5', 'bid_price5', 'bid_size5']]
                                        factor = add_factor_process(depth=depth, trade=trade)
                                        factor['datetime'] = pd.to_datetime(factor['closetime'] + 28800000, unit='ms')
                                        factor['vwap'] = (factor['price'].fillna(0) * abs(factor['size'].fillna(0))).rolling(120).sum() / abs(factor['size'].fillna(0)).rolling(120).sum()
                                        signal = factor.set_index('datetime').groupby(pd.Grouper(freq='1min')).apply('last')
                                        signal = signal.dropna(axis=0)
                                        signal_filter = signal.iloc[ :, 25:]
                                        final_signal = signal_filter.iloc[-2]
                                        X_test = np.array(final_signal).reshape(1, -1)

                                        y_pred_0 = model_0.predict(X_test, num_iteration=model_0.best_iteration)
                                        y_pred_1 = model_1.predict(X_test, num_iteration=model_1.best_iteration)
                                        y_pred_2 = model_2.predict(X_test, num_iteration=model_2.best_iteration)
                                        y_pred_3 = model_3.predict(X_test, num_iteration=model_3.best_iteration)
                                        y_pred_4 = model_4.predict(X_test, num_iteration=model_4.best_iteration)
                                        y_pred = (y_pred_0[0] + y_pred_1[0] + y_pred_2[0] + y_pred_3[0] + y_pred_4[0])/5
                                        # y_pred = model.predict(X_test, num_iteration=model.best_iteration)
                                        print('信号:', y_pred,
                                              '现在时间:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                                        self.y_pred_list.append(y_pred)
                                        price = final_signal['vwap']
                                        buy_size = self.initial_cash / df_depth['ask_price1'].iloc[-1]
                                        sell_size = self.initial_cash / df_depth['bid_price1'].iloc[-1]
                                        self.last_place_order_time = closetime
                                        # othertime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(closetime / 1000))
                                        # self.signal_df.append([othertime, y_pred[0]])


                                        if self.y_pred_list[-1] < short_signal and position > 0:
                                            print('-------------平仓之前撤销所有订单-------------')
                                            cancel_order = self._cancel_order(cancel_timestamp=closetime,
                                                                              symbol=self.symbol, cancel_type='all')
                                            await websocket.send(json.dumps(cancel_order))
                                            bid_price1 = df_depth['bid_price1'].iloc[-1]
                                            print(
                                                '------------------------------------------下空单平多仓--------------------------------------',
                                                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(closetime / 1000)))
                                            place_order = self._place_order(create_timestamp=closetime,
                                                                            symbol=self.symbol,
                                                                            price=bid_price1 * 0.9999, side='sell',
                                                                            size=position, order_type='gtc',
                                                                            status='close')
                                            await websocket.send(json.dumps(place_order))

                                        if self.y_pred_list[-1] > long_signal and position < 0:
                                            print('-------------平仓之前撤销所有订单-------------')
                                            cancel_order = self._cancel_order(cancel_timestamp=closetime,
                                                                              symbol=self.symbol, cancel_type='all')
                                            await websocket.send(json.dumps(cancel_order))
                                            ask_price1 = df_depth['ask_price1'].iloc[-1]
                                            print(
                                                '------------------------------------------下多单平空仓--------------------------------------',
                                                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(closetime / 1000)))
                                            place_order = self._place_order(create_timestamp=closetime,
                                                                            symbol=self.symbol,
                                                                            price=ask_price1 * 1.0001, side='buy',
                                                                            size=-position, order_type='gtc',
                                                                            status='close')
                                            await websocket.send(json.dumps(place_order))

                                        # sell
                                        if self.y_pred_list[-1] < short_signal and position * price > -0.8 * self.initial_cash:  # and position == 0
                                            print(
                                                '------------------------------------------下空单----------------------------------------------',
                                                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(closetime / 1000)))
                                            place_order = self._place_order(create_timestamp=closetime,
                                                                            symbol=self.symbol, price=price * 0.9998,
                                                                            side='sell',
                                                                            size=sell_size, order_type='gtc',
                                                                            status='open',
                                                                            auto_cancel_interval=60000 * 5)
                                            await websocket.send(json.dumps(place_order))

                                        # buy
                                        if self.y_pred_list[-1] > long_signal and position * price < 0.8 * self.initial_cash:  # and position == 0:
                                            print(
                                                '------------------------------------------下多单----------------------------------------------',
                                                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(closetime / 1000)))
                                            place_order = self._place_order(create_timestamp=closetime,
                                                                            symbol=self.symbol, price=price * 1.0002,
                                                                            side='buy',
                                                                            size=buy_size, order_type='gtc',
                                                                            status='open',
                                                                            auto_cancel_interval=60000 * 5)
                                            await websocket.send(json.dumps(place_order))
                                # signal_df = pd.DataFrame(self.signal_df, columns=['datetime', 'y_pred'])
                                # signal_df.to_csv('/songhe/ETHUSDT/ethusdt_online_signal_20221216.csv'

                        except Exception as e:
                            print('接收数据 出错:%s-----错误所在行数:%s' % (e, e.__traceback__.tb_lineno), res)
                            break
            except Exception as e:
                print('ws重连 出错:%s-----错误所在行数:%s' % (e, e.__traceback__.tb_lineno))
                await asyncio.sleep(2)

    def run_(self):
        threading.Thread(target=asyncio.get_event_loop().run_until_complete, args=(self.step_one(),)).start()


if __name__ == '__main__':
    OnlineByKline(symbol='btcusdt').run_()
