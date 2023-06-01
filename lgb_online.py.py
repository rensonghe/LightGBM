import os
import sys
import datetime
import time
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from pandas.core.frame import DataFrame
import pymysql
import math
import threading
import pandas as pd
import numpy as np
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
sys.path.append(rootPath)

import ccxt
import decimal
from common.exchange.ExchangeApi import ExchangeApi
from common.tools.Log import logInf,logErr
from common.tools import Ding

from redis import Redis as _Redis
import json
from pickle import dumps, loads, HIGHEST_PROTOCOL, UnpicklingError

logInf.info(ccxt.__version__)  # 检查ccxt版本，需要最新版本，1.44.21以上

class RedisMQ(_Redis):
    def set(self, name, value, ex=None, px=None, nx=False, xx=False):
        '''添加了 pickle 处理的 set'''
        pickled = dumps(value, HIGHEST_PROTOCOL)
        return super().set(name, pickled, ex, px, nx, xx)

    def get(self, name):
        '''添加了反序列化的 get'''
        pickled = super().get(name)
        try:
            return loads(pickled)
        except (UnpicklingError, TypeError):
            return pickled

    def lpush(self, name, *values):
        if self.llen(name) >= 10:
            for i in range(self.llen(name) - 9):
                self.rpop(name)
        return super().lpush(name, *values)

    def rpop(self, name):
        return super().rpop(name)

    def llen(self, name) -> int:
        return super().llen(name)

    def mget(self, keys, *args):
        pickled_list = super().mget(keys, *args)
        data = []
        for p in pickled_list:
            try:
                data.append(loads(p))
            except (UnpicklingError, TypeError):
                data.append(p)
        return data

class BADB(object):
    def __init__(self, **badb):
        self.username = badb['username']
        self.password = badb['password']
        self.database = badb['database']
        self.host = badb['host']
        self.conn = pymysql.connect(db=self.database, user=self.username, password=self.password, host=self.host,port=3306, autocommit=True)

    def get_symbols(self):
        sql = '''
            select symbol from binance_symbols_filter
        '''
        cursor = self.conn.cursor()
        cursor.execute(sql)
        df = cursor.fetchall()
        df = DataFrame(df, columns=['symbol'])
        cursor.close()
        return df

    def get_symbol_list_data(self, symbol_list):
        columns = ['event_time', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'end_time', 'amount', 'trades',
                   'taker_volume', 'taker_amount', 'symbol']

        if len(symbol_list) == 0:
            return None
        elif len(symbol_list) == 1:
            symbol_list = symbol_list.append('BTC')

        sql = '''
                select {0} from exchange.perpetual_kline_1m
                where symbol in {1}
                order by id desc
                limit {2}
            '''.format(','.join(columns), tuple(symbol_list), len(symbol_list) * 1450)
        cur = self.conn.cursor()
        cur.execute(sql)
        df = cur.fetchall()
        df = DataFrame(df, columns=columns)
        cur.close()
        return df

class lgb_strategy():

    def __init__(self, **param):
        self.db = BADB(**param['badb'])
        self.exchange_name = param['exchange_name']
        self.market_type = param['market_type']
        self.symbols = param['symbols']
        self.every_money = param['every_money']
        self.pos_threshold = param['pos_threshold']
        self.neg_threshold = param['neg_threshold']

        # self.exchange_name = 'binance'
        # self.market_type = 'umfuture'
        # self.symbols = self.db.get_symbols()['symbol'].to_list()[0:20]
        # self.every_money = 8
        # self.pos_threshold = 0.75
        # self.neg_threshold = 0.75

        self.exchangeApi = ExchangeApi()
        self.rds = RedisMQ(**param['redis'])

        self.gbm_pos = lgb.Booster(model_file=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pos_lgb_version4.txt'))
        self.gbm_neg = lgb.Booster(model_file=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'neg_lgb_version4.txt'))

        self.trade_id_list = []
        self.del_trade_id_list = []
        self.deal_trade_id_list = []

        self.columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'trades', 'taker_volume', 'taker_amount']
        self.column_list = ['open', 'high', 'low', 'close', 'volume', 'amount', 'trades', 'taker_volume', 'taker_amount',
                       'day_amount','day_volume', 'day_avg_price', 'amp', 'max_oc', 'min_oc', 'up_amp', 'down_amp', 'amp_ma',
                       'up_amp_ma','down_amp_ma', 'delta_ma', 'avg_amount', 'take_pct', 'delta_take_pct', 'take_pct_ma', 'pct1',
                       'skew','kurt','up_vol', 'up_vol_ma', 'skew_ma', 'kurt_ma', 'delta_up_vol', 'delta_skew', 'delta_kurt', 'pct3',
                       'pct5','pct10','last_pct1', 'last_pct3', 'last_pct5', 'last_pct10', 'ma_pct1', 'ma_pct3', 'ma_pct5', 'ma_pct10',
                       'std_pct1','std_pct3', 'std_pct5', 'std_pct10', 'ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'bias', 'bias1',
                       'ma_bbi','ddd', 'dma','pos_price', 'pos_value', 'pos_delta_time', 'neg_price', 'neg_value', 'neg_delta_time',
                       'pos_value1','neg_value1','pos_rate', 'neg_rate', 'last_pos_price', 'last_pos_value', 'last_pos_value1',
                       'last_pos_delta_time','last_pos_rate', 'last_neg_price', 'last_neg_value', 'last_neg_value1', 'last_neg_delta_time','last_neg_rate']

    def fetch_position_test(self,exchangeApi):
        exchange_name = self.exchange_name
        market_type = self.market_type
        for i in range(5):
            try:
                ret_dict = exchangeApi.fetch_position(exchange_name, market_type, None)
                assert ret_dict['success']
                break
            except Exception as e:
                ms = '持仓信息获取失败,err: %s' % str(e)
                Ding.send_dingding_msg(ms, False, [])
                logErr.error(ms)
                return None
            time.sleep(0.5)
        return ret_dict['data']

    def fetch_trades_test(self,exchangeApi, symbol):
        exchange_name = self.exchange_name
        market_type = self.market_type
        for i in range(5):
            try:
                ret_dict = exchangeApi.fetch_orders(exchange_name, market_type, symbol)
                assert ret_dict['success']
                break
            except Exception as e:
                ms = '成交信息获取失败,err: %s' % str(ret_dict)
                Ding.send_dingding_msg(ms, False, [])
                logErr.error(ms)
                return None
            time.sleep(0.5)

        if len(ret_dict['data']) == 0:
            return {'long': {}, 'short': {}}
        result = {}
        long_result = {}
        short_result = {}

        for item in ret_dict['data']:
            open1 = item['info']['side'] == 'BUY' and item['info']['positionSide'] == 'LONG'
            open2 = item['info']['side'] == 'SELL' and item['info']['positionSide'] == 'SHORT'
            if item['info']['status'] == 'FILLED' and (open1 or open2):
                if item['info']['positionSide'] == 'LONG':
                    long_result[item['timestamp']] = item
                elif item['info']['positionSide'] == 'SHORT':
                    short_result[item['timestamp']] = item
        result['long'] = long_result
        result['short'] = short_result
        return result

    def fetch_open_order_test(self,exchangeApi, symbol=None):
        '''
        symbol不传，获取所有未成交订单
        '''
        exchange_name = self.exchange_name
        market_type = self.market_type
        for i in range(5):
            try:
                ret_dict = exchangeApi.fetch_open_order(exchange_name, market_type, symbol)
                assert ret_dict['success']
                break
            except Exception as e:
                ms = '未成交订单信息获取失败,err: %s' % str(ret_dict)
                Ding.send_dingding_msg(ms, False, [])
                logErr.error(ms)
                return None
            time.sleep(0.5)
        return ret_dict['data']

    def future_place_gtc_order_test(self,exchangeApi, symbol, long_or_short, price, amount, sig=None):
        exchange_name = self.exchange_name
        market_type = self.market_type
        tif = 'GTC'
        if sig[0:2] == 'in':
            if sig[0:4] == 'inre':
                sig = sig + '_' + symbol + '_' + str(np.random.randint(1000))
            else:
                sig = sig + '_' + symbol + '_' + str(int(time.time() * 1000))
            try:
                ret_dict = exchangeApi.future_place_gtc_order(exchange_name, market_type, tif, symbol, long_or_short, price,
                                                          amount, sig)
            except Exception as e:
                ms = '开仓下单失败，symbol:%s,long_or_short: %s,price: %s,amount: %s,err: %s' % (symbol,long_or_short,price,amount,str(e))
                Ding.send_dingding_msg(ms, False, [])
                logErr.error(ms)
                return None

        else:
            for i in range(5):
                try:
                    sig = sig + '_' + symbol + '_' + str(int(time.time() * 1000))
                    ret_dict = exchangeApi.future_place_gtc_order(exchange_name, market_type, tif, symbol,
                                                                  long_or_short, price, amount, sig)
                    assert ret_dict['success']
                    break
                except Exception as e:
                    ms = '平仓下单失败,人工介入，symbol:%s,long_or_short: %s,price: %s,amount: %s,err: %s' % (symbol, long_or_short, price, amount, str(e))
                    Ding.send_dingding_msg(ms, False, [])
                    logErr.error(ms)
                    return None

                time.sleep(0.5)
        return ret_dict

    def cancel_order_test(self,exchangeApi, symbol, order_id):
        exchange_name = self.exchange_name
        market_type = self.market_type
        for i in range(10):
            try:
                ret_dict = exchangeApi.cancel_order(exchange_name, market_type, symbol, order_id)
                assert ret_dict['success']
                break
            except Exception as e:
                ms = '撤单失败，symbol:%s,order_id: %s,err: %s' % (symbol,order_id,str(ret_dict))
                Ding.send_dingding_msg(ms, False, [])
                logErr.error(ms)
                return None
            time.sleep(0.5)
        return ret_dict

    def fetch_markets_test(self,exchangeApi):
        exchange_name = self.exchange_name
        market_type = self.market_type
        for i in range(5):
            try:
                ret_dict = exchangeApi.fetch_markets(exchange_name, market_type)
                assert ret_dict['success']
                assert ret_dict['data']
                break
            except Exception as e:
                ms = 'markets公共信息获取失败,err: %s' % str(e)
                Ding.send_dingding_msg(ms, False, [])
                logErr.error(ms)
                return None
            time.sleep(0.5)
        symbols = ret_dict['data']
        temp_dict = {'symbol': [], 'price_min': [], 'amount_min': []}
        for s in symbols:
            temp_dict['symbol'].append(s['id'])
            temp_dict['price_min'].append(s['limits']['price']['min'])
            temp_dict['amount_min'].append(s['limits']['amount']['min'])
        return pd.DataFrame(temp_dict)

    def get_depth_info(self,symbol):
        key_mq = "%s_MQ" % symbol
        try:
            data = self.rds.lrange(key_mq, 0, -1)
        except Exception as e:
            ms = 'depth获取失败,symbol:%s,err: %s' %(symbol,str(e))
            Ding.send_dingding_msg(ms, False, [])
            logErr.error(ms)
            return None
        temp = [json.loads(i.decode()) for i in data]
        depth_dict = []
        for ret_dict in temp:
            bid_price = np.array([decimal.Decimal(bids[0]) for bids in ret_dict['b']])
            bid_volume = np.array([decimal.Decimal(bids[1]) for bids in ret_dict['b']])
            ask_price = np.array([decimal.Decimal(asks[0]) for asks in ret_dict['a']])
            ask_volume = np.array([decimal.Decimal(asks[1]) for asks in ret_dict['a']])
            book_dict = {'symbol': ret_dict['s'], 'bid_price': bid_price, 'bid_volume': bid_volume,
                         'ask_price': ask_price, 'ask_volume': ask_volume, 'time': ret_dict['E']}
            depth_dict.append(book_dict)
        return depth_dict

    def _cal_pos_price(self,value, timeserise):
        start_value = [0] * len(value)
        timeserise_value = [0] * len(value)
        sum_value = [0] * len(value)
        delta_time = [0] * len(value)

        for i in range(len(value)):
            if i == 0:
                continue
            if value[i] >= value[i - 1]:
                if start_value[i - 1] == 0:
                    start_value[i] = value[i - 1]
                    timeserise_value[i] = timeserise[i - 1]
                else:
                    start_value[i] = start_value[i - 1]
                    timeserise_value[i] = timeserise_value[i - 1]
            else:
                start_value[i] = 0
                timeserise_value[i] = 0

            if start_value[i] != 0:
                sum_value[i] = value[i] - start_value[i]
                delta_time[i] = delta_time[i - 1] + 1

        return start_value, sum_value, timeserise_value, delta_time

    def _cal_neg_price(self,value, timeserise):
        start_value = [0] * len(value)
        timeserise_value = [0] * len(value)
        sum_value = [0] * len(value)
        delta_time = [0] * len(value)

        for i in range(len(value)):
            if i == 0:
                continue
            if value[i] <= value[i - 1]:
                if start_value[i - 1] == 0:
                    start_value[i] = value[i - 1]
                    timeserise_value[i] = timeserise[i - 1]
                else:
                    start_value[i] = start_value[i - 1]
                    timeserise_value[i] = timeserise_value[i - 1]
            else:
                start_value[i] = 0
                timeserise_value[i] = 0

            if start_value[i] != 0:
                sum_value[i] = value[i] - start_value[i]
                delta_time[i] = delta_time[i - 1] + 1
        return start_value, sum_value, timeserise_value, delta_time

    def get_feature_data(self,data):
        data = data.tail(60)
        out_time = data.tail(1)['open_time'].values[0]
        data['day_amount'] = data['amount'].rolling(20).sum()
        data['day_volume'] = data['volume'].rolling(20).sum()
        data['day_avg_price'] = data['day_amount'] / data['day_volume']

        data['amp'] = (data['high'] - data['low']) / data['close'].shift(1)
        data['max_oc'] = data[['open', 'close']].apply(max, axis=1)
        data['min_oc'] = data[['open', 'close']].apply(min, axis=1)
        data['up_amp'] = (data['high'] - data['max_oc']) / data['close'].shift(1)
        data['down_amp'] = (data['min_oc'] - data['low']) / data['close'].shift(1)
        data['amp_ma'] = data['amp'].rolling(20).mean()
        data['up_amp_ma'] = data['up_amp'].rolling(20).mean()
        data['down_amp_ma'] = data['down_amp'].rolling(20).mean()
        data['delta_ma'] = data['amp'].diff(1)

        data['avg_amount'] = data['amount'] / data['trades']
        data['take_pct'] = data['taker_amount'] / data['amount']
        data['delta_take_pct'] = data['take_pct'].diff(1)
        data['take_pct_ma'] = data['take_pct'].rolling(20).mean()

        data['pct1'] = data['close'].pct_change(1)
        data['skew'] = data['pct1'].rolling(20).skew()
        data['kurt'] = data['pct1'].rolling(20).kurt()
        data['up_vol'] = data['pct1'].rolling(20).apply(lambda x: (x[x > 0] ** 2).sum() / (x ** 2).sum())

        data['up_vol_ma'] = data['up_vol'].rolling(20).mean()
        data['skew_ma'] = data['skew'].rolling(20).mean()
        data['kurt_ma'] = data['kurt'].rolling(20).mean()
        data['delta_up_vol'] = data['up_vol'].diff(1)
        data['delta_skew'] = data['skew'].diff(1)
        data['delta_kurt'] = data['kurt'].diff(1)

        data['pct3'] = data['close'].pct_change(3)
        data['pct5'] = data['close'].pct_change(5)
        data['pct10'] = data['close'].pct_change(10)

        data['last_pct1'] = data['pct1'].shift(1)
        data['last_pct3'] = data['pct3'].shift(1)
        data['last_pct5'] = data['pct5'].shift(1)
        data['last_pct10'] = data['pct10'].shift(1)

        data['ma_pct1'] = data['pct1'].rolling(20).mean()
        data['ma_pct3'] = data['pct3'].rolling(20).mean()
        data['ma_pct5'] = data['pct5'].rolling(20).mean()
        data['ma_pct10'] = data['pct10'].rolling(20).mean()

        data['std_pct1'] = data['pct1'].rolling(20).std()
        data['std_pct3'] = data['pct3'].rolling(20).std()
        data['std_pct5'] = data['pct5'].rolling(20).std()
        data['std_pct10'] = data['pct10'].rolling(20).std()

        data['ma5'] = data['close'].rolling(5).mean()
        data['ma10'] = data['close'].rolling(10).mean()
        data['ma20'] = data['close'].rolling(20).mean()
        data['ma30'] = data['close'].rolling(30).mean()
        data['ma60'] = data['close'].rolling(60).mean()

        data['bias'] = data['close'] / data['ma60'] - 1
        data['bias1'] = data['close'] / data['day_avg_price'] - 1

        data['ma_bbi'] = (data['ma5'] + data['ma10'] + data['ma20'] + data['ma30'] + data['ma60']) / 5
        data['ddd'] = data['ma5'] / data['ma60'] - 1
        data['dma'] = data['ddd'].rolling(10).mean()

        data['pos_price'], data['pos_value'], data['pos_time'], data['pos_delta_time'] = self._cal_pos_price(
            data['close'].values, data['open_time'].values)
        data['neg_price'], data['neg_value'], data['neg_time'], data['neg_delta_time'] = self._cal_neg_price(
            data['close'].values, data['open_time'].values)

        data['pos_value1'] = data['pos_value'] / data['pos_price']
        data['neg_value1'] = data['neg_value'] / data['neg_price']
        data['pos_value1'] = data['pos_value1'].fillna(0)
        data['neg_value1'] = data['neg_value1'].fillna(0)

        data['pos_rate'] = data['pos_value1'] / data['pos_delta_time']
        data['neg_rate'] = data['neg_value1'] / data['neg_delta_time']
        data['pos_rate'] = data['pos_rate'].fillna(0)
        data['neg_rate'] = data['neg_rate'].fillna(0)

        data['last_pos_price'] = data['pos_price'].replace(0, method='ffill')
        data['last_pos_value'] = data['pos_value'].replace(0, method='ffill')
        data['last_pos_value1'] = data['pos_value1'].replace(0, method='ffill')
        data['last_pos_delta_time'] = data['pos_delta_time'].replace(0, method='ffill')
        data['last_pos_rate'] = data['pos_rate'].replace(0, method='ffill')

        data['last_pos_price'] = (data['last_pos_price'] - data['pos_price']).replace(0, method='ffill')
        data['last_pos_value'] = (data['last_pos_value'] - data['pos_value']).replace(0, method='ffill')
        data['last_pos_value1'] = (data['last_pos_value1'] - data['pos_value1']).replace(0, method='ffill')
        data['last_pos_delta_time'] = (data['last_pos_delta_time'] - data['pos_delta_time']).replace(0, method='ffill')
        data['last_pos_rate'] = (data['last_pos_rate'] - data['pos_rate']).replace(0, method='ffill')

        data['last_neg_price'] = data['neg_price'].replace(0, method='ffill')
        data['last_neg_value'] = data['neg_value'].replace(0, method='ffill')
        data['last_neg_value1'] = data['neg_value1'].replace(0, method='ffill')
        data['last_neg_delta_time'] = data['neg_delta_time'].replace(0, method='ffill')
        data['last_neg_rate'] = data['neg_rate'].replace(0, method='ffill')

        data['last_neg_price'] = (data['last_neg_price'] - data['neg_price']).replace(0, method='ffill')
        data['last_neg_value'] = (data['last_neg_value'] - data['neg_value']).replace(0, method='ffill')
        data['last_neg_value1'] = (data['last_neg_value1'] - data['neg_value1']).replace(0, method='ffill')
        data['last_neg_delta_time'] = (data['last_neg_delta_time'] - data['neg_delta_time']).replace(0, method='ffill')
        data['last_neg_rate'] = (data['last_neg_rate'] - data['neg_rate']).replace(0, method='ffill')

        data['buy1'] = ((data['last_neg_value1'] <= -0.005) & (data['last_neg_rate'] < -0.002) & (
                    data['last_neg_delta_time'] > 1) & (data['pct1'] > 0))
        data['buy2'] = ((data['last_neg_value1'] <= -0.008) & (data['last_neg_rate'] < -0.0015) & (
                    data['last_neg_delta_time'] > 1) & (data['pct1'] > 0))
        data['buy3'] = ((data['last_neg_value1'] <= -0.01) & (data['last_neg_rate'] < -0.001) & (
                    data['last_neg_delta_time'] > 1) & (data['pct1'] > 0))
        data['buy3'] = (data['buy1'] | data['buy2'] | data['buy3']).apply(lambda x: 1 if x else 0)
        data['buy4'] = data['buy3'].rolling(5).apply(lambda x: 1 if len(x[x >= 1]) > 0 else 0)
        data['buy'] = ((data['buy4'] == 1) & (data['close'] < data['ma5'])).apply(lambda x: 1 if x else 0)

        data['sell1'] = ((data['last_pos_value1'] >= 0.005) & (data['last_pos_rate'] > 0.002) & (
                    data['last_pos_delta_time'] > 1) & (data['pct1'] < 0))
        data['sell2'] = ((data['last_pos_value1'] >= 0.008) & (data['last_pos_rate'] > 0.0015) & (
                    data['last_pos_delta_time'] > 1) & (data['pct1'] < 0))
        data['sell3'] = ((data['last_pos_value1'] >= 0.01) & (data['last_pos_rate'] > 0.001) & (
                    data['last_pos_delta_time'] > 1) & (data['pct1'] < 0))
        data['sell3'] = (data['sell1'] | data['sell2'] | data['sell3']).apply(lambda x: -1 if x else 0)
        data['sell4'] = data['sell3'].rolling(5).apply(lambda x: -1 if len(x[x <= -1]) > 0 else 0)
        data['sell'] = ((data['sell4'] == -1) & (data['close'] > data['ma5'])).apply(lambda x: -1 if x else 0)
        data['signal'] = data['buy'] + data['sell']
        predict_data = data[(data['open_time'] == out_time)]

        if predict_data['signal'].values[0] == 1:
            predict_data['predict_pos'] = self.gbm_pos.predict(predict_data[self.column_list],num_iteration=self.gbm_pos.best_iteration)
            pos = predict_data['predict_pos'].values[0]
            predict_data['predict_pos'] = predict_data['predict_pos'].apply(lambda x: 1 if x >= self.pos_threshold else 0)
            result ={'symbol': predict_data['symbol'].values[0], 'signal': predict_data['predict_pos'].values[0], 'open_time': out_time,'close': predict_data['close'].values[0],'predict_pos':pos}
            return result

        elif predict_data['signal'].values[0] == -1:
            predict_data['predict_neg'] = self.gbm_neg.predict(predict_data[self.column_list],num_iteration=self.gbm_neg.best_iteration)
            neg = predict_data['predict_neg'].values[0]
            predict_data['predict_neg'] = predict_data['predict_neg'].apply(lambda x: -1 if x >= self.neg_threshold else 0)
            result = {'symbol': predict_data['symbol'].values[0], 'signal': predict_data['predict_neg'].values[0], 'open_time': out_time,'close': predict_data['close'].values[0],'predict_neg':neg}
            return result
        else:
            result = {'symbol': predict_data['symbol'].values[0], 'signal': 0, 'open_time': out_time,'close': predict_data['close'].values[0],'predict0':None}
            return result

    def timeStamp2Datetime(self,timeStamp):
        timeArr = pd.to_datetime(timeStamp,unit='ms') + pd.Timedelta(hours=8)
        return str(timeArr)[0:19]

    def get_channel_rds(self,symbols, all_data_dict):
        all_symbol_dict = {}
        while 1:
            rds_data = self.rds.mget(symbols)
            if rds_data is None:
                ms = 'rds_data is None,人工介入，rds_data:%s'%rds_data
                Ding.send_dingding_msg(ms, False, [])
                logInf.info(ms)
                continue
            elif len(rds_data)==0:
                ms = 'rds_data 为空,人工介入，rds_data:%s' % rds_data
                Ding.send_dingding_msg(ms, False, [])
                logInf.info(ms)
                continue

            for msg in rds_data:
                out = 0
                if msg is None or len(msg)==0:
                    ms = 'rds_data中msg为None，rds_data:%s,msg:%s' %(rds_data,msg)
                    Ding.send_dingding_msg(ms, False, [])
                    logInf.info(ms)
                    continue
                if msg['ps'] in all_symbol_dict:
                    if all_symbol_dict[msg['ps']]['open_time'][-1] != self.timeStamp2Datetime(msg['k']['t']):
                        out = 1
                else:
                    out = 1

                if out == 1:
                    last_dict = {'event_time': [self.timeStamp2Datetime(msg['E'])], 'symbol': [msg['ps']],
                                 'open_time': [self.timeStamp2Datetime(msg['k']['t'])],
                                 'end_time': [self.timeStamp2Datetime(msg['k']['T'])], 'open': [msg['k']['o']],
                                 'close': [msg['k']['c']],
                                 'high': [msg['k']['h']], 'low': [msg['k']['l']], 'volume': [msg['k']['v']],
                                 'trades': [msg['k']['n']], 'amount': [msg['k']['q']], 'taker_volume': [msg['k']['V']],
                                 'taker_amount': [msg['k']['Q']]}
                    all_symbol_dict[msg['ps']] = last_dict
                    last_df = pd.DataFrame(last_dict)
                    columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'trades', 'taker_volume', 'taker_amount']
                    last_df[columns] = last_df[columns].astype(float)

                    data_last_open_time = all_data_dict[msg['ps']].iloc[-1]['open_time']
                    delta_symbol_time = (pd.to_datetime(last_dict['open_time']) - pd.to_datetime(data_last_open_time)).seconds
                    if delta_symbol_time > 60:
                        ms = 'symbol:%s,与上根k线时间相差大于60秒，上根k线时间:%s，此根k线时间:%s，时间差:%s,本地存储数据:%s,rds_data中msg数据:%s' %(msg['ps'],data_last_open_time,last_dict['open_time'],delta_symbol_time,all_data_dict[msg['ps']],msg)
                        Ding.send_dingding_msg(ms, False, [])
                        logInf.info(ms)
                    elif delta_symbol_time == 60:
                        all_data_dict[msg['ps']] = pd.concat([all_data_dict[msg['ps']], last_df]).tail(60)

                        ret = self.get_feature_data(all_data_dict[msg['ps']])
                        threading.Thread(target=self.run_task_with_process, args=(ret, msg['ps'])).start()

    def check_orders(self,symbols):
        '''
        每10秒进行check
        获取全部挂单，对每笔开仓挂单进行check，判断是否撤单还是继续等待
        :return:
        '''
        while 1:

            t1 = time.time()
            logInf.info("start check_orders")
            order_list = self.fetch_open_order_test(self.exchangeApi)
            if order_list == None:
                continue

            for order in order_list:
                order_info = order['info']
                symbol = order_info['symbol']

                if symbol not in symbols:
                    continue
                amount = float(order['remaining'])
                logInf.info("check_orders symbol:%s,order_info:%s"%(symbol,order_info))
                if order_info['clientOrderId'][0:2] == 'in':
                    now = time.time()
                    time_diff = now - float(order_info['time']) / 1000

                    if order_info['clientOrderId'][0:4] == 'inre':
                        time_diff = now - float(order_info['clientOrderId'][4:17]) / 1000

                    if time_diff > 120:
                        logInf.info('开仓时间大于2分钟，撤单,symbol:%s,订单信息 %s' % (symbol,order_info))
                        try:
                            self.cancel_order_test(self.exchangeApi, symbol, order['id'])
                        except Exception as e:
                            logErr.error(str(e))
                            continue
                    else:
                        order_book = self.get_depth_info(symbol)[0]

                        if order_book == None:
                            logInf.info('行情获取失败，撤单,%s' % order_info)
                            try:
                                self.cancel_order_test(self.exchangeApi, symbol, order['id'])
                            except Exception as e:
                                logErr.error(str(e))
                                continue
                        elif order_info['positionSide'] == 'LONG':
                            if float(order_book['bid_price'][0]) / float(order_info['price']) - 1 >= 0.003:
                                logInf.info('多头盘口涨幅涨幅过大，撤单,盘口价格 %s，挂单价格 %s,价格差值 %s,订单信息 %s' % (
                                order_book['bid_price'][0], order_info['price'],
                                float(order_book['bid_price'][0]) / float(order_info['price']) - 1, order_info))
                                try:
                                    self.cancel_order_test(self.exchangeApi, symbol, order['id'])
                                except Exception as e:
                                    logErr.error(str(e))
                                    continue
                            elif float(order_info['price']) <= float(order_book['bid_price'][3]):
                                logInf.info('多头盘口价格大于4档，撤单重报,盘口价格 %s，挂单价格 %s,订单信息 %s' % (
                                order_book['bid_price'][4], order_info['price'], order_info))
                                try:
                                    ret = self.cancel_order_test(self.exchangeApi, symbol, order['id'])
                                except Exception as e:
                                    logErr.error(str(e))
                                    continue
                                if ret is None:
                                    logInf.info("多头盘口价格大于4档，撤单失败,symbol:%s,order_id:%s"%(symbol,order['id']))
                                    continue
                                sig = 'inre' + str(order['timestamp'])
                                logInf.info('多头重报,symbol %s,挂单价格 %s,自定义订单id %s' % (symbol, order_book['bid_price'][0], sig))
                                ret = self.future_place_gtc_order_test(self.exchangeApi, symbol, 'long',
                                                                  price=float(order_book['bid_price'][0]),
                                                                  amount=float(amount), sig=sig)
                                try:
                                    self.trade_id_list.append(ret['data']['orderId'])
                                except Exception as e:
                                    ms = '多头开仓重报失败,symbol %s,挂单价格 %s,自定义订单id %s，err:%s' % (symbol, order_book['bid_price'][0], sig,str(e))
                                    Ding.send_dingding_msg(ms, False, [])
                                    logErr.error(ms)


                        elif order_info['positionSide'] == 'SHORT':
                            if 1 - float(order_book['ask_price'][0]) / float(order_info['price']) >= 0.003:
                                logInf.info('空头盘口涨幅涨幅过大，撤单,盘口价格%s，挂单价格%s,价格差值%s,订单信息%s' % (
                                order_book['ask_price'][0], order_info['price'],
                                1 - float(order_book['ask_price'][0]) / float(order_info['price']), order_info))
                                try:
                                    self.cancel_order_test(self.exchangeApi, symbol, order['id'])
                                except Exception as e:
                                    logErr.error(str(e))
                                    continue
                            elif float(order_info['price']) >= float(order_book['ask_price'][3]):
                                logInf.info('空头盘口盘口价格大于4档，撤单重报,盘口价格%s，挂单价格%s,订单信息%s' % (
                                order_book['ask_price'][4], order_info['price'], order_info))
                                try:
                                    ret = self.cancel_order_test(self.exchangeApi, symbol, order['id'])
                                except Exception as e:
                                    logErr.error(str(e))
                                    continue
                                if ret is None:
                                    logInf.info("空头盘口价格大于4档，撤单失败,symbol:%s,order_id:%s"%(symbol,order['id']))
                                    continue

                                sig = 'inre' + str(order['timestamp'])
                                logInf.info('空头重报,symbol %s,挂单价格 %s,自定义订单id %s' % (symbol, order_book['ask_price'][0], sig))
                                ret = self.future_place_gtc_order_test(self.exchangeApi, symbol, 'short',
                                                                  price=float(order_book['ask_price'][0]),
                                                                  amount=float(amount), sig=sig)
                                try:
                                    self.trade_id_list.append(ret['data']['orderId'])
                                except Exception as e:
                                    ms = '空头开仓重报失败,symbol %s,挂单价格 %s,自定义订单id %s，err:%s' % (symbol, order_book['ask_price'][0], sig, str(e))
                                    Ding.send_dingding_msg(ms, False, [])
                                    logErr.error(ms)

                elif order_info['clientOrderId'][0:2] == 'ou':
                    now = time.time()
                    time_diff = now - float(order_info['time']) / 1000
                    if time_diff > 10:
                        order_book = self.get_depth_info(symbol)[0]
                        if order_book == None:
                            continue
                        elif order_info['positionSide'] == 'LONG':
                            sum_bid_money = sum(order_book['bid_price'][0:10] * order_book['bid_volume'][0:10])
                            sum_ask_money = sum(order_book['ask_price'][0:10] * order_book['ask_volume'][0:10])
                            c1 = time_diff >= 120
                            c2 = sum_ask_money / sum_bid_money >= 2
                            c3 = sum_ask_money / sum_bid_money >= 1.3 and float(order_info['price']) <= float(
                                order_book['ask_price'][3])
                            c4 = float(order_info['price']) >= float(order_book['ask_price'][4])

                            if c1 or c2 or c3 or c4:
                                logInf.info('多头平仓撤单,symbol %s, %s' % (symbol, order_info))
                                try:
                                    ret = self.cancel_order_test(self.exchangeApi, symbol, order['id'])
                                except Exception as e:
                                    logErr.error(str(e))
                                    continue

                                if ret is None:
                                    logInf.info("多头平仓撤单失败,symbol %s, %s"% (symbol, order_info))
                                    continue

                                logInf.info('多头平仓重报,symbol %s,挂单价格 %s' % (symbol, order_book['bid_price'][1]))
                                ret = self.future_place_gtc_order_test(self.exchangeApi, symbol, 'close_long',
                                                            price=float(order_book['bid_price'][1]), amount=float(amount),
                                                            sig='ou')
                                try:
                                    self.del_trade_id_list.append(ret['data']['orderId'])
                                except Exception as e:
                                    ms = '多头平仓重报下单失败，请人工介入，symbol:%s,amount: %s,err:%s' % (symbol, amount,str(e))
                                    Ding.send_dingding_msg(ms, False, [])
                                    logInf.info(ms)

                        elif order_info['positionSide'] == 'SHORT':
                            sum_bid_money = sum(order_book['bid_price'][0:10] * order_book['bid_volume'][0:10])
                            sum_ask_money = sum(order_book['ask_price'][0:10] * order_book['ask_volume'][0:10])
                            c1 = time_diff >= 120
                            c2 = sum_bid_money / sum_ask_money >= 2
                            c3 = sum_bid_money / sum_ask_money >= 1.3 and float(order_info['price']) >= float(
                                order_book['bid_price'][3])
                            c4 = float(order_info['price']) <= float(order_book['bid_price'][4])

                            if c1 or c2 or c3 or c4:
                                logInf.info('空头平仓撤单,symbol %s, %s' % (symbol, order_info))
                                try:
                                    ret = self.cancel_order_test(self.exchangeApi, symbol, order['id'])
                                except Exception as e:
                                    logErr.error(str(e))
                                    continue

                                if ret is None:
                                    logInf.info("空头平仓撤单失败,symbol %s, %s"% (symbol, order_info))
                                    continue

                                logInf.info('空头平仓重报,symbol %s,挂单价格 %s' % (symbol, order_book['ask_price'][1]))
                                ret = self.future_place_gtc_order_test(self.exchangeApi, symbol, 'close_short',
                                                            price=float(order_book['ask_price'][1]), amount=float(amount),
                                                            sig='ou')
                                try:
                                    self.del_trade_id_list.append(ret['data']['orderId'])
                                except Exception as e:
                                    ms = '空头平仓重报下单失败，请人工介入，symbol:%s,amount: %s,err:%s' % (symbol, amount,str(e))
                                    Ding.send_dingding_msg(ms, False, [])
                                    logInf.info(ms)

            t2 = time.time()
            dalta_t = t2 - t1
            time.sleep(max(10 - dalta_t, 0))

    def check_all_positions(self,symbols):
        '''
        每一分钟进行check
        获取全部持仓，分别计算每个品种每笔交易的止盈，止损，时间出场
        :return:
        '''
        while 1:
            t3 = time.time()
            logInf.info("start check_all_positions")
            position_list = self.fetch_position_test(self.exchangeApi)
            if position_list == None:
                continue
            sym_dict = {}
            for pos in position_list:
                if pos['symbol'] not in sym_dict:
                    sym_dict[pos['symbol']] = {'SHORT': {}, 'LONG': {}}
                    sym_dict[pos['symbol']][pos['positionSide']] = pos
                else:
                    sym_dict[pos['symbol']][pos['positionSide']] = pos

            for symbol in sym_dict:
                if symbol not in symbols:
                    continue
                logInf.info("check_position symbol:%s" % (symbol,))
                trade_order_list = self.fetch_trades_test(self.exchangeApi, symbol)
                if trade_order_list == None:
                    continue
                if len(trade_order_list['long']) == 0 and len(trade_order_list['short']) == 0:
                    continue
                if len(trade_order_list['long']) > 0 and len(sym_dict[symbol]['LONG']) > 0:

                    keys_list = sorted(trade_order_list['long'].keys(), reverse=True)
                    temp_amt = float(sym_dict[symbol]['LONG']['positionAmt'])

                    for trade_order_id in keys_list:
                        if temp_amt <= 0:
                            break
                        trade_order = trade_order_list['long'][trade_order_id]

                        id_con = (trade_order['id'] in self.trade_id_list) and (trade_order['id'] not in self.del_trade_id_list) and (
                                    trade_order['id'] not in self.deal_trade_id_list)

                        if trade_order['status'] == 'closed' and trade_order['clientOrderId'][0:2] == 'in' and id_con:
                            time_diff = time.time() - float(trade_order['info']['time']) / 1000
                            pf = float(sym_dict[symbol]['LONG']['markPrice']) / float(trade_order['price']) - 1
                            con1 = 0
                            if time_diff >= 330:
                                con1 = 1
                                logInf.info("多头时间离场，%s,%s,%s" % (symbol, time_diff, trade_order['info']))
                            elif pf > 0.03:
                                con1 = 1
                                logInf.info("多头止盈离场，%s,%s,%s" % (symbol, pf, trade_order['info']))
                            elif pf <= -0.005:
                                con1 = 1
                                logInf.info("多头止损离场，%s,%s,%s" % (symbol, pf, trade_order['info']))
                            if con1 == 1:

                                order_book = self.get_depth_info(symbol)[0]
                                if order_book != None:
                                    sum_bid_money = sum(order_book['bid_price'][0:10] * order_book['bid_volume'][0:10])
                                    sum_ask_money = sum(order_book['ask_price'][0:10] * order_book['ask_volume'][0:10])
                                    out_price = float(order_book['ask_price'][0])
                                    if sum_ask_money / sum_bid_money >= 1.5:
                                        out_price = float(order_book['bid_price'][0])

                                    ret = self.future_place_gtc_order_test(self.exchangeApi, symbol, 'close_long', price=out_price,
                                                                      amount=float(trade_order['amount']), sig='ou')
                                    try:
                                        self.del_trade_id_list.append(ret['data']['orderId'])
                                        self.deal_trade_id_list.append(trade_order['id'])
                                        temp_amt = temp_amt - float(trade_order['amount'])
                                    except Exception as e:
                                        logErr.error(str(e))
                                        continue

                                    if ret is None:
                                        ms = '多头平仓下单失败，请人工介入，symbol:%s,amount: %s,' % (symbol, float(trade_order['amount']))
                                        Ding.send_dingding_msg(ms, False, [])
                                        logInf.info(ms)
                                else:
                                    ms = '多头行情获取失败，需要人工介入平仓，symbol:%s,amount: %s,' % (symbol, float(trade_order['amount']))
                                    Ding.send_dingding_msg(ms, False, [])
                                    logInf.info(ms)

                if len(trade_order_list['short']) > 0 and len(sym_dict[symbol]['SHORT']) > 0:
                    keys_list = sorted(trade_order_list['short'].keys(), reverse=True)
                    temp_amt = abs(float(sym_dict[symbol]['SHORT']['positionAmt']))

                    for trade_order_id in keys_list:
                        if temp_amt <= 0:
                            break
                        trade_order = trade_order_list['short'][trade_order_id]
                        id_con = (trade_order['id'] in self.trade_id_list) and (trade_order['id'] not in self.del_trade_id_list) and (
                                    trade_order['id'] not in self.deal_trade_id_list)
                        if trade_order['status'] == 'closed' and trade_order['clientOrderId'][0:2] == 'in' and id_con:
                            time_diff = time.time() - int(trade_order['info']['time']) / 1000
                            pf = 1 - float(sym_dict[symbol]['SHORT']['markPrice']) / float(trade_order['price'])
                            con1 = 0
                            if time_diff >= 330:
                                con1 = 1
                                logInf.info("空头时间离场，%s,%s,%s" % (symbol, time_diff, trade_order['info']))
                            elif pf > 0.03:
                                con1 = 1
                                logInf.info("空头止盈离场，%s,%s,%s" % (symbol, pf, trade_order['info']))
                            elif pf <= -0.005:
                                con1 = 1
                                logInf.info("空头止损离场，%s,%s,%s" % (symbol, pf, trade_order['info']))
                            if con1 == 1:
                                temp_amt = temp_amt - float(trade_order['amount'])
                                order_book = self.get_depth_info(symbol)[0]

                                if order_book != None:
                                    sum_bid_money = sum(order_book['bid_price'][0:10] * order_book['bid_volume'][0:10])
                                    sum_ask_money = sum(order_book['ask_price'][0:10] * order_book['ask_volume'][0:10])
                                    out_price = float(order_book['bid_price'][0])
                                    if sum_bid_money / sum_ask_money >= 1.5:
                                        out_price = float(order_book['ask_price'][0])

                                    ret = self.future_place_gtc_order_test(self.exchangeApi, symbol, 'close_short', price=out_price,
                                                                      amount=float(trade_order['amount']), sig='ou')
                                    try:
                                        self.del_trade_id_list.append(ret['data']['orderId'])
                                        self.deal_trade_id_list.append(trade_order['id'])
                                    except Exception as e:
                                        logErr.error(str(e))
                                        continue

                                    if ret is None:
                                        ms = '空头平仓下单失败，请人工介入，symbol:%s,amount: %s,' % (symbol,float(trade_order['amount']))
                                        Ding.send_dingding_msg(ms, False, [])
                                        logInf.info(ms)
                                else:
                                    ms = '空头行情获取失败，需要人工介入平仓，symbol:%s,amount: %s,' % (symbol, float(trade_order['amount']))
                                    Ding.send_dingding_msg(ms, False, [])
                                    logInf.info(ms)

            t4 = time.time()
            dalta_t1 = t4 - t3
            time.sleep(max(60 - dalta_t1, 0))

    def run_task_with_process(self,ret, symbol):
        if ret['signal'] == 1:
            logInf.info(ret)
            long_or_short = 'long'
            order_book = self.get_depth_info(symbol)[0]
            price = ret['close']
            if order_book != None:
                price = float(order_book['bid_price'][0])
                sum_bid_money = sum(order_book['bid_price'][0:10] * order_book['bid_volume'][0:10])
                sum_ask_money = sum(order_book['ask_price'][0:10] * order_book['ask_volume'][0:10])
                if sum_bid_money / sum_ask_money >= 3:
                    price = float(order_book['ask_price'][0])

            price_min = float(self.amount_df[self.amount_df['symbol'] == symbol]['price_min'].values[0])*1.1
            if price>price_min:
                amount_min = float(self.amount_df[self.amount_df['symbol'] == symbol]['amount_min'].values[0])
                str_amount_min = str(self.amount_df[self.amount_df['symbol'] == symbol]['amount_min'].values[0])
                cal_amount = str(max(math.ceil(self.every_money / price / amount_min), 1))
                amount = decimal.Decimal(cal_amount) * decimal.Decimal(str_amount_min)

                logInf.info("开始做多, %s,%s, %s: " % (symbol, price, amount))
                ret = self.future_place_gtc_order_test(self.exchangeApi, symbol, long_or_short, price=decimal.Decimal(str(price)),
                                                  amount=amount, sig='in')
                try:
                    self.trade_id_list.append(ret['data']['orderId'])
                except Exception as e:
                    ms = '多头初步开仓下单失败，symbol:%s,amount: %s,err:%s' % (symbol, amount,str(e))
                    logErr.error(ms)
            else:
                logInf.info("多头不下单，价格小于最小价格限制, %s: " % symbol)

        elif ret['signal'] == -1:
            logInf.info(ret)
            exchangeApi = ExchangeApi()
            long_or_short = 'short'
            order_book = self.get_depth_info(symbol)[0]
            price = ret['close']
            if order_book != None:
                price = float(order_book['ask_price'][0])
                sum_bid_money = sum(order_book['bid_price'][0:10] * order_book['bid_volume'][0:10])
                sum_ask_money = sum(order_book['ask_price'][0:10] * order_book['ask_volume'][0:10])
                if sum_ask_money / sum_bid_money >= 3:
                    price = float(order_book['bid_price'][0])

            price_min = float(self.amount_df[self.amount_df['symbol'] == symbol]['price_min'].values[0])*1.1
            if price>price_min:
                amount_min = float(self.amount_df[self.amount_df['symbol'] == symbol]['amount_min'].values[0])
                str_amount_min = str(self.amount_df[self.amount_df['symbol'] == symbol]['amount_min'].values[0])
                cal_amount = str(max(math.ceil(self.every_money / price / amount_min), 1))
                amount = decimal.Decimal(cal_amount) * decimal.Decimal(str_amount_min)
                logInf.info("开始做空, %s,%s, %s: " % (symbol, price, amount))
                ret = self.future_place_gtc_order_test(exchangeApi, symbol, long_or_short, price=decimal.Decimal(str(price)),
                                                  amount=amount, sig='in')
                try:
                    self.trade_id_list.append(ret['data']['orderId'])
                except Exception as e:
                    ms = '空头初步开仓下单失败，symbol:%s,amount: %s,err:%s' % (symbol, amount,str(e))
                    logErr.error(ms)
            else:
                logInf.info("空头不下单，价格小于最小价格限制, %s: " % symbol)
        else:
            logInf.info(ret)
            logInf.info("啥都不干, %s: " % symbol)

    def my_account(self,currency='USDT'):
        while 1:
            try:
                ret_dict = self.exchangeApi.fetch_balance(self.exchange_name, self.market_type, currency)
            except Exception as e:
                ms = '账户资金获取失败，人工介入,err:%s' % str(e)
                Ding.send_dingding_msg(ms, False, [])
                logErr.error(ms)
            ms = '\n' \
                 '【lgb策略 账户信息 %s】\n' \
                 '资金信息:%s ' %(self.timeStamp2Datetime(int(time.time()*1000)),str(ret_dict['data'][currency]))
            logInf.info(ms)
            logInf.info(ret_dict)
            Ding.send_dingding_msg(ms, False, [])
            time.sleep(60*60)

    def main_entry(self):
        try:
            self.all_data = self.db.get_symbol_list_data(self.symbols)
            self.all_data = self.all_data.sort_values('event_time')
            self.all_data[self.columns] = self.all_data[self.columns].astype(float)
            self.amount_df = self.fetch_markets_test(self.exchangeApi)
        except Exception as e:
            ms = 'symbol数据初始化失败,人工介入,请重新载入数据库,err:%s'% str(e)
            logErr.error(ms)
            Ding.send_dingding_msg(ms, False, [])
            return None
        if self.amount_df is None:
            ms = '下单最小值参考数据初始化失败,人工介入,请重新载入数据'
            logErr.error(ms)
            Ding.send_dingding_msg(ms, False, [])
            return None
        all_data_dict = {}
        for symbol in self.symbols:
            all_data_dict[symbol] = self.all_data[self.all_data['symbol'] == symbol].tail(61)
        logInf.info('数据载入成功！！！')
        threading.Thread(target=self.check_orders, args=(self.symbols,)).start()
        threading.Thread(target=self.check_all_positions, args=(self.symbols,)).start()
        threading.Thread(target=self.my_account,).start()
        logInf.info('check_orders，check_all_positions,my_account 分别启动成功！！！')

        self.get_channel_rds(self.symbols, all_data_dict)

if __name__ == '__main__':
    strategy = lgb_strategy()
    strategy.main_entry()











