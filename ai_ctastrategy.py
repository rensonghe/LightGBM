import os
import sys
from tz_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData

)
from tzquant.trader.utility import (
    BarGenerator,
    ArrayManager,
    Interval
)
import time
import datetime
# from time import time
from HFT_binance import factor_calculation, cols_list
from factor import *
import numpy as np
import pandas as pd
import joblib
from tz_ctastrategy.base import EngineType
from tzquant.market.dingtalker import WebHook, dingmessage
# import tracemalloc
from tzquant.market.log_model import get_logger
# tracemalloc.start()
# snapshot1 = tracemalloc.take_snapshot()


tick_data_list = ['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1',
                  'ask_price2', 'ask_size2', 'bid_price2', 'bid_size2',
                  'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3',
                  'ask_price4', 'ask_size4', 'bid_price4', 'bid_size4',
                  'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5',
                  'ask_price6', 'ask_size6', 'bid_price6', 'bid_size6',
                  'ask_price7', 'ask_size7', 'bid_price7', 'bid_size7',
                  'ask_price8', 'ask_size8', 'bid_price8', 'bid_size8',
                  'ask_price9', 'ask_size9', 'bid_price9', 'bid_size9',
                  'ask_price10', 'ask_size10', 'bid_price10', 'bid_size10',
                  'price', 'size', 'cum_size', 'turnover']



class ai_TickStrategy(CtaTemplate):
    """"""
    author = "zq"

    split_count = 5
    place_rate = 2 / 10000

    init_size = 60
    pos_rate = 0.3  # 持仓比例
    record = False  # 是否记录成交记录

    # last_time = int(time.time())

    test_trigger = 10

    tick_count = 0
    test_all_done = False

    parameters = ["test_trigger"]
    variables = ["tick_count", "test_all_done"]
    log = get_logger(log_file='{}/songhe_CTA.log'.format(os.path.abspath('.')), name=f'siganl', line=True)
    tick_dict_ = {i: np.array([]) for i in tick_data_list}  # 原始tick数据存储

    tick_data = []
    tick_1s = []  # 聚合后的一秒一条数据，取【-1】
    # trade = []
    fill_order_time = 0

    i = []  # 用于记录每一秒内多少个毫秒

    col_dict_ = {key: i for i, key in enumerate(cols_list)}
    feat_dict_ = {i: np.array([]) for i in cols_list}
    df_2d = np.atleast_2d(np.zeros(123))

    threshold = 70_000
    side_long = 0.9
    side_short = 0.1
    out = 0.8
    rolling_bar = 15
    
    # print(len(y_pred_side_list))
    y_pred_side_list = []
    y_pred_side_list_ = []
    y_pred_out_list = []

    model_symbol = 'adausdt'
    # base_path = '{}/'.format(os.path.abspath('..'))
    base_path = '/tmp/strdt/deployment/songhe/'
    # try:
    #     base_path = '{}/'.format(os.path.abspath('..'))
    #     joblib.load('{}/model/{}_lightGBM_side_0.pkl'.format(base_path, model_symbol))
    # except (Exception, BaseException) as e:
    #     try:
    #         base_path = '/tmp/strdt/deployment/songhe/'
    #         joblib.load('{}/model/{}_lightGBM_side_0.pkl'.format(base_path, model_symbol))
    #     except (Exception, BaseException) as e:
    #         print('出错原因:%s-----错误所在行数:%s' % (repr(e), e.__traceback__.tb_lineno))
    #         base_path = '../..{}'.format(os.path.abspath('.'))
    model_side_0 = joblib.load('{}/model/{}_lightGBM_side_0.pkl'.format(base_path, model_symbol))
    model_side_1 = joblib.load('{}/model/{}_lightGBM_side_1.pkl'.format(base_path, model_symbol))
    model_side_2 = joblib.load('{}/model/{}_lightGBM_side_2.pkl'.format(base_path, model_symbol))
    model_side_3 = joblib.load('{}/model/{}_lightGBM_side_3.pkl'.format(base_path, model_symbol))
    model_side_4 = joblib.load('{}/model/{}_lightGBM_side_4.pkl'.format(base_path, model_symbol))
    model_out_0 = joblib.load('{}/model/{}_lightGBM_out_0.pkl'.format(base_path, model_symbol))
    model_out_1 = joblib.load('{}/model/{}_lightGBM_out_1.pkl'.format(base_path, model_symbol))
    model_out_2 = joblib.load('{}/model/{}_lightGBM_out_2.pkl'.format(base_path, model_symbol))
    model_out_3 = joblib.load('{}/model/{}_lightGBM_out_3.pkl'.format(base_path, model_symbol))
    model_out_4 = joblib.load('{}/model/{}_lightGBM_out_4.pkl'.format(base_path, model_symbol))

    # print('打印模型：',model_side_0)
    
    
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting, rolling_info=None):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting, rolling_info)
        self.last_tick = None
        self.bg = BarGenerator(self.on_bar)
        self.turnover = 0
        self.cum_size = 0
        # self.side_long = 0.9
        # self.side_short = 0.1
        self.signal_time = int(time.time())
        self.factor_time = int(time.time())
        self.kill_time = 0
        self.single_result = None
        self.strategy_time = int(time.time())
        self.ms_time = int(time.time() * 1000)
        self.print_time = time.time() // 1800
        self.strategy_trades = []
        # self.tick_1s = [None]*4000 # 聚合后的一秒一条数据，取【-1】
        # self.rolling_bar = 15
        # self.y_pred_side_list:np.ndarray = np.zeros(self.rolling_bar)   # 1st模型信号存储
        # print(len(self.y_pred_side_list))
        self.cta_engine.strategy_id = 11
        
        
        
        self.tick_dict_['closetime']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_price1']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_size1']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_price1']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_size1']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_price2']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_size2']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_price2']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_size2']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_price3']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_size3']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_price3']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_size3']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_price4']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_size4']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_price4']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_size4']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_price5']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_size5']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_price5']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_size5']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_price6']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_size6']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_price6']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_size6']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_price7']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_size7']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_price7']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_size7']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_price8']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_size8']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_price8']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_size8']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_price9']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_size9']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_price9']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_size9']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_price10']:np.ndarray = np.zeros(2000)
        self.tick_dict_['ask_size10']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_price10']:np.ndarray = np.zeros(2000)
        self.tick_dict_['bid_size10']:np.ndarray = np.zeros(2000)
        self.tick_dict_['price']:np.ndarray = np.zeros(2000)
        self.tick_dict_['size']:np.ndarray = np.zeros(2000)
        self.tick_dict_['cum_size']:np.ndarray = np.zeros(2000)
        self.tick_dict_['turnover']:np.ndarray = np.zeros(2000)
        
        
        

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.log.info(f'{self.cta_engine.symbol}策略启动')
        self.put_event()
        
    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")
        
        
        
    def on_stop(self):
        """
        Callback when strategy is stopped.
        """

        self.write_log("策略停止")
        # if self.record:
        #     if "win" in sys.platform:
        #         base_path = os.path.abspath(".") + "\\result\\{}\\".format(
        #             self.cta_engine.symbol
        #         )
        #     else:
        #         base_path = os.path.abspath(".") + "/result/{}/".format(
        #             self.cta_engine.symbol
        #         )
        #     if os.path.exists(base_path) is False:
        #         os.mkdir(base_path)
        #     tm_year = self.cta_engine.start.year
        #     tm_month = self.cta_engine.start.month
        #     tm_day = self.cta_engine.start.day
        #     file_name = "{}_{}_{}_{}_{}_{}_{}_trades.csv".format(
        #         base_path,
        #         tm_year,
        #         tm_month,
        #         tm_day,
        #         self.cta_engine.end.year,
        #         self.cta_engine.end.month,
        #         self.cta_engine.end.day,
        #     )
        #     if self.trades:
        #         df = pd.DataFrame(self.trades)
        #         df.to_csv(
        #             file_name,
        #             index=False,
        #             columns=[
        #                 "price",
        #                 "size",
        #                 "direction",
        #                 "o_id",
        #                 "t_id",
        #                 "datetime",
        #                 "pos",
        #                 "closetime",
        #             ],
        #         )

        self.put_event()

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """

        # tick data to kline data
        self.bg.update_tick(tick)
        # self.put_event()
        # print(tick.closetime)
        # print(len(self.cta_engine.active_limit_orders))
        self.tick_dict_['closetime'][:-1] = self.tick_dict_['closetime'][1:]
        self.tick_dict_['closetime'][-1] = tick.closetime
        self.tick_dict_['ask_price1'][:-1] = self.tick_dict_['ask_price1'][1:]
        self.tick_dict_['ask_price1'][-1] = tick.ask_price_1
        self.tick_dict_['ask_size1'][:-1] = self.tick_dict_['ask_size1'][1:]
        self.tick_dict_['ask_size1'][-1] =  tick.ask_volume_1
        self.tick_dict_['bid_price1'][:-1] = self.tick_dict_['bid_price1'][1:]
        self.tick_dict_['bid_price1'][-1] = tick.bid_price_1
        self.tick_dict_['bid_size1'][:-1] = self.tick_dict_['bid_size1'][1:]
        self.tick_dict_['bid_size1'][-1] = tick.bid_volume_1
        self.tick_dict_['ask_price2'][:-1] = self.tick_dict_['ask_price2'][1:]
        self.tick_dict_['ask_price2'][-1] = tick.ask_price_2
        self.tick_dict_['ask_size2'][:-1] = self.tick_dict_['ask_size2'][1:]
        self.tick_dict_['ask_size2'][-1] = tick.ask_volume_2
        self.tick_dict_['bid_price2'][:-1] = self.tick_dict_['bid_price2'][1:]
        self.tick_dict_['bid_price2'][-1] = tick.bid_price_2
        self.tick_dict_['bid_size2'][:-1] = self.tick_dict_['bid_size2'][1:]
        self.tick_dict_['bid_size2'][-1] =  tick.bid_volume_2
        self.tick_dict_['ask_price3'][:-1] = self.tick_dict_['ask_price3'][1:]
        self.tick_dict_['ask_price3'][-1] = tick.ask_price_3
        self.tick_dict_['ask_size3'][:-1] = self.tick_dict_['ask_size3'][1:]
        self.tick_dict_['ask_size3'][-1] = tick.ask_volume_3
        self.tick_dict_['bid_price3'][:-1] = self.tick_dict_['bid_price3'][1:]
        self.tick_dict_['bid_price3'][-1] = tick.bid_price_3
        self.tick_dict_['bid_size3'][:-1] = self.tick_dict_['bid_size3'][1:]
        self.tick_dict_['bid_size3'][-1] = tick.bid_volume_3
        self.tick_dict_['ask_price4'][:-1] = self.tick_dict_['ask_price4'][1:]
        self.tick_dict_['ask_price4'][-1] = tick.ask_price_4
        self.tick_dict_['ask_size4'][:-1] = self.tick_dict_['ask_size4'][1:]
        self.tick_dict_['ask_size4'][-1] = tick.ask_volume_4
        self.tick_dict_['bid_price4'][:-1] = self.tick_dict_['bid_price4'][1:]
        self.tick_dict_['bid_price4'][-1] = tick.bid_price_4
        self.tick_dict_['bid_size4'][:-1] = self.tick_dict_['bid_size4'][1:]
        self.tick_dict_['bid_size4'][-1] = tick.bid_volume_4
        self.tick_dict_['ask_price5'][:-1] = self.tick_dict_['ask_price5'][1:]
        self.tick_dict_['ask_price5'][-1] = tick.ask_price_5
        self.tick_dict_['ask_size5'][:-1] = self.tick_dict_['ask_size5'][1:]
        self.tick_dict_['ask_size5'][-1] = tick.ask_volume_5
        self.tick_dict_['bid_price5'][:-1] = self.tick_dict_['bid_price5'][1:]
        self.tick_dict_['bid_price5'][-1] = tick.bid_price_5
        self.tick_dict_['bid_size5'][:-1] = self.tick_dict_['bid_size5'][1:]
        self.tick_dict_['bid_size5'][-1] = tick.bid_volume_5
        self.tick_dict_['ask_price6'][:-1] = self.tick_dict_['ask_price6'][1:]
        self.tick_dict_['ask_price6'][-1] = tick.ask_price_6
        self.tick_dict_['ask_size6'][:-1] = self.tick_dict_['ask_size6'][1:]
        self.tick_dict_['ask_size6'][-1] = tick.ask_volume_6
        self.tick_dict_['bid_price6'][:-1] = self.tick_dict_['bid_price6'][1:]
        self.tick_dict_['bid_price6'][-1] = tick.bid_price_6
        self.tick_dict_['bid_size6'][:-1] = self.tick_dict_['bid_size6'][1:]
        self.tick_dict_['bid_size6'][-1] = tick.bid_volume_6
        self.tick_dict_['ask_price7'][:-1] = self.tick_dict_['ask_price7'][1:]
        self.tick_dict_['ask_price7'][-1] = tick.ask_price_7
        self.tick_dict_['ask_size7'][:-1] = self.tick_dict_['ask_size7'][1:]
        self.tick_dict_['ask_size7'][-1] = tick.ask_volume_7
        self.tick_dict_['bid_price7'][:-1] = self.tick_dict_['bid_price7'][1:]
        self.tick_dict_['bid_price7'][-1] = tick.bid_price_7
        self.tick_dict_['bid_size7'][:-1] = self.tick_dict_['bid_size7'][1:]
        self.tick_dict_['bid_size7'][-1] = tick.bid_volume_7
        self.tick_dict_['ask_price8'][:-1] = self.tick_dict_['ask_price8'][1:]
        self.tick_dict_['ask_price8'][-1] = tick.ask_price_8
        self.tick_dict_['ask_size8'][:-1] = self.tick_dict_['ask_size8'][1:]
        self.tick_dict_['ask_size8'][-1] = tick.ask_volume_8
        self.tick_dict_['bid_price8'][:-1] = self.tick_dict_['bid_price8'][1:]
        self.tick_dict_['bid_price8'][-1] = tick.bid_price_8
        self.tick_dict_['bid_size8'][:-1] = self.tick_dict_['bid_size8'][1:]
        self.tick_dict_['bid_size8'][-1] = tick.bid_volume_8
        self.tick_dict_['ask_price9'][:-1] = self.tick_dict_['ask_price9'][1:]
        self.tick_dict_['ask_price9'][-1] = tick.ask_price_9
        self.tick_dict_['ask_size9'][:-1] = self.tick_dict_['ask_size9'][1:]
        self.tick_dict_['ask_size9'][-1] = tick.ask_volume_9
        self.tick_dict_['bid_price9'][:-1] = self.tick_dict_['bid_price9'][1:]
        self.tick_dict_['bid_price9'][-1] = tick.bid_price_9
        self.tick_dict_['bid_size9'][:-1] = self.tick_dict_['bid_size9'][1:]
        self.tick_dict_['bid_size9'][-1] = tick.bid_volume_9
        self.tick_dict_['ask_price10'][:-1] = self.tick_dict_['ask_price10'][1:]
        self.tick_dict_['ask_price10'][-1] = tick.ask_price_10
        self.tick_dict_['ask_size10'][:-1] = self.tick_dict_['ask_size10'][1:]
        self.tick_dict_['ask_size10'][-1] =  tick.ask_volume_10
        self.tick_dict_['bid_price10'][:-1] = self.tick_dict_['bid_price10'][1:]
        self.tick_dict_['bid_price10'][-1] = tick.bid_price_10
        self.tick_dict_['bid_size10'][:-1] = self.tick_dict_['bid_size10'][1:]
        self.tick_dict_['bid_size10'][-1] = tick.bid_volume_10
        self.tick_dict_['price'][:-1] = self.tick_dict_['price'][1:]
        self.tick_dict_['price'][-1] = tick.last_price
        self.tick_dict_['size'][:-1] = self.tick_dict_['size'][1:]
        self.tick_dict_['size'][-1] = tick.last_volume
        last_volume = np.where(np.isnan(tick.last_volume), 0, tick.last_volume)
        last_price = np.where(np.isnan(tick.last_price), 0, tick.last_price)
        self.cum_size += abs(last_volume)
        self.turnover += (abs(last_volume)) * last_price
        self.tick_dict_['cum_size'][:-1] = self.tick_dict_['cum_size'][1:]
        self.tick_dict_['cum_size'][-1] = self.cum_size
        self.tick_dict_['turnover'][:-1] = self.tick_dict_['turnover'][1:]
        self.tick_dict_['turnover'][-1] = self.turnover
        if len(self.tick_dict_['cum_size']) > 2:
            self.tick_dict_['cum_size'][-1] = np.where(np.isnan(self.tick_dict_['size'][-1]), self.tick_dict_['cum_size'][-2],
                                                       self.tick_dict_['cum_size'][-1])
            self.tick_dict_['turnover'][-1] = np.where(np.isnan(self.tick_dict_['size'][-1]), self.tick_dict_['turnover'][-2],
                                                       self.tick_dict_['turnover'][-1])
        if len(self.tick_dict_['closetime']) > 1:
            if time.localtime(self.tick_dict_['closetime'][-1] / 1000).tm_mday != time.localtime(
                    self.tick_dict_['closetime'][-2] / 1000).tm_mday:
                self.cum_size = 0
                self.tick_dict_['cum_size'][:-1] = self.tick_dict_['cum_size'][1:]
                self.tick_dict_['cum_size'][-1] = self.cum_size
                self.turnover = 0
                self.tick_dict_['turnover'][:-1] = self.tick_dict_['turnover'][1:]
                self.tick_dict_['turnover'][-1] = self.turnover
            if time.localtime(self.tick_dict_['closetime'][-1] / 1000).tm_sec != time.localtime(
                    self.tick_dict_['closetime'][-2] / 1000).tm_sec:
                tick_dict = np.hstack((self.tick_dict_['closetime'][-2],
                                       self.tick_dict_['ask_price1'][-2], self.tick_dict_['ask_size1'][-2],
                                       self.tick_dict_['bid_price1'][-2], self.tick_dict_['bid_size1'][-2],
                                       self.tick_dict_['ask_price2'][-2], self.tick_dict_['ask_size2'][-2],
                                       self.tick_dict_['bid_price2'][-2], self.tick_dict_['bid_size2'][-2],
                                       self.tick_dict_['ask_price3'][-2], self.tick_dict_['ask_size3'][-2],
                                       self.tick_dict_['bid_price3'][-2], self.tick_dict_['bid_size1'][-2],
                                       self.tick_dict_['ask_price4'][-2], self.tick_dict_['ask_size4'][-2],
                                       self.tick_dict_['bid_price4'][-2], self.tick_dict_['bid_size4'][-2],
                                       self.tick_dict_['ask_price5'][-2], self.tick_dict_['ask_size5'][-2],
                                       self.tick_dict_['bid_price5'][-2], self.tick_dict_['bid_size5'][-2],
                                       self.tick_dict_['ask_price6'][-2], self.tick_dict_['ask_size6'][-2],
                                       self.tick_dict_['bid_price6'][-2], self.tick_dict_['bid_size6'][-2],
                                       self.tick_dict_['ask_price7'][-2], self.tick_dict_['ask_size7'][-2],
                                       self.tick_dict_['bid_price7'][-2], self.tick_dict_['bid_size7'][-2],
                                       self.tick_dict_['ask_price8'][-2], self.tick_dict_['ask_size8'][-2],
                                       self.tick_dict_['bid_price8'][-2], self.tick_dict_['bid_size8'][-2],
                                       self.tick_dict_['ask_price9'][-2], self.tick_dict_['ask_size9'][-2],
                                       self.tick_dict_['bid_price9'][-2], self.tick_dict_['bid_size9'][-2],
                                       self.tick_dict_['ask_price10'][-2], self.tick_dict_['ask_size10'][-2],
                                       self.tick_dict_['bid_price10'][-2], self.tick_dict_['bid_size10'][-2],
                                       self.tick_dict_['price'][-2], self.tick_dict_['size'][-2],
                                       self.tick_dict_['cum_size'][-2], self.tick_dict_['turnover'][-2],))
                # print(tick_dict)
                if np.isnan(tick_dict).any():
                    global len_
                    len_tick = len(self.tick_dict_['ask_price1'][:-1]) - len_ + 1

                    ap1_ = self.tick_dict_['ask_price1'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_price1'][len_tick:-1])],
                    av1_ = self.tick_dict_['ask_size1'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_size1'][len_tick:-1])],
                    bp1_ = self.tick_dict_['bid_price1'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_price1'][len_tick:-1])],
                    bv1_ = self.tick_dict_['bid_size1'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_size1'][len_tick:-1])],
                    ap2_ = self.tick_dict_['ask_price2'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_price2'][len_tick:-1])],
                    av2_ = self.tick_dict_['ask_size2'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_size2'][len_tick:-1])],
                    bp2_ = self.tick_dict_['bid_price2'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_price2'][len_tick:-1])],
                    bv2_ = self.tick_dict_['bid_size2'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_size2'][len_tick:-1])],
                    ap3_ = self.tick_dict_['ask_price3'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_price3'][len_tick:-1])],
                    av3_ = self.tick_dict_['ask_size3'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_size3'][len_tick:-1])],
                    bp3_ = self.tick_dict_['bid_price3'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_price3'][len_tick:-1])],
                    bv3_ = self.tick_dict_['bid_size3'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_size3'][len_tick:-1])],
                    ap4_ = self.tick_dict_['ask_price4'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_price4'][len_tick:-1])],
                    av4_ = self.tick_dict_['ask_size4'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_size4'][len_tick:-1])],
                    bp4_ = self.tick_dict_['bid_price4'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_price4'][len_tick:-1])],
                    bv4_ = self.tick_dict_['bid_size4'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_size4'][len_tick:-1])],
                    ap5_ = self.tick_dict_['ask_price5'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_price5'][len_tick:-1])],
                    av5_ = self.tick_dict_['ask_size5'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_size5'][len_tick:-1])],
                    bp5_ = self.tick_dict_['bid_price5'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_price5'][len_tick:-1])],
                    bv5_ = self.tick_dict_['bid_size5'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_size5'][len_tick:-1])],
                    ap6_ = self.tick_dict_['ask_price6'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_price6'][len_tick:-1])],
                    av6_ = self.tick_dict_['ask_size6'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_size6'][len_tick:-1])],
                    bp6_ = self.tick_dict_['bid_price6'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_price6'][len_tick:-1])],
                    bv6_ = self.tick_dict_['bid_size6'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_size6'][len_tick:-1])],
                    ap7_ = self.tick_dict_['ask_price7'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_price7'][len_tick:-1])],
                    av7_ = self.tick_dict_['ask_size7'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_size7'][len_tick:-1])],
                    bp7_ = self.tick_dict_['bid_price7'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_price7'][len_tick:-1])],
                    bv7_ = self.tick_dict_['bid_size7'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_size7'][len_tick:-1])],
                    ap8_ = self.tick_dict_['ask_price8'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_price8'][len_tick:-1])],
                    av8_ = self.tick_dict_['ask_size8'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_size8'][len_tick:-1])],
                    bp8_ = self.tick_dict_['bid_price8'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_price8'][len_tick:-1])],
                    bv8_ = self.tick_dict_['bid_size8'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_size8'][len_tick:-1])],
                    ap9_ = self.tick_dict_['ask_price9'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_price9'][len_tick:-1])],
                    av9_ = self.tick_dict_['ask_size9'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['ask_size9'][len_tick:-1])],
                    bp9_ = self.tick_dict_['bid_price9'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_price9'][len_tick:-1])],
                    bv9_ = self.tick_dict_['bid_size9'][len_tick:-1][
                               ~np.isnan(self.tick_dict_['bid_size9'][len_tick:-1])],
                    ap10_ = self.tick_dict_['ask_price10'][len_tick:-1][
                                ~np.isnan(self.tick_dict_['ask_price10'][len_tick:-1])],
                    av10_ = self.tick_dict_['ask_size10'][len_tick:-1][
                                ~np.isnan(self.tick_dict_['ask_size10'][len_tick:-1])],
                    bp10_ = self.tick_dict_['bid_price10'][len_tick:-1][
                                ~np.isnan(self.tick_dict_['bid_price10'][len_tick:-1])],
                    bv10_ = self.tick_dict_['bid_size10'][len_tick:-1][
                                ~np.isnan(self.tick_dict_['bid_size10'][len_tick:-1])],
                    p_ = self.tick_dict_['price'][len_tick:-1][~np.isnan(self.tick_dict_['price'][len_tick:-1])],
                    v_ = self.tick_dict_['size'][len_tick:-1][~np.isnan(self.tick_dict_['size'][len_tick:-1])],
                    cv_ = self.tick_dict_['cum_size'][len_tick:-1][~np.isnan(self.tick_dict_['cum_size'][len_tick:-1])],
                    a_ = self.tick_dict_['turnover'][len_tick:-1][~np.isnan(self.tick_dict_['turnover'][len_tick:-1])]
                    # print('len_turnover_',len(a_))
                    # print('turnover_',[a_[-1][-1] if len(a_[-1])>0 else np.nan][-1])
                    tick_ = np.hstack((self.tick_dict_['closetime'][-2],
                                       [ap1_[-1][-1] if len(ap1_[-1]) > 0 else np.nan][-1],
                                       [av1_[-1][-1] if len(av1_[-1]) > 0 else np.nan][-1],
                                       [bp1_[-1][-1] if len(bp1_[-1]) > 0 else np.nan][-1],
                                       [bv1_[-1][-1] if len(bv1_[-1]) > 0 else np.nan][-1],
                                       [ap2_[-1][-1] if len(ap2_[-1]) > 0 else np.nan][-1],
                                       [av2_[-1][-1] if len(av2_[-1]) > 0 else np.nan][-1],
                                       [bp2_[-1][-1] if len(bp2_[-1]) > 0 else np.nan][-1],
                                       [bv2_[-1][-1] if len(bv2_[-1]) > 0 else np.nan][-1],
                                       [ap3_[-1][-1] if len(ap3_[-1]) > 0 else np.nan][-1],
                                       [av3_[-1][-1] if len(av3_[-1]) > 0 else np.nan][-1],
                                       [bp3_[-1][-1] if len(bp3_[-1]) > 0 else np.nan][-1],
                                       [bv3_[-1][-1] if len(bv3_[-1]) > 0 else np.nan][-1],
                                       [ap4_[-1][-1] if len(ap4_[-1]) > 0 else np.nan][-1],
                                       [av4_[-1][-1] if len(av4_[-1]) > 0 else np.nan][-1],
                                       [bp4_[-1][-1] if len(bp4_[-1]) > 0 else np.nan][-1],
                                       [bv4_[-1][-1] if len(bv4_[-1]) > 0 else np.nan][-1],
                                       [ap5_[-1][-1] if len(ap5_[-1]) > 0 else np.nan][-1],
                                       [av5_[-1][-1] if len(av5_[-1]) > 0 else np.nan][-1],
                                       [bp5_[-1][-1] if len(bp5_[-1]) > 0 else np.nan][-1],
                                       [bv5_[-1][-1] if len(bv5_[-1]) > 0 else np.nan][-1],
                                       [ap6_[-1][-1] if len(ap6_[-1]) > 0 else np.nan][-1],
                                       [av6_[-1][-1] if len(av6_[-1]) > 0 else np.nan][-1],
                                       [bp6_[-1][-1] if len(bp6_[-1]) > 0 else np.nan][-1],
                                       [bv6_[-1][-1] if len(bv6_[-1]) > 0 else np.nan][-1],
                                       [ap7_[-1][-1] if len(ap7_[-1]) > 0 else np.nan][-1],
                                       [av7_[-1][-1] if len(av7_[-1]) > 0 else np.nan][-1],
                                       [bp7_[-1][-1] if len(bp7_[-1]) > 0 else np.nan][-1],
                                       [bv7_[-1][-1] if len(bv7_[-1]) > 0 else np.nan][-1],
                                       [ap8_[-1][-1] if len(ap8_[-1]) > 0 else np.nan][-1],
                                       [av8_[-1][-1] if len(av8_[-1]) > 0 else np.nan][-1],
                                       [bp8_[-1][-1] if len(bp8_[-1]) > 0 else np.nan][-1],
                                       [bv8_[-1][-1] if len(bv8_[-1]) > 0 else np.nan][-1],
                                       [ap9_[-1][-1] if len(ap9_[-1]) > 0 else np.nan][-1],
                                       [av9_[-1][-1] if len(av9_[-1]) > 0 else np.nan][-1],
                                       [bp9_[-1][-1] if len(bp9_[-1]) > 0 else np.nan][-1],
                                       [bv9_[-1][-1] if len(bv9_[-1]) > 0 else np.nan][-1],
                                       [ap10_[-1][-1] if len(ap10_[-1]) > 0 else np.nan][-1],
                                       [av10_[-1][-1] if len(av10_[-1]) > 0 else np.nan][-1],
                                       [bp10_[-1][-1] if len(bp10_[-1]) > 0 else np.nan][-1],
                                       [bv10_[-1][-1] if len(bv10_[-1]) > 0 else np.nan][-1],
                                       [p_[-1][-1] if len(p_[-1]) > 0 else np.nan][-1],
                                       [v_[-1][-1] if len(v_[-1]) > 0 else np.nan][-1],
                                       [cv_[-1][-1] if len(cv_[-1]) > 0 else np.nan][-1],
                                       [a_[-1] if len(a_) > 0 else np.nan][-1]))
                    # print('tick_:',tick_, '时间:',tick.datetime)
                    self.tick_1s.append(tick_)
                    # self.tick_1s[:-1] = self.tick_1s[1:]
                    # self.tick_1s[-1] = tick_
                    self.i = []

                else:
                    self.tick_1s.append(tick_dict)
                    # self.tick_1s[:-1] = self.tick_1s[1:]
                    # self.tick_1s[-1] = tick_dict
                    self.i = []
                # print('closetime:',time.localtime(self.tick_1s[-1][0]/1000).tm_hour,time.localtime(self.tick_1s[-1][0]/1000).tm_min,time.localtime(self.tick_1s[-1][0]/1000).tm_sec)
                # print('price:',self.tick_1s[-1][-4])
                # print('size:',self.tick_1s[-1][-3])
                # else:
                #     self.i.append(self.tick_dict_['closetime'][-1])
                #     len_ = len(self.i)
                # print('1s---:',self.tick_1s[-1])
                sum_num = 0
                # t1 = time.time()
                self.tick_1s = self.tick_1s[-3000:]
                # factor = factor_calculation(self.tick_1s[-1], self.feat_dict_, sum_num, self.df_2d, self.col_dict_, )
                tick1s = pd.DataFrame(self.tick_1s)
                tick1s.columns = tick_data_list
                tick1s['datetime'] = pd.to_datetime(tick1s['closetime'] + 28800000, unit='ms')
                tick1s_ = tick1s.set_index('datetime').groupby(pd.Grouper(freq='1000ms')).apply('last')
                tick1s_ = tick1s_.dropna(subset=['ask_price1'])
                # print('因子计算时间:',time.time() - t1, self.model_symbol)
                
                if time.time() - self.strategy_time > 600:
                    print('每十分钟打印一次阈值:',self.tick_1s[-1][-1] - self.tick_1s[-2][-1],'时间:',tick.datetime, self.model_symbol)
                    self.strategy_time = time.time()
                time_1s = tick.closetime/1000
                # print(len(np.isnan(self.tick_1s)))
                if time_1s - self.factor_time > 2500:
                    tick1s_['vwap'] = (tick1s_['price'].fillna(0) * abs(tick1s_['size'].fillna(0))).rolling(120).sum() / abs(tick1s_['size'].fillna(0)).rolling(120).sum()
                    trade = tick1s_.loc[:, ['closetime', 'price', 'size', 'cum_size', 'turnover']]
                    depth = tick1s_.loc[:,['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1', 'ask_price2', 'ask_size2', 'bid_price2',
                                         'bid_size2', 'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4', 'bid_price4',
                                         'bid_size4', 'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5', 'ask_price6', 'ask_size6', 'bid_price6', 'bid_size6',
                                         'ask_price7', 'ask_size7', 'bid_price7', 'bid_size7', 'ask_price8', 'ask_size8', 'bid_price8', 'bid_size8',
                                         'ask_price9', 'ask_size9', 'bid_price9', 'bid_size9', 'ask_price10', 'ask_size10', 'bid_price10', 'bid_size10']]
                    add_factor = add_factor_process(depth=depth, trade=trade)
                    if self.tick_1s[-1][-1] - self.tick_1s[-2][-1] >= self.threshold:
                        print('bar采样触发阈值时间:', tick.datetime, '品种:',self.model_symbol)
                        signal = add_factor.iloc[-1:,:]
                        X_test_ = np.array(signal.iloc[:,45:102]).reshape(1, -1)
                        
                        # X_test = (factor[-1][45:102]).reshape(1, -1)
                        # print('因子值:',X_test)
                        # msg_ = f'因子值:{X_test}--批式因子值:{X_test_}---time:{tick.datetime}---symbol:{self.cta_engine.symbol}'
                        # self.log.info(msg_)
                        # y_pred_side_0 = self.model_side_0.predict(X_test, num_iteration=self.model_side_0.best_iteration)
                        # y_pred_side_1 = self.model_side_1.predict(X_test, num_iteration=self.model_side_1.best_iteration)
                        # y_pred_side_2 = self.model_side_2.predict(X_test, num_iteration=self.model_side_2.best_iteration)
                        # y_pred_side_3 = self.model_side_3.predict(X_test, num_iteration=self.model_side_3.best_iteration)
                        # y_pred_side_4 = self.model_side_4.predict(X_test, num_iteration=self.model_side_4.best_iteration)
                        # y_pred_side = (y_pred_side_0[0] + y_pred_side_1[0] + y_pred_side_2[0] + y_pred_side_3[0] +
                        #                y_pred_side_4[0]) / 5
                        # self.y_pred_side_list.append([y_pred_side])
                        
                        y_pred_side_0_ = self.model_side_0.predict(X_test_, num_iteration=self.model_side_0.best_iteration)
                        y_pred_side_1_ = self.model_side_1.predict(X_test_, num_iteration=self.model_side_1.best_iteration)
                        y_pred_side_2_ = self.model_side_2.predict(X_test_, num_iteration=self.model_side_2.best_iteration)
                        y_pred_side_3_ = self.model_side_3.predict(X_test_, num_iteration=self.model_side_3.best_iteration)
                        y_pred_side_4_ = self.model_side_4.predict(X_test_, num_iteration=self.model_side_4.best_iteration)
                        y_pred_side_ = (y_pred_side_0_[0] + y_pred_side_1_[0] + y_pred_side_2_[0] + y_pred_side_3_[0] +
                                       y_pred_side_4_[0]) / 5
                        self.y_pred_side_list_.append([y_pred_side_])
                        msg_ = f'批式方向信号:{self.y_pred_side_list_[-1]}--time:{tick.datetime}---symbol:{self.cta_engine.symbol}'
                        self.log.info(msg_)
               
                        y_pred_side_df = pd.DataFrame(self.y_pred_side_list_, columns=['predict'])
                                                   
                        if y_pred_side_df['predict'].iloc[-1] > self.side_long or y_pred_side_df['predict'].iloc[-1] < self.side_short:
                            y_pred_out_0 = self.model_out_0.predict(X_test_, num_iteration=self.model_out_0.best_iteration)
                            y_pred_out_1 = self.model_out_1.predict(X_test_, num_iteration=self.model_out_1.best_iteration)
                            y_pred_out_2 = self.model_out_2.predict(X_test_, num_iteration=self.model_out_2.best_iteration)
                            y_pred_out_3 = self.model_out_3.predict(X_test_, num_iteration=self.model_out_3.best_iteration)
                            y_pred_out_4 = self.model_out_4.predict(X_test_, num_iteration=self.model_out_4.best_iteration)
                            y_pred_out = (y_pred_out_0[0] + y_pred_out_1[0] + y_pred_out_2[0] + y_pred_out_3[0] +
                                          y_pred_out_4[0]) / 5
                            self.y_pred_out_list.append([y_pred_out])
                            y_pred_out_df = pd.DataFrame(self.y_pred_out_list, columns=['out'])
                            msg_ = f'入场信号:{self.y_pred_out_list[-1]}-----time:{tick.datetime}---symbol:{self.cta_engine.symbol}'
                            self.log.info(msg_)

                            # 策略逻辑
                            # price = factor[-1][-1]
                            price = tick1s_['vwap'].iloc[-1]
                            position_value = self.pos * price
                            place_value = self.cta_engine.capital * self.pos_rate / self.split_count
                            buy_size = round(place_value / tick.ask_price_1, 8)
                            sell_size = round(place_value / tick.bid_price_1, 8)
                            max_limited_order_value = self.cta_engine.capital * self.pos_rate

                            # 计算挂单金额
                            limit_orders_values = 0
                            final_values = 0
                            for key in self.cta_engine.active_limit_orders.keys():
                                limit_orders_values += float(self.cta_engine.active_limit_orders[key].price)*abs(float(self.cta_engine.active_limit_orders[key].volume))
                            # 持仓金额+挂单金额
                            final_values = limit_orders_values+self.cta_engine.ca_balance_now['price']*abs(self.pos)
                            
                            # 平多仓
                            if float(y_pred_side_df['predict'].iloc[-1]) < self.side_short and float(
                                    y_pred_out_df['out'].iloc[-1]) > self.out and self.pos > 0:
                                print('-------------平仓之前撤销所有订单-------------')
                                self.cancel_all(closetime=tick.closetime / 1000)
                                print('---------------------------下空单平多仓-----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                                self.sell(price=tick.bid_price_3, volume=self.pos,  # stop=True,
                                          net=True, closetime=tick.closetime / 1000)
                                if self.cta_engine.engine_type == EngineType.REAL:
                                    dingmessage(webhook_key=WebHook.songhe_ai_crypto.value,
                                                msg=f'下空单平多仓, 平仓价格:{tick.bid_price_3} size:{self.pos} time:{tick.datetime} symbol:{self.cta_engine.symbol}')
                                    msg_ = f'下空单平多仓---平仓价格:{tick.bid_price_3}---size:{self.pos}---time:{tick.datetime}---symbol:{self.cta_engine.symbol}'
                                    self.log.info(msg_)

                            # 平空仓
                            if float(y_pred_side_df['predict'].iloc[-1]) > self.side_long and float(
                                    y_pred_out_df['out'].iloc[-1]) > self.out and self.pos < 0:
                                print('-------------平仓之前撤销所有订单-------------')
                                self.cancel_all(closetime=tick.closetime / 1000)
                                print('-----------------------------下多单平空仓----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                                self.buy(price=tick.ask_price_3, volume=-self.pos,  # stop=True,
                                         net=True, closetime=tick.closetime / 1000)
                                if self.cta_engine.engine_type == EngineType.REAL:
                                    dingmessage(webhook_key=WebHook.songhe_ai_crypto.value,
                                                msg=f'下多单平空仓, 平仓价格:{tick.ask_price_3} size:{self.pos} time:{tick.datetime} symbol:{self.cta_engine.symbol}')
                                    msg_ = f'下多单平空仓---平仓价格:{tick.ask_price_3}---size:{self.pos}---time:{tick.datetime}---symbol:{self.cta_engine.symbol}'
                                    self.log.info(msg_)

                            # 开空仓
                            if float(y_pred_side_df['predict'].iloc[-1]) < self.side_short and float(
                                    y_pred_out_df['out'].iloc[-1]) > self.out and position_value >= -self.pos_rate * self.cta_engine.capital * (
                                    1 - 1 / self.split_count):
                                if len(self.cta_engine.active_limit_orders)>0:
                                    last_order = list(self.cta_engine.active_limit_orders.keys())[-1]
                                    if self.cta_engine.active_limit_orders[last_order].volume>0:
                                        self.cancel_all(closetime=bar.closetime / 1000)
                                print('--------------开空仓----------------', '品种:',self.model_symbol)
                                if max_limited_order_value <= final_values*1.00001:
                                    self.cancel_all(closetime=bar.closetime / 1000)
                                self.sell(price=price * (1 - self.place_rate), volume=sell_size,  # stop=True,
                                          net=True, closetime=tick.closetime / 1000)
                                if self.cta_engine.engine_type == EngineType.REAL:
                                    dingmessage(webhook_key=WebHook.songhe_ai_crypto.value,
                                                msg=f'开空仓, 开仓价格:{price * (1 - self.place_rate)} size:{sell_size} time:{tick.datetime} symbol:{self.cta_engine.symbol}')
                                    msg_ = f'开空仓---开仓价格:{price * (1 - self.place_rate)}---size:{sell_size}---time:{tick.datetime}---symbol:{self.cta_engine.symbol}'
                                    self.log.info(msg_)
                                # self.fill_order_time = tick.closetime
                            # 开多仓
                            if float(y_pred_side_df['predict'].iloc[-1]) > self.side_long and float(
                                   y_pred_out_df['out'].iloc[-1]) > self.out and position_value <= self.pos_rate * self.cta_engine.capital * (
                                    1 - 1 / self.split_count):
                                # 如果此时有挂多单，全部撤掉
                                if len(self.cta_engine.active_limit_orders)>0:
                                    last_order = list(self.cta_engine.active_limit_orders.keys())[-1]
                                    if self.cta_engine.active_limit_orders[last_order].volume>0:
                                        self.cancel_all(closetime=bar.closetime / 1000)
                                print('--------------开多仓----------------', '品种:',self.model_symbol)
                                if max_limited_order_value <= final_values*1.00001:
                                    self.cancel_all(closetime=bar.closetime / 1000)
                                self.buy(price=price * (1 + self.place_rate), volume=buy_size,  # stop=True,
                                         net=True, closetime=tick.closetime / 1000)
                                if self.cta_engine.engine_type == EngineType.REAL:
                                    dingmessage(webhook_key=WebHook.songhe_ai_crypto.value,
                                                msg=f'开多仓, 开仓价格:{price * (1 + self.place_rate)} size:{buy_size} time:{tick.datetime} symbol:{self.cta_engine.symbol}')
                                    msg_ = f'开多仓---开仓价格:{price * (1 + self.place_rate)}---size:{buy_size}---time:{tick.datetime}---symbol:{self.cta_engine.symbol}'
                                    self.log.info(msg_)
                                # self.fill_order_time = tick.closetime

                                # return





                            #保持仓位
                            else:
                                return

            else:
                self.i.append(self.tick_dict_['closetime'][-1])
                len_ = len(self.i)
                
        msg = '当前时间:{}\r\n' \
              '品种:{}\r\n' \
              '上一分钟收盘价:{}\r\n' \
              '账户净值:{:.2f}\r\n' \
              '持仓数量:{:.8f}\r\n' \
              '持仓价值:{:.2f}\r\n' \
              '上一次持仓信息更新时间:{}\r\n' \
              '上一次账户信息更新时间:{}\r\n' \
              '当前挂单:{}' \
            .format(tick.datetime,
                    self.cta_engine.symbol,
                    (tick.ask_price_1 + tick.bid_price_1)/2,
                    self.cta_engine.total_balance + self.cta_engine.total_unrealized_pnl, self.pos,
                    self.pos * ((tick.ask_price_1 + tick.bid_price_1) / 2),
                    datetime.datetime.fromtimestamp(self.cta_engine.position_time).strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.datetime.fromtimestamp(self.cta_engine.account_time).strftime('%Y-%m-%d %H:%M:%S'),
                    self.cta_engine.active_limit_orders,
                    )

        # 每30分钟做钉钉推送 只有实盘且有持仓或者有挂单会输出
        if (tick.closetime / 1000) // 1800 > self.print_time and self.cta_engine.engine_type == EngineType.REAL:
            if time.time() - tick.closetime / 1000 < 120:
                self.print_time = (tick.closetime / 1000) // 1800
                dingmessage(webhook_key=WebHook.songhe_ai_crypto.value, msg=msg)
                
#         snapshot2 = tracemalloc.take_snapshot()

#         top_stats = snapshot2.compare_to(snapshot1, 'lineno')

#         stat = top_stats[0]
#         print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
#         for line in stat.traceback.format():
#             print(line)
        
        
    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        # 1m kline data to 1d kline data
        # self.bg1d.update_bar(bar)
        # print('均价的价格:',bar.close)
        # 策略逻辑
        # 每五分钟判断一次
        if int(bar.closetime/1000) - int(self.kill_time/1000) > 60: 
            #多仓止盈止损
            if self.pos > 0:
                # print('-------------平仓之前撤销所有订单-------------', '品种:',self.model_symbol)
                # self.cancel_all(closetime=bar.closetime / 1000)
                pf = float(bar.close/self.cta_engine.strategy.price)-1
                con1 = 0
                if pf > 0.05:
                    con1 = 1
                    print('-------------多头止盈离场-------------', '品种:',self.model_symbol)
                    if self.cta_engine.engine_type == EngineType.REAL:
                        dingmessage(webhook_key=WebHook.songhe_ai_crypto.value,
                                msg=f'多头止盈离场, 平仓价格:{bar.close*(1+self.place_rate)} size:{self.pos} time:{bar.datetime} symbol:{self.cta_engine.symbol}')
                        msg_ = f'多头止盈离场---平仓价格:{bar.close*(1+self.place_rate)}---size:{self.pos}---time:{bar.datetime}---symbol:{self.cta_engine.symbol}'
                        self.log.info(msg_)
                elif pf <= -0.005:
                    con1 = 1
                    print('-------------多头止损离场-------------', '品种:',self.model_symbol)
                    if self.cta_engine.engine_type == EngineType.REAL:
                        dingmessage(webhook_key=WebHook.songhe_ai_crypto.value,
                                msg=f'多头止损离场, 平仓价格:{bar.close*(1+self.place_rate)} size:{self.pos} time:{bar.datetime} symbol:{self.cta_engine.symbol}')
                        msg_ = f'多头止损离场---平仓价格:{bar.close*(1+self.place_rate)}---size:{self.pos}---time:{bar.datetime}---symbol:{self.cta_engine.symbol}'
                        self.log.info(msg_)
                if con1 == 1:
                    # print('-------------离场时间-----------------',
                    # time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                    self.cancel_all(closetime=bar.closetime / 1000)
                    self.kill_time = bar.closetime/1000
                    self.sell(price=bar.close*(1+self.place_rate), volume=self.pos,  # stop=True,
                          net=True, closetime=bar.closetime / 1000)
            # 空仓止盈止损
            if self.pos < 0:
                # print('-------------平仓之前撤销所有订单-------------', '品种:',self.model_symbol)
                # self.cancel_all(closetime=bar.closetime / 1000)
                pf = 1- float(bar.close/self.cta_engine.strategy.price)
                con1 = 0
                if pf > 0.05:
                    con1 = 1
                    print('-------------空头止盈离场-------------', '品种:',self.model_symbol)
                    if self.cta_engine.engine_type == EngineType.REAL:
                        dingmessage(webhook_key=WebHook.songhe_ai_crypto.value,
                                msg=f'空头止盈离场, 平仓价格:{bar.close*(1-self.place_rate)} size:{self.pos} time:{bar.datetime} symbol:{self.cta_engine.symbol}')
                        msg_ = f'空头止盈离场---平仓价格:{bar.close*(1-self.place_rate)}---size:{self.pos}---time:{bar.datetime}---symbol:{self.cta_engine.symbol}'
                        self.log.info(msg_)
                elif pf <= -0.005:
                    con1 = 1
                    print('-------------空头止损离场-------------', '品种:',self.model_symbol)
                    if self.cta_engine.engine_type == EngineType.REAL:
                        dingmessage(webhook_key=WebHook.songhe_ai_crypto.value,
                                msg=f'空头止损离场, 平仓价格:{bar.close*(1-self.place_rate)} size:{self.pos} time:{bar.datetime} symbol:{self.cta_engine.symbol}')
                        msg_ = f'空头止损离场---平仓价格:{bar.close*(1-self.place_rate)}---size:{self.pos}---time:{bar.datetime}---symbol:{self.cta_engine.symbol}'
                        self.log.info(msg_)
                if con1 == 1:
                    # print('-------------离场时间-----------------',
                    # time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                    self.cancel_all(closetime=bar.closetime / 1000)
                    self.kill_time = bar.closetime/1000
                    self.buy(price=bar.close*(1-self.place_rate), volume=-self.pos,  # stop=True,
                          net=True, closetime=bar.closetime / 1000)
        else:
            return

            pass

    def on_1h_bar(self, bar: BarData):
        self.bg1d.update_bar(bar)
        # print(bar)

    def on_1d_bar(self, bar: BarData):
        # print('1d', bar)
        self.put_event()

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        self.put_event()

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        # 交易所成交记录
        self.cta_engine.output(
            'trades:{}'.format({"price": trade.price, "size": trade.volume, "direction": trade.direction,
                                "o_id": trade.orderid, "t_id": trade.tradeid, "datetime": trade.datetime,
                                "pos": self.pos, 'closetime': trade.datetime.timestamp() * 1000 - 1}))
        # 策略成交记录
        # if trade:
        #     self.strategy_trades.append({"price": trade.price, "size": trade.volume, "direction": trade.direction,
        #                                 "o_id": trade.orderid, "t_id": trade.tradeid, "datetime": trade.datetime,
        #                                 "pos": self.pos, 'closetime': trade.datetime.timestamp() * 1000 - 1})
        
        if self.cta_engine.engine_type == EngineType.REAL:
            price = self.price
        else:
            price = self.cta_engine.ca_balance_now['price']

        # 基本信息
        base = {'strategy_id': self.cta_engine.strategy_id,
                'engine_type': EngineType.BACKTESTING.value,
                'new_lag': False}

        # 账户信息
        account = {'balance': self.cta_engine.total_balance,
                   'net': self.cta_engine.total_balance + self.cta_engine.total_unrealized_pnl,
                   'available_balance': self.cta_engine.available_balance,
                   'unrealized_pnl': self.cta_engine.total_unrealized_pnl}

        account_data = base.copy()
        account_data['type'] = 'account'
        account_data['data'] = account
        print(self.send_msg(account_data))

        # 持仓信息
        positon = {'symbol': self.vt_symbol,
                   'price': price,
                   'pos': self.pos,
                   'pos_value': price * self.pos}

        positon_data = base.copy()
        positon_data['type'] = 'positon'
        positon_data['data'] = positon
        print(self.send_msg(positon_data))

        # 成交记录
        trade_dic = trade.__toDict__()
        # 账户净值
        trade_dic['net'] = self.cta_engine.total_balance + self.cta_engine.total_unrealized_pnl
        # 持仓
        trade_dic['position'] = self.pos
        # 持仓价值
        trade_dic['pos_value'] = price * self.pos
        # 交易对净值
        trade_dic['symbol_net'] = self.cta_engine.balance_now + self.cta_engine.unrealized_pnl
        # 成交时间戳13位
        trade_dic['closetime'] = 123546890123

        trade_data = base.copy()
        trade_data['type'] = 'trade'
        trade_data['data'] = trade_dic
        self.send_msg(trade_data)

        # 挂单列表
        open_orders = []
        for orderid in self.cta_engine.active_limit_orders.keys():
            open_orders.append(self.cta_engine.active_limit_orders[orderid].__toDict__())

        open_orders_data = base.copy()
        open_orders_data['type'] = 'open_orders'
        open_orders_data['data'] = open_orders
        self.send_msg(open_orders_data)
        
        

        msg = "\033[93m {} trades:{} \033[0m".format(
            self.cta_engine.symbol,
            {
                "price": trade.price,
                "size": trade.volume,
                "direction": trade.direction,
                "o_id": trade.orderid,
                "t_id": trade.tradeid,
                "datetime": trade.datetime,
                "pos": self.pos,
                'role': trade.role,
                'fee': trade.fee,
                'label': trade.label,
                "closetime": trade.datetime.timestamp() * 1000 - 1,
                'net': self.cta_engine.total_balance + self.cta_engine.total_unrealized_pnl
            }
        )
        if self.cta_engine.engine_type == EngineType.REAL:
            # print trades
            dingmessage(webhook_key=WebHook.songhe_ai_crypto.value, msg=msg)
            # every 300s print signal
            if time.time() - self.signal_time > 300:
                dingmessage(webhook_key=WebHook.songhe_ai_crypto.value, msg=self.single_result)
                self.signal_time = time.time()
            self.cta_engine.output(
                msg
            )
        if self.record:
            self.trades.append(
                msg
            )

        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        self.put_event()

    def test_market_order(self):
        """"""
        self.buy(self.last_tick.limit_up, 1)
        self.write_log("执行市价单测试")

    def test_limit_order(self):
        """"""
        self.buy(self.last_tick.limit_down, 1)
        self.write_log("执行限价单测试")

    def test_stop_order(self):
        """"""
        self.buy(self.last_tick.ask_price_1, 1, True)
        self.write_log("执行停止单测试")

    def test_cancel_all(self):
        """"""
        self.cancel_all()
        self.write_log("执行全部撤单测试")
