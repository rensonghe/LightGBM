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
from time import time


class TickStrategy(CtaTemplate):
    """"""
    author = "zq"

    test_trigger = 10

    tick_count = 0
    test_all_done = False

    parameters = ["test_trigger"]
    variables = ["tick_count", "test_all_done"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting, rolling_info=None):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting, rolling_info)
        self.last_tick = None
        self.bg = BarGenerator(self.on_bar)
        # self.bg1h = BarGenerator(self.on_bar, 60, self.on_1h_bar)
        self.bg1d = BarGenerator(self.on_bar, 24, self.on_1d_bar, interval=Interval.HOUR)

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

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

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        # tick data to kline data
        self.bg.update_tick(tick)
        self.put_event()

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        # 1m kline data to 1d kline data
        self.bg1d.update_bar(bar)
        # print(bar)
        pass

    def on_1h_bar(self, bar: BarData):
        self.bg1d.update_bar(bar)
        # print(bar)

    def on_1d_bar(self, bar: BarData):
        print('1d', bar)
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
