# -*- coding: utf-8 -*-
# zq
import ast
from datetime import datetime, timedelta
import os
import sys

BASE_DIR = os.path.abspath('../..')
sys.path.append(BASE_DIR)
from tzquant.trader.optimize import OptimizationSetting
from tz_ctastrategy.backtesting import BacktestingEngine
from tz_ctastrategy.strategies.tick_strategy import TickStrategy
from tz_ctastrategy.base import (
    BacktestingMode,
    DataType
)

if __name__ == '__main__':
    engine = BacktestingEngine()
    # 设置时间从今天的零点开始
    now_datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    engine.set_parameters(
        vt_symbol=f"{'btcusdt'}.{'binance_swap_u'}",
        start=now_datetime - timedelta(days=15),  # 形如：datetime(2022,1,1)
        end=now_datetime - timedelta(days=12),
        maker_fee=1.5 / 10000,  # 挂单手续费
        taker_fee=1.5 / 10000,  # 吃单手续费
        slippage=0 / 10000,  # 滑点
        size=1,  # 杠杆倍数 默认为1
        pricetick=0.00000001,  # 价格精度
        capital=1000000,  # 本金
        annual_days=365,  # 一年的连续交易天数
        label=DataType.DCT,  # tick级别的市场选择
        mode=BacktestingMode.TICK  # tick级别回测
    )
    engine.add_strategy(TickStrategy, {})

    engine.load_data()
    engine.run_backtesting()

    # ----------- 回测并画图 --------------
    df = engine.calculate_result()
    engine.calculate_statistics()
    engine.show_chart()
