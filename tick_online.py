# -*- coding: utf-8 -*-
# zq
import multiprocessing
import threading
from datetime import datetime, timedelta
from time import sleep

import numpy as np
import objgraph
from ai_strategy import ai_TickStrategy
# from HFT_binance import cols_list, factor_calculation
import factor
from factor import *
from tz_ctastrategy import BarData, CtaTemplate, OrderData, TickData, TradeData
from tz_ctastrategy.backtesting import BacktestingEngine
from tz_ctastrategy.base import BacktestingMode, DataType, EngineType
from tz_riskmanager.risk_engine import save_setting
from tzquant.market.dingtalker import WebHook
from tzquant.market.get_gate_private_info import GetRestInfo
from tzquant.trader.utility import ArrayManager, BarGenerator, Interval


def run_single(
    symbol: str,
    capital: int,
    threshold: int,
    place_rate: int,
    side_short: int,
    side_long: int,
    out: int,
    model_symbol: str,
    queue,
):
    engine = BacktestingEngine()
    now_datetime = datetime.now().replace(minute=0, second=0, microsecond=0)
    # ---------------- 实盘 -----------------------
    engine.engine_type = EngineType.REAL
    engine.gateway_name = EngineType.REAL
    # 默认用的1m的kline
    engine.set_parameters(
        vt_symbol=f"{symbol}.{'binance_swap_u'}",
        start=now_datetime - timedelta(days=0.2),  # 形如：datetime(2022,1,1)
        end=now_datetime,  # 形如：datetime(2022,1,1)
        maker_fee=0 / 10000,  # 挂单手续费
        taker_fee=3 / 10000,  # 吃单手续费
        slippage=2 / 10000,  # 滑点
        size=1,  # 杠杆倍数 默认为1
        pricetick=0.00000001,  # 价格精度
        capital=capital,  # 本金
        annual_days=365,  # 一年的连续交易天数
        label=DataType.DCT,  # tick级别的市场选择
        mode=BacktestingMode.TICK,  # tick级别回测
    )
    engine.add_strategy(ai_TickStrategy, {})
    engine.strategy.add_msgClient(user_name='tzadmin',user_pass="tzlh123456",vm_host="8.218.98.219",vm_port=5672)
    # 连接实盘私有信息
    engine.queue_pri = queue
    engine.strategy.threshold = threshold
    engine.strategy.place_rate = place_rate
    engine.strategy.side_long = side_long
    engine.strategy.side_short = side_short
    engine.strategy.out = out
    engine.strategy.model_symbol = model_symbol
    print(engine.queue_pri, f"threshold:{engine.strategy.threshold}")
    # 启动风控 实盘不管大资金还是小资金都要启动风控
    engine.strategy.add_risk_manager(
        symbol_list=[symbol_ for symbol_ in config],
        key=risk_setting["key"],
        secret=risk_setting["secret"],
    )
    # threading.Thread(target=engine.subscribe_data_new, args=(True,)).start()  # 接收实盘行情信息
    # threading.Thread(target=engine.receive_depth_data_new).start()  # 接收实盘行情信息
    threading.Thread(target=engine.pub_info.sub_all_info).start()  # 接收实盘行情信息
    threading.Thread(target=engine.dma_depth_data).start()  # 接收实盘行情信息
    engine.receive_private_queue_data()  # 接收实盘私有信息


def private_run(queue_d: dict):
    # ----------------- 单独开启一个进程连接实盘私有信息 并向接收信息的进程发送私有信息 --------------------
    private_info = GetRestInfo(
        symbol_list=[symbol_ for symbol_ in config],
        queue_dict=queue_d,
        key=risk_setting["key"],
        secret=risk_setting["secret"],
    )
    private_info.futures_private_info(settle="usdt", open_status=True)


# ------------ 配置当前账户的每个品种的本金 --------------------------
config = {
    "bnbusdt": {
        "capital": 200,
        "threshold": 110_000,
        "place_rate": 4 / 10000,
        "side_short": 0.486669092309669,
        "side_long": 0.501161192426651,
        "out": 0.868409470852621,
        "model_symbol": "bnbusdt",
    },
    "xrpusdt": {
        "capital": 200,
        "threshold": 120_000,
        "place_rate": 4 / 10000,
        "side_short": 0.32127930817156064,
        "side_long": 0.6946675097360484,
        "out": 0.8762844744962693,
        "model_symbol": "xrpusdt",
    },
    # "ethusdt": {
    #     "capital": 500,
    #     "threshold": 1_300_000,
    #     "place_rate": 3 / 10000,
    #     "side_short": 0.2634882811428467,
    #     "side_long": 0.680843594470116,
    #     "out": 0.8882035683500493,
    #     "model_symbol": "ethusdt",
    # },
    # "adausdt": {
    #     "capital": 200,
    #     "threshold": 80_000,
    #     "place_rate": 4 / 10000,
    #     "side_short": 0.45787615871100296,
    #     "side_long": 0.5311125422982662,
    #     "out": 0.878213465847594,
    #     "model_symbol": "adausdt",
    # },
    "ltcusdt": {
        "capital": 200,
        "threshold": 60_000,
        "place_rate": 4 / 10000,
        "side_short": 0.3669318531087676,
        "side_long": 0.6364078827350275,
        "out": 0.8665249736082488,
        "model_symbol": "ltcusdt",
    },
}
# ------------------ 修改风控的配置信息 密钥 ---------------------------
risk_setting = {
    "key": "b51681ff6a8503b30b9cc8f552fa0adb",
    "secret": "e5d73c91ec8623909bab8b5f426d784c2fd6c94259c86923ccdfd525576e0348",
    "active": True,  # True 风控启动
    "order_flow_limit_1m": 5,  # 根据下单的标签来做1m的下单次数限制
    "net_max_down_level_1": 15,  # 实时最大回撤值1
    "net_max_down_level_2": 10,
    "net_max_down_level_3": 5,
    "net_level_1": 60*0.7,
    "net_level_2": 60*0.8,
    "net_level_3": 60*0.9,
    "position_limit": {
        "bnbusdt": 200 * 0.3 * 1.2,
        "xrpusdt": 200 * 0.3 * 1.2,
        "adausdt": 200 * 0.3 * 1.2,
        "ethusdt": 500 * 0.3 * 1.2,
        "ltcusdt": 200 * 0.3 * 1.2,
    },
    "position_time_out": 30,  # 持仓信息超过30秒没来会提醒
    "webhook_key": WebHook.songhe_ai_crypto.value,  # 钉钉发送的接收用户
}
total_capital = sum([config[symbol]["capital"] for symbol in config])
balance = 64
print("当前实际杠杆倍数：", total_capital * 0.3 / balance)
if __name__ == "__main__":
    save_setting(setting=risk_setting)  # 将risk_setting 的信息写入配置文件

    process = []
    # 创建一个进程间内存共享的队列
    queue_dict = {}
    for index, key in enumerate(config):
        q = multiprocessing.Queue()
        queue_dict[key] = q
        # 先启动私有信息的发送进程 确保初始化有数据
    p = multiprocessing.Process(target=private_run, args=(queue_dict,))
    p.start()
    sleep(3)

    for index, key in enumerate(config):
        p = multiprocessing.Process(
            target=run_single,
            args=(
                key,
                config[key]["capital"],
                config[key]["threshold"],
                config[key]["place_rate"],
                config[key]["side_short"],
                config[key]["side_long"],
                config[key]["out"],
                config[key]["model_symbol"],
                queue_dict[key],
            ),
        )
        p.start()
        process.append(p)
        sleep(10)
    for p in process:
        p.join()