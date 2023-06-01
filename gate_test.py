# -*- coding: utf-8 -*-
# Author:songhe

# 时序因子计算
import minio
import numpy as np
import pandas as pd
import os
import sys
import time
import math
import threading
from decimal import Decimal
from math import log10, floor
from copy import deepcopy
from datetime import datetime
from queue import Queue

# from execute_model.execute_model_v230216 import execute_model, execute_signal, factor_put_model_list, \
# write_file_by_line, kline_and_feat_col_list
# from tz_ctastrategy import (
#     BarData,
# )

cols_list = ['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1',
             'ask_price2', 'ask_size2', 'bid_price2', 'bid_size2', 'ask_price3',
             'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4',
             'bid_price4', 'bid_size4', 'ask_price5', 'ask_size5', 'bid_price5',
             'bid_size5', 'price', 'size', 'turnover', 'cum_size', 'ask_age',
             'bid_age', 'inf_ratio', 'arrive_rate', 'depth_price_range', 'bp_rank',
             'ap_rank', 'price_impact', 'depth_price_skew', 'depth_price_kurt',
             'rolling_return', 'buy_increasing', 'sell_increasing', 'price_idxmax',
             'center_deri_two', 'quasi', 'last_range', 'avg_trade_volume',
             'avg_spread', 'avg_turnover', 'abs_volume_kurt', 'abs_volume_skew',
             'volume_kurt', 'volume_skew', 'price_kurt', 'price_skew',
             'bv_divide_tn', 'av_divide_tn', 'weighted_price_to_mid',
             'ask_withdraws', 'bid_withdraws', 'z_t', 'voi', 'voi2', 'wa', 'wb',
             'slope', 'mpb', 'price_weighted_pressure', 'volume_order_imbalance',
             'get_mid_price_change']


def round2(x, sig):
    if abs(x) < 1e-8:
        return 0
    sig = str(10 ** (-sig + int(floor(log10(abs(x)))) + 1))
    x = str(round(x, 10))
    return float(Decimal(x).quantize(Decimal(sig), rounding="ROUND_HALF_EVEN"))


def shift_(interval):
    return -(interval + 1)


def skew_(df):
    n = len(df)
    x = df - np.mean(df)
    # y = np.sqrt(n) * sum(np.power(x, 3)) / np.power(sum(np.power(x, 2)), (3 / 2))
    y = np.sqrt(n) * sum(x ** 3) / sum(x ** 2) ** (3 / 2)
    y = y * np.sqrt(n * (n - 1)) / (n - 2)
    return y


def kurt_(df):
    n = len(df)
    x = df - np.mean(df)
    r = n * sum(x ** 4) / sum(x ** 2) ** 2
    y = ((n + 1) * (r - 3) + 6) * (n - 1) / ((n - 2) * (n - 3))
    return y


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def test_data_consistency(test_data: float, col_name: str, ori_data=np.array([]), index=0, sum_num_=0,
                          df_2d_=np.atleast_2d(np.zeros(65)), col_dict={}):
    if test_data == 0:
        if ori_data[col_dict[col_name]] == 0:
            dif_rate = 0
        else:
            dif_rate = abs((test_data - ori_data[col_dict[col_name]]) / ori_data[col_dict[col_name]])
    else:
        # print('test_data:',test_data)
        # print('ori_data:',ori_data[col_dict[col_name]])
        dif_rate = abs((test_data - ori_data[col_dict[col_name]]) / test_data)
    # if dif_rate > 0.0001 and round(test_data, 6) != round(ori_data[col_dict[col_name]], 6):
    if dif_rate > 0.0001:
        print('{}该数据批计算和实时流计算数值不一致---下标:{}---流数据:{}---批数据:{}'.format(col_name, index, test_data,
                                                                     ori_data[col_dict[col_name]]))
        # if col_name == 'depth_1s_buy_vwap_percentile_rolling_60':
        #     print(col_name)

    # df = df.replace(np.inf, 1)
    # df = df.replace(-np.inf, -1)
    if col_name in cols_list:
        col_name_index = cols_list.index(col_name)
        # 极大值和极小值强制赋值为 1 -1
        if np.isinf(test_data):
            df_2d_[0][col_name_index] = 1
        elif np.isneginf(test_data):
            df_2d_[0][col_name_index] = -1
        else:
            df_2d_[0][col_name_index] = test_data  # 组装入模的数据
    # test_columns.append(col_name)
    return sum_num_


def get_age(prices):
    last_value = prices[-1]
    age = 0
    for i in range(2, len(prices)):
        if prices[-i] != last_value:
            return age
        age += 1
    return age


def first_location_of_maximum(x):
    max_value = max(x)  # 一个for 循环
    for loc in range(len(x)):
        if x[loc] == max_value:
            return loc + 1


def mean_second_derivative_centra(x):
    sum_value = 0
    for i in range(len(x) - 5):
        sum_value += (x[i + 5] - 2 * x[i + 3] + x[i]) / 2
    return sum_value / (2 * (len(x) - 5))


def _bid_withdraws_volume(l, n, levels=5):
    withdraws = 0
    for price_index in range(2, 2 + 4 * levels, 4):
        now_p = n[price_index]
        for price_last_index in range(2, 2 + 4 * levels, 4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

    return withdraws


def _ask_withdraws_volume(l, n, levels=5):
    withdraws = 0
    for price_index in range(0, 4 * levels, 4):
        now_p = n[price_index]
        for price_last_index in range(0, 4 * levels, 4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

    return withdraws


def factor_calculation(df: bytearray, index, feat_dict: dict, sum_num_: int, df_2d_, col_dict):
    closetime = df[col_dict['closetime']]
    ask_price1 = df[col_dict['ask_price1']]
    bid_price1 = df[col_dict['bid_price1']]
    ask_price2 = df[col_dict['ask_price2']]
    bid_price2 = df[col_dict['bid_price2']]
    ask_price3 = df[col_dict['ask_price3']]
    bid_price3 = df[col_dict['bid_price3']]
    ask_price4 = df[col_dict['ask_price4']]
    bid_price4 = df[col_dict['bid_price4']]
    ask_price5 = df[col_dict['ask_price5']]
    bid_price5 = df[col_dict['bid_price5']]
    ask_size1 = df[col_dict['ask_size1']]
    bid_size1 = df[col_dict['bid_size1']]
    ask_size2 = df[col_dict['ask_size2']]
    bid_size2 = df[col_dict['bid_size2']]
    ask_size3 = df[col_dict['ask_size3']]
    bid_size3 = df[col_dict['bid_size3']]
    ask_size4 = df[col_dict['ask_size4']]
    bid_size4 = df[col_dict['bid_size4']]
    ask_size5 = df[col_dict['ask_size5']]
    bid_size5 = df[col_dict['bid_size5']]
    price = df[col_dict['price']]
    size = df[col_dict['size']]
    cum_size = df[col_dict['cum_size']]
    turnover = df[col_dict['turnover']]
    feat_dict['closetime'] = np.append(feat_dict['closetime'], closetime)
    feat_dict['ask_price1'] = np.append(feat_dict['ask_price1'], ask_price1)
    feat_dict['bid_price1'] = np.append(feat_dict['bid_price1'], bid_price1)
    feat_dict['ask_price2'] = np.append(feat_dict['ask_price2'], ask_price2)
    feat_dict['bid_price2'] = np.append(feat_dict['bid_price2'], bid_price2)
    feat_dict['ask_price3'] = np.append(feat_dict['ask_price3'], ask_price3)
    feat_dict['bid_price3'] = np.append(feat_dict['bid_price3'], bid_price3)
    feat_dict['ask_price4'] = np.append(feat_dict['ask_price4'], ask_price4)
    feat_dict['bid_price4'] = np.append(feat_dict['bid_price4'], bid_price4)
    feat_dict['ask_price5'] = np.append(feat_dict['ask_price5'], ask_price5)
    feat_dict['bid_price5'] = np.append(feat_dict['bid_price5'], bid_price5)
    feat_dict['ask_size1'] = np.append(feat_dict['ask_size1'], ask_size1)
    feat_dict['bid_size1'] = np.append(feat_dict['bid_size1'], bid_size1)
    feat_dict['ask_size2'] = np.append(feat_dict['ask_size2'], ask_size2)
    feat_dict['bid_size2'] = np.append(feat_dict['bid_size2'], bid_size2)
    feat_dict['ask_size3'] = np.append(feat_dict['ask_size3'], ask_size3)
    feat_dict['bid_size3'] = np.append(feat_dict['bid_size3'], bid_size3)
    feat_dict['ask_size4'] = np.append(feat_dict['ask_size4'], ask_size4)
    feat_dict['bid_size4'] = np.append(feat_dict['bid_size4'], bid_size4)
    feat_dict['ask_size5'] = np.append(feat_dict['ask_size5'], ask_size5)
    feat_dict['bid_size5'] = np.append(feat_dict['bid_size5'], bid_size5)
    feat_dict['price'] = np.append(feat_dict['price'], price)
    feat_dict['size'] = np.append(feat_dict['size'], size)
    feat_dict['cum_size'] = np.append(feat_dict['cum_size'], cum_size)
    feat_dict['turnover'] = np.append(feat_dict['turnover'], turnover)
    sum_num_ = test_data_consistency(feat_dict['closetime'][-1], 'closetime',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['ask_price1'][-1], 'ask_price1',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['bid_price1'][-1], 'bid_price1',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['ask_price2'][-1], 'ask_price2',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['bid_price2'][-1], 'bid_price2',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['ask_price3'][-1], 'ask_price3',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['bid_price3'][-1], 'bid_price3',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['ask_price4'][-1], 'ask_price4',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['bid_price4'][-1], 'bid_price4',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['ask_price5'][-1], 'ask_price5',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['bid_price5'][-1], 'bid_price5',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['ask_size1'][-1], 'ask_size1',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['bid_size1'][-1], 'bid_size1',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['ask_size2'][-1], 'ask_size2',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['bid_size2'][-1], 'bid_size2',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['ask_size3'][-1], 'ask_size3',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['bid_size3'][-1], 'bid_size3',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['ask_size4'][-1], 'ask_size4',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['bid_size4'][-1], 'bid_size4',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['ask_size5'][-1], 'ask_size5',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['bid_size5'][-1], 'bid_size5',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['price'][-1], 'price',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['size'][-1], 'size',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['cum_size'][-1], 'cum_size',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    sum_num_ = test_data_consistency(feat_dict['turnover'][-1], 'turnover',
                                     df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass
    # bid_age
    if len(feat_dict['bid_price1']) > 10:
        # bp1_changes = bp1.rolling(rolling).apply(get_age, engine='numba', raw=True).fillna(0)
        bp1_changes = get_age(feat_dict['bid_price1'][-10:])
        # print(feat_dict)
        if bp1_changes is None:
            feat_dict['bid_age'] = np.append(feat_dict['bid_age'], 0)
        else:
            feat_dict['bid_age'] = np.append(feat_dict['bid_age'], bp1_changes)

        sum_num_ = test_data_consistency(feat_dict['bid_age'][-1], 'bid_age',
                                         df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass

    # ask_age
    if len(feat_dict['ask_price1']) > 10:

        ap1_changes = get_age(feat_dict['ask_price1'][-10:])
        if ap1_changes is None:
            feat_dict['ask_age'] = np.append(feat_dict['ask_age'], 0)
        else:
            feat_dict['ask_age'] = np.append(feat_dict['ask_age'], ap1_changes)

        sum_num_ = test_data_consistency(feat_dict['ask_age'][-1], 'ask_age',
                                         df, index, sum_num_, df_2d_, col_dict)  # 数据一致性校验 pass

    # inf_ratio
    if len(feat_dict['price']) > 100:
        # p = feat_dict['price'].copy
        # feat_dict['price'][np.isnan(feat_dict['price'])] = 0
        # p[np.isnan(p)] = 0
        quasi = np.sum(np.abs(np.diff(feat_dict['price'])[-100:]))
        # quasi = np.sum(np.abs(np.diff(p)[-100:]))
        dif = np.abs(np.diff(feat_dict['price'][-100:]))
        # dif = np.abs(np.diff(p[-100:]))
        if quasi is None or dif is None:
            quasi = 10
            dif = 10
        else:
            quasi = quasi
            dif = dif
        # final = quasi/(dif+quasi)
        feat_dict['inf_ratio'] = np.append(feat_dict['inf_ratio'], quasi / (dif + quasi))
        sum_num_ = test_data_consistency(feat_dict['inf_ratio'][-1], 'inf_ratio',
                                         df, index, sum_num_, df_2d_, col_dict)

    # depth_price_range
    if len(feat_dict['ask_price1']) > 100:
        depth_price_range = ((np.max(feat_dict['ask_price1'][-100:])) / (np.min(feat_dict['ask_price1'][-100:]))) - 1
        if depth_price_range is None:
            feat_dict['depth_price_range'] = np.append(feat_dict['depth_price_range'], 0)
        else:
            feat_dict['depth_price_range'] = np.append(feat_dict['depth_price_range'], depth_price_range)
        sum_num_ = test_data_consistency(feat_dict['depth_price_range'][-1], 'depth_price_range',
                                         df, index, sum_num_, df_2d_, col_dict)

    # arrive_rate
    if len(feat_dict['closetime']) > 300:
        res = feat_dict['closetime'][-1] - feat_dict['closetime'][shift_(300)]
        if res is None:
            res = 0
        else:
            res = res
        feat_dict['arrive_rate'] = np.append(feat_dict['arrive_rate'], res / 300)
        sum_num_ = test_data_consistency(feat_dict['arrive_rate'][-1], 'arrive_rate',
                                         df, index, sum_num_, df_2d_, col_dict)

    # bp_rank
    if len(feat_dict['bid_price1']) > 100:
        bp_rank = feat_dict['bid_price1'][-100:].argsort().argsort()

        if bp_rank is None:
            feat_dict['bp_rank'] = np.append(feat_dict['bp_rank'], 0)
        else:
            feat_dict['bp_rank'] = np.append(feat_dict['bp_rank'], bp_rank / 100 * 2 - 1)
            # feat_dict['bp_rank'] = np.append(feat_dict['bp_rank'], bp_rank)
        sum_num_ = test_data_consistency(feat_dict['bp_rank'][-1], 'bp_rank',
                                         df, index, sum_num_, df_2d_, col_dict)

    # ap_rank
    if len(feat_dict['ask_price1']) > 100:
        ap_rank = feat_dict['ask_price1'][-100:].argsort().argsort()
        if bp_rank is None:
            feat_dict['ap_rank'] = np.append(feat_dict['ap_rank'], 0)
        else:
            feat_dict['ap_rank'] = np.append(feat_dict['ap_rank'], ap_rank / 100 * 2 - 1)
        sum_num_ = test_data_consistency(feat_dict['ap_rank'][-1], 'ap_rank',
                                         df, index, sum_num_, df_2d_, col_dict)

    # price_impact
    ask, bid, ask_v, bid_v = 0, 0, 0, 0
    for i in range(1, 6):
        ask += df[col_dict[f'ask_price{i}']] * df[col_dict[f'ask_size{i}']]
        bid += df[col_dict[f'bid_price{i}']] * df[col_dict[f'bid_size{i}']]
        ask_v += df[col_dict[f'ask_size{i}']]
        bid_v += df[col_dict[f'bid_size{i}']]
    ask /= ask_v
    bid /= bid_v
    price_impact = -(df[col_dict['ask_price1']] - ask) / df[col_dict['ask_price1']] - (
                df[col_dict['bid_price1']] - bid) / df[col_dict['bid_price1']]
    feat_dict['price_impact'] = np.append(feat_dict['price_impact'], price_impact)
    sum_num_ = test_data_consistency(feat_dict['price_impact'][-1], 'price_impact',
                                     df, index, sum_num_, df_2d_, col_dict)

    # depth_price_skew
    # if len(feat_dict['ask_price1'])>1:
    # price_skew = np.hstack((feat_dict['bid_price5'][-1], feat_dict['bid_price4'][-1], feat_dict['bid_price3'][-1], feat_dict['bid_price2'][-1],feat_dict['bid_price1'][-1],
    # feat_dict['ask_price5'][-1], feat_dict['ask_price4'][-1], feat_dict['ask_price3'][-1], feat_dict['ask_price2'][-1],feat_dict['ask_price1'][-1]))
    price_skew = np.hstack((df[col_dict['bid_price5']], df[col_dict['bid_price4']], df[col_dict['bid_price3']],
                            df[col_dict['bid_price2']], df[col_dict['bid_price1']],
                            df[col_dict['ask_price5']], df[col_dict['ask_price4']], df[col_dict['ask_price3']],
                            df[col_dict['ask_price2']], df[col_dict['ask_price1']]))
    depth_price_skew = skew_(price_skew)
    feat_dict['depth_price_skew'] = np.append(feat_dict['depth_price_skew'], depth_price_skew)
    sum_num_ = test_data_consistency(feat_dict['depth_price_skew'][-1], 'depth_price_skew',
                                     df, index, sum_num_, df_2d_, col_dict)

    # depth_price_kurt
    price_kurt = np.hstack((df[col_dict['bid_price5']], df[col_dict['bid_price4']], df[col_dict['bid_price3']],
                            df[col_dict['bid_price2']], df[col_dict['bid_price1']],
                            df[col_dict['ask_price5']], df[col_dict['ask_price4']], df[col_dict['ask_price3']],
                            df[col_dict['ask_price2']], df[col_dict['ask_price1']]))
    depth_price_kurt = kurt_(price_kurt)
    feat_dict['depth_price_kurt'] = np.append(feat_dict['depth_price_kurt'], depth_price_kurt)
    sum_num_ = test_data_consistency(feat_dict['depth_price_kurt'][-1], 'depth_price_kurt',
                                     df, index, sum_num_, df_2d_, col_dict)

    # rolling_return
    if len(feat_dict['ask_price1']) > 100 and len(feat_dict['bid_price1']) > 100:
        mp = (feat_dict['ask_price1'][-1] + feat_dict['bid_price1'][-1]) / 2
        mp_100 = (feat_dict['ask_price1'][shift_(100)] + feat_dict['bid_price1'][shift_(100)]) / 2
        rolling_return = (mp - mp_100) / mp
        if rolling_return is None:
            feat_dict['rolling_return'] = np.append(feat_dict['rolling_return'], 0)
        else:
            feat_dict['rolling_return'] = np.append(feat_dict['rolling_return'], rolling_return)
        sum_num_ = test_data_consistency(feat_dict['rolling_return'][-1], 'rolling_return',
                                         df, index, sum_num_, df_2d_, col_dict)

    # buy_increasing
    if len(feat_dict['size']) > 200:
        b_v = feat_dict['size'].copy()
        b_v[b_v < 0] = 0
        # if v[-1] <0:
        #     v[-1] = 0
        # else:
        #     v[-1] = v[-1]
        b_v = np.where(np.isnan(b_v), 0, b_v)
        buy_increasing = np.log1p((np.sum(b_v[-100 * 2:]) + 1) / (np.sum(b_v[-100:]) + 1))
        if buy_increasing is None:
            feat_dict['buy_increasing'] = np.append(feat_dict['buy_increasing'], 1)
        else:
            feat_dict['buy_increasing'] = np.append(feat_dict['buy_increasing'], buy_increasing)
        sum_num_ = test_data_consistency(feat_dict['buy_increasing'][-1], 'buy_increasing',
                                         df, index, sum_num_, df_2d_, col_dict)

    # sell_increasing
    if len(feat_dict['size']) > 200:
        s_v = feat_dict['size'].copy()
        s_v[s_v > 0] = 0
        # if v[-1] >0:
        #     v[-1] = 0
        # else:
        #     v[-1] = v[-1]
        s_v = np.where(np.isnan(s_v), 0, s_v)
        sell_increasing = np.log1p((np.sum(s_v[-100 * 2:]) - 1) / (np.sum(s_v[-100:]) - 1))
        if sell_increasing is None:
            feat_dict['sell_increasing'] = np.append(feat_dict['sell_increasing'], 1)
        else:
            feat_dict['sell_increasing'] = np.append(feat_dict['sell_increasing'], sell_increasing)
        sum_num_ = test_data_consistency(feat_dict['sell_increasing'][-1], 'sell_increasing',
                                         df, index, sum_num_, df_2d_, col_dict)

    # price_idxmax
    if len(feat_dict['ask_price1']) > 20:
        price_idxmax = first_location_of_maximum(feat_dict['ask_price1'][-20:])
        if price_idxmax is None:
            feat_dict['price_idxmax'] = np.append(feat_dict['price_idxmax'], 0)
        else:
            feat_dict['price_idxmax'] = np.append(feat_dict['price_idxmax'], price_idxmax)
        sum_num_ = test_data_consistency(feat_dict['price_idxmax'][-1], 'price_idxmax',
                                         df, index, sum_num_, df_2d_, col_dict)

    # center_deri_two
    if len(feat_dict['ask_price1']) > 20:
        center_deri_two = mean_second_derivative_centra(feat_dict['ask_price1'][-20:])
        if center_deri_two is None:
            feat_dict['center_deri_two'] = np.append(feat_dict['center_deri_two'], 0)
        else:
            feat_dict['center_deri_two'] = np.append(feat_dict['center_deri_two'], center_deri_two)
        sum_num_ = test_data_consistency(feat_dict['center_deri_two'][-1], 'center_deri_two',
                                         df, index, sum_num_, df_2d_, col_dict)

    # quasi
    if len(feat_dict['ask_price1']) > 100:
        quasi = np.sum(np.abs(np.diff(feat_dict['ask_price1'])[-100:]))
        if quasi is None:
            feat_dict['quasi'] = np.append(feat_dict['quasi'], 0)
        else:
            feat_dict['quasi'] = np.append(feat_dict['quasi'], quasi)
        sum_num_ = test_data_consistency(feat_dict['quasi'][-1], 'quasi',
                                         df, index, sum_num_, df_2d_, col_dict)

    # last_range
    if len(feat_dict['price']) > 100:
        last_range = np.sum(np.abs(np.diff(feat_dict['price'])[-100:]))
        if last_range is None:
            feat_dict['last_range'] = np.append(feat_dict['last_range'], 0)
        else:
            feat_dict['last_range'] = np.append(feat_dict['last_range'], last_range)
        sum_num_ = test_data_consistency(feat_dict['last_range'][-1], 'last_range',
                                         df, index, sum_num_, df_2d_, col_dict)

    # avg_trade_volume
    if len(feat_dict['size']) > 100:
        a_s = feat_dict['size'].copy()
        a_s = np.where(np.isnan(a_s), 0, a_s)
        sizes = np.abs(a_s[::-1])
        rolling_sum = np.cumsum(sizes)
        rolling_sum = rolling_sum - np.concatenate((np.zeros(100), rolling_sum[:-100]))
        avg_trade_volume = np.concatenate((np.full(100 - 1, np.nan), rolling_sum))[:-1]

        feat_dict['avg_trade_volume'] = np.append(feat_dict['avg_trade_volume'], avg_trade_volume[::-1])
        sum_num_ = test_data_consistency(feat_dict['avg_trade_volume'][-1], 'avg_trade_volume',
                                         df, index, sum_num_, df_2d_, col_dict)

    # avg_spread
    if len(feat_dict['ask_price1']) > 200 and len(feat_dict['bid_price1']) > 200:
        avg_spread = np.mean((feat_dict['ask_price1'] - feat_dict['bid_price1'])[-200:])
        if avg_spread is None:
            feat_dict['avg_spread'] = np.append(feat_dict['avg_spread'], 0)
        else:
            feat_dict['avg_spread'] = np.append(feat_dict['avg_spread'], avg_spread)
        sum_num_ = test_data_consistency(feat_dict['avg_spread'][-1], 'avg_spread',
                                         df, index, sum_num_, df_2d_, col_dict)

    # avg_turnover
    avg_turnover = np.sum(np.hstack((df[col_dict['bid_size5']], df[col_dict['bid_size4']], df[col_dict['bid_size3']],
                                     df[col_dict['bid_size2']], df[col_dict['bid_size1']],
                                     df[col_dict['ask_size5']], df[col_dict['ask_size4']], df[col_dict['ask_size3']],
                                     df[col_dict['ask_size2']], df[col_dict['ask_size1']])))
    feat_dict['avg_turnover'] = np.append(feat_dict['avg_turnover'], avg_turnover)
    sum_num_ = test_data_consistency(feat_dict['avg_turnover'][-1], 'avg_turnover',
                                     df, index, sum_num_, df_2d_, col_dict)

    # abs_volume_kurt
    if len(feat_dict['size']) > 500:
        abs_volume_kurt = kurt_(np.abs(feat_dict['size'])[-500:])
        if abs_volume_kurt is None:
            feat_dict['abs_volume_kurt'] = np.append(feat_dict['abs_volume_kurt'], 0)
        else:
            feat_dict['abs_volume_kurt'] = np.append(feat_dict['abs_volume_kurt'], abs_volume_kurt)
        sum_num_ = test_data_consistency(feat_dict['abs_volume_kurt'][-1], 'abs_volume_kurt',
                                         df, index, sum_num_, df_2d_, col_dict)

    # abs_volume_skew
    if len(feat_dict['size']) > 500:
        abs_volume_skew = skew_(np.abs(feat_dict['size'])[-500:])
        if abs_volume_skew is None:
            feat_dict['abs_volume_skew'] = np.append(feat_dict['abs_volume_skew'], 0)
        else:
            feat_dict['abs_volume_skew'] = np.append(feat_dict['abs_volume_skew'], abs_volume_skew)
        sum_num_ = test_data_consistency(feat_dict['abs_volume_skew'][-1], 'abs_volume_skew',
                                         df, index, sum_num_, df_2d_, col_dict)

    # volume_kurt
    if len(feat_dict['size']) > 500:
        volume_kurt = kurt_(feat_dict['size'][-500:])
        if volume_kurt is None:
            feat_dict['volume_kurt'] = np.append(feat_dict['volume_kurt'], 0)
        else:
            feat_dict['volume_kurt'] = np.append(feat_dict['volume_kurt'], volume_kurt)
        sum_num_ = test_data_consistency(feat_dict['volume_kurt'][-1], 'volume_kurt',
                                         df, index, sum_num_, df_2d_, col_dict)

    # volume_skew
    if len(feat_dict['size']) > 500:
        volume_skew = skew_(feat_dict['size'][-500:])
        if volume_skew is None:
            feat_dict['volume_skew'] = np.append(feat_dict['volume_skew'], 0)
        else:
            feat_dict['volume_skew'] = np.append(feat_dict['volume_skew'], volume_skew)
        sum_num_ = test_data_consistency(feat_dict['volume_skew'][-1], 'volume_skew',
                                         df, index, sum_num_, df_2d_, col_dict)

    # price_kurt
    if len(feat_dict['price']) > 500:
        price_kurt = kurt_(feat_dict['price'][-500:])
        if price_kurt is None:
            feat_dict['price_kurt'] = np.append(feat_dict['price_kurt'], 0)
        else:
            feat_dict['price_kurt'] = np.append(feat_dict['price_kurt'], price_kurt)
        sum_num_ = test_data_consistency(feat_dict['price_kurt'][-1], 'price_kurt',
                                         df, index, sum_num_, df_2d_, col_dict)

    # price_skew
    if len(feat_dict['price']) > 500:
        price_skew = np.abs(skew_(feat_dict['price'][-500:]))
        if price_skew is None:
            feat_dict['price_skew'] = np.append(feat_dict['price_skew'], 0)
        else:
            feat_dict['price_skew'] = np.append(feat_dict['price_skew'], price_skew)
        sum_num_ = test_data_consistency(feat_dict['price_skew'][-1], 'price_skew',
                                         df, index, sum_num_, df_2d_, col_dict)

    # bv_divide_tn
    if len(feat_dict['size']) > 10:
        bvs = feat_dict['bid_size1'][-1] + feat_dict['bid_size2'][-1] + feat_dict['bid_size3'][-1] + \
              feat_dict['bid_size4'][-1] + feat_dict['bid_size5'][-1]
        bv = feat_dict['size'].copy()
        # if bv > 0:
        #     bv = 0
        # else:
        # bv[-1] = bv[-1]
        bv[bv > 0] = 0
        # print(np.sum(bv[-10:]))
        bv_divide_tn = np.sum(bv[-10:]) / bvs
        # print('bv_divide_tn:',bv_divide_tn)
        if bv_divide_tn is None:
            feat_dict['bv_divide_tn'] = np.append(feat_dict['bv_divide_tn'], 0)
        else:
            feat_dict['bv_divide_tn'] = np.append(feat_dict['bv_divide_tn'], bv_divide_tn)
        sum_num_ = test_data_consistency(feat_dict['bv_divide_tn'][-1], 'bv_divide_tn',
                                         df, index, sum_num_, df_2d_, col_dict)

    # av_divide_tn
    if len(feat_dict['size']) > 10:
        avs = feat_dict['ask_size1'][-1] + feat_dict['ask_size2'][-1] + feat_dict['ask_size3'][-1] + \
              feat_dict['ask_size4'][-1] + feat_dict['ask_size5'][-1]
        av = feat_dict['size'].copy()
        # print('av_last:',av)
        # if av[-1] < 0:
        #     av[-1] = 0
        # else:
        #     av[-1] = av[-1]
        # print('av_now:',av)
        av[av < 0] = 0
        av_divide_tn = np.sum(av[-10:]) / avs
        if av_divide_tn is None:
            feat_dict['av_divide_tn'] = np.append(feat_dict['av_divide_tn'], 0)
        else:
            feat_dict['av_divide_tn'] = np.append(feat_dict['bv_divide_tn'], av_divide_tn)
        sum_num_ = test_data_consistency(feat_dict['av_divide_tn'][-1], 'av_divide_tn',
                                         df, index, sum_num_, df_2d_, col_dict)

    # weighted_price_to_mid
    avs_aps, bvs_bps, avs, bvs = 0, 0, 0, 0
    for i in range(1, 6):
        avs_aps += df[col_dict[f'ask_price{i}']] * df[col_dict[f'ask_size{i}']]
        bvs_bps += df[col_dict[f'bid_price{i}']] * df[col_dict[f'bid_size{i}']]
        avs += df[col_dict[f'ask_size{i}']]
        bvs += df[col_dict[f'bid_size{i}']]
    mp = (df[col_dict['ask_price1']] + df[col_dict['bid_price1']]) / 2
    weighted_price_to_mid = (avs_aps + bvs_bps) / (avs + bvs) - mp
    feat_dict['weighted_price_to_mid'] = np.append(feat_dict['weighted_price_to_mid'], weighted_price_to_mid)
    sum_num_ = test_data_consistency(feat_dict['weighted_price_to_mid'][-1], 'weighted_price_to_mid',
                                     df, index, sum_num_, df_2d_, col_dict)

    # ask_withdraws
    # ask_withdraws = np.array([])
    if len(feat_dict['ask_price1']) > 1:
        ask_ob_values_last = np.hstack((feat_dict['ask_price1'][-2], feat_dict['ask_size1'][-2],
                                        feat_dict['bid_price1'][-2], feat_dict['bid_size1'][-2],
                                        feat_dict['ask_price2'][-2], feat_dict['ask_size2'][-2],
                                        feat_dict['bid_price2'][-2], feat_dict['bid_size2'][-2],
                                        feat_dict['ask_price3'][-2], feat_dict['ask_size3'][-2],
                                        feat_dict['bid_price3'][-2], feat_dict['bid_size3'][-2],
                                        feat_dict['ask_price4'][-2], feat_dict['ask_size4'][-2],
                                        feat_dict['bid_price4'][-2], feat_dict['bid_size4'][-2],
                                        feat_dict['ask_price5'][-2], feat_dict['ask_size5'][-2],
                                        feat_dict['bid_price5'][-2], feat_dict['bid_size5'][-2]))
        ask_ob_values_now = np.hstack((feat_dict['ask_price1'][-1], feat_dict['ask_size1'][-1],
                                       feat_dict['bid_price1'][-1], feat_dict['bid_size1'][-1],
                                       feat_dict['ask_price2'][-1], feat_dict['ask_size2'][-1],
                                       feat_dict['bid_price2'][-1], feat_dict['bid_size2'][-1],
                                       feat_dict['ask_price3'][-1], feat_dict['ask_size3'][-1],
                                       feat_dict['bid_price3'][-1], feat_dict['bid_size3'][-1],
                                       feat_dict['ask_price4'][-1], feat_dict['ask_size4'][-1],
                                       feat_dict['bid_price4'][-1], feat_dict['bid_size4'][-1],
                                       feat_dict['ask_price5'][-1], feat_dict['ask_size5'][-1],
                                       feat_dict['bid_price5'][-1], feat_dict['bid_size5'][-1]))

        ask_withdraws = _ask_withdraws_volume(ask_ob_values_last, ask_ob_values_now)

        feat_dict['ask_withdraws'] = np.append(feat_dict['ask_withdraws'], ask_withdraws)
        sum_num_ = test_data_consistency(feat_dict['ask_withdraws'][-1], 'ask_withdraws',
                                         df, index, sum_num_, df_2d_, col_dict)

    # bid_withdraws
    # bid_withdraws = np.array([])
    if len(feat_dict['ask_price1']) > 1:
        bid_ob_values_last = np.hstack((feat_dict['ask_price1'][-2], feat_dict['ask_size1'][-2],
                                        feat_dict['bid_price1'][-2], feat_dict['bid_size1'][-2],
                                        feat_dict['ask_price2'][-2], feat_dict['ask_size2'][-2],
                                        feat_dict['bid_price2'][-2], feat_dict['bid_size2'][-2],
                                        feat_dict['ask_price3'][-2], feat_dict['ask_size3'][-2],
                                        feat_dict['bid_price3'][-2], feat_dict['bid_size3'][-2],
                                        feat_dict['ask_price4'][-2], feat_dict['ask_size4'][-2],
                                        feat_dict['bid_price4'][-2], feat_dict['bid_size4'][-2],
                                        feat_dict['ask_price5'][-2], feat_dict['ask_size5'][-2],
                                        feat_dict['bid_price5'][-2], feat_dict['bid_size5'][-2]))
        bid_ob_values_now = np.hstack((feat_dict['ask_price1'][-1], feat_dict['ask_size1'][-1],
                                       feat_dict['bid_price1'][-1], feat_dict['bid_size1'][-1],
                                       feat_dict['ask_price2'][-1], feat_dict['ask_size2'][-1],
                                       feat_dict['bid_price2'][-1], feat_dict['bid_size2'][-1],
                                       feat_dict['ask_price3'][-1], feat_dict['ask_size3'][-1],
                                       feat_dict['bid_price3'][-1], feat_dict['bid_size3'][-1],
                                       feat_dict['ask_price4'][-1], feat_dict['ask_size4'][-1],
                                       feat_dict['bid_price4'][-1], feat_dict['bid_size4'][-1],
                                       feat_dict['ask_price5'][-1], feat_dict['ask_size5'][-1],
                                       feat_dict['bid_price5'][-1], feat_dict['bid_size5'][-1]))

        bid_withdraws = _bid_withdraws_volume(bid_ob_values_last, bid_ob_values_now)

        feat_dict['bid_withdraws'] = np.append(feat_dict['bid_withdraws'], bid_withdraws)
        sum_num_ = test_data_consistency(feat_dict['bid_withdraws'][-1], 'bid_withdraws',
                                         df, index, sum_num_, df_2d_, col_dict)

    # z_t
    tick_fac_data = np.log(df[col_dict['price']]) - np.log(
        (df[col_dict['ask_price1']] + df[col_dict['bid_price1']]) / 2)
    feat_dict['z_t'] = np.append(feat_dict['z_t'], tick_fac_data)
    sum_num_ = test_data_consistency(feat_dict['z_t'][-1], 'z_t',
                                     df, index, sum_num_, df_2d_, col_dict)

    # voi
    if len(feat_dict['bid_price1']) > 1:
        bid_sub_price = feat_dict['bid_price1'][-1] - feat_dict['bid_price1'][shift_(1)]
        ask_sub_price = feat_dict['ask_price1'][-1] - feat_dict['ask_price1'][shift_(1)]
        bid_sub_volume = feat_dict['bid_size1'][-1] - feat_dict['bid_size1'][shift_(1)]
        ask_sub_volume = feat_dict['ask_size1'][-1] - feat_dict['ask_size1'][shift_(1)]
        bid_volume_change = bid_sub_volume
        ask_volume_change = ask_sub_volume
        if bid_sub_price < 0:
            bid_volume_change = 0
        if bid_sub_price > 0:
            bid_volume_change = feat_dict['bid_size1'][-1]
        if ask_sub_price > 0:
            ask_volume_change = 0
        if ask_sub_price < 0:
            ask_volume_change = feat_dict['ask_size1'][-1]
        voi = (bid_volume_change - ask_volume_change) / feat_dict['cum_size'][-1]
        feat_dict['voi'] = np.append(feat_dict['voi'], voi)
        sum_num_ = test_data_consistency(feat_dict['voi'][-1], 'voi',
                                         df, index, sum_num_, df_2d_, col_dict)

    # cal_weight_volume
    w = [1 - (i - 1) / 5 for i in range(1, 6)]
    w = np.array(w) / sum(w)
    wb = df[col_dict['bid_size1']] * w[0] + df[col_dict['bid_size2']] * w[1] + df[col_dict['bid_size3']] * w[2] + df[
        col_dict['bid_size4']] * w[3] + df[col_dict['bid_size5']] * w[4]
    wa = df[col_dict['ask_size1']] * w[0] + df[col_dict['ask_size2']] * w[1] + df[col_dict['ask_size3']] * w[2] + df[
        col_dict['ask_size4']] * w[3] + df[col_dict['ask_size5']] * w[4]
    feat_dict['wb'] = np.append(feat_dict['wb'], wa)
    feat_dict['wa'] = np.append(feat_dict['wa'], wb)
    sum_num_ = test_data_consistency(feat_dict['wb'][-1], 'wb',
                                     df, index, sum_num_, df_2d_, col_dict)
    sum_num_ = test_data_consistency(feat_dict['wa'][-1], 'wa',
                                     df, index, sum_num_, df_2d_, col_dict)

    # voi2
    if len(feat_dict['ask_price1']) > 1:
        bid_sub_price_2 = feat_dict['bid_price1'][-1] - feat_dict['bid_price1'][shift_(1)]
        ask_sub_price_2 = feat_dict['ask_price1'][-1] - feat_dict['ask_price1'][shift_(1)]
        bid_sub_volume_2 = feat_dict['wa'][-1] - feat_dict['wa'][shift_(1)]
        ask_sub_volume_2 = feat_dict['wb'][-1] - feat_dict['wb'][shift_(1)]
        bid_volume_change_2 = bid_sub_volume_2
        ask_volume_change_2 = ask_sub_volume_2
        if bid_sub_price_2 < 0:
            bid_volume_change_2 = 0
        if bid_sub_price_2 > 0:
            bid_volume_change_2 = feat_dict['wa'][-1]
        if ask_sub_price_2 > 0:
            ask_volume_change_2 = 0
        if ask_sub_price_2 < 0:
            ask_volume_change_2 = feat_dict['wb'][-1]
        voi2 = (bid_volume_change_2 - ask_volume_change_2) / feat_dict['cum_size'][-1]
        feat_dict['voi2'] = np.append(feat_dict['voi2'], voi2)
        sum_num_ = test_data_consistency(feat_dict['voi2'][-1], 'voi2',
                                         df, index, sum_num_, df_2d_, col_dict)

    # mpb
    if len(feat_dict['ask_price1']) > 1:
        tp = feat_dict['turnover'][-1] / feat_dict['cum_size'][-1]
        # tp[np.isinf(tp)] = np.nan
        # if tp is None:
        #     tp = tp[-2]
        # else:
        #     tp = tp
        mid_last = (feat_dict['bid_price1'][-2] + feat_dict['ask_price1'][-2]) / 2
        mid = (feat_dict['bid_price1'][-1] + feat_dict['ask_price1'][-1]) / 2
        mpb = tp - (mid + mid_last) / 1000 / 2
        feat_dict['mpb'] = np.append(feat_dict['mpb'], mpb)
        sum_num_ = test_data_consistency(feat_dict['mpb'][-1], 'mpb',
                                         df, index, sum_num_, df_2d_, col_dict)

    # slope
    slope = (df[col_dict['ask_price1']] - df[col_dict['bid_price1']]) / (
                df[col_dict['ask_size1']] + df[col_dict['bid_size1']]) * 2
    feat_dict['slope'] = np.append(feat_dict['slope'], slope)
    sum_num_ = test_data_consistency(feat_dict['slope'][-1], 'slope',
                                     df, index, sum_num_, df_2d_, col_dict)

    # price_weighted_pressure
    kws = {}
    n1 = kws.setdefault("n1", 1)
    n2 = kws.setdefault("n2", 5)

    bench = kws.setdefault("bench_type", "MID")
    _ = np.arange(n1, n2 + 1)
    if bench == "MID":
        bench_prices = df[col_dict['ask_price1']] + df[col_dict['bid_price1']]
    elif bench == "SPECIFIC":
        bench_prices = kws.get("bench_price")
    else:
        raise Exception("")
    bid_d = [bench_prices / (bench_prices - df[col_dict['bid_price%s' % s]]) for s in _]
    bid_denominator = np.sum(bid_d)
    # bid_weights = [(d / bid_denominator).replace(np.nan,1) for d in bid_d]
    bid_weights = np.array([])
    for d in bid_d:
        if d / bid_denominator == np.nan:
            bid_weights = np.append(bid_weights, 1)
        else:
            bid_weights = np.append(bid_weights, d / bid_denominator)
    press_buy = np.sum([df[col_dict["bid_size%s" % (i + 1)]] * w for i, w in enumerate(bid_weights)])
    ask_d = [bench_prices / (df[col_dict['ask_price%s' % s]] - bench_prices) for s in _]
    ask_denominator = np.sum(ask_d)
    ask_weights = [d / ask_denominator for d in ask_d]
    press_sell = sum([df[col_dict['ask_size%s' % (i + 1)]] * w for i, w in enumerate(ask_weights)])
    price_weighted_pressure = np.log(press_buy) - np.log(press_sell)
    if price_weighted_pressure == np.inf or price_weighted_pressure == -np.inf:
        feat_dict['price_weighted_pressure'] = np.append(feat_dict['price_weighted_pressure'], np.nan)
    else:
        feat_dict['price_weighted_pressure'] = np.append(feat_dict['price_weighted_pressure'], price_weighted_pressure)
    sum_num_ = test_data_consistency(feat_dict['price_weighted_pressure'][-1], 'price_weighted_pressure',
                                     df, index, sum_num_, df_2d_, col_dict)

    # volume_order_imbalance
    if len(feat_dict['bid_price1']) > 1:
        kws = {}
        drop_first = kws.setdefault("drop_first", True)
        current_bid_price = feat_dict['bid_price1'][-1]
        bid_price_diff = current_bid_price - feat_dict['bid_price1'][shift_(1)]
        current_bid_vol = feat_dict['bid_size1'][-1]
        bvol_diff = current_bid_vol - feat_dict['bid_size1'][shift_(1)]
        bid_increment = current_bid_vol if bid_price_diff > 0 else (
            0 if bid_price_diff < 0 else (bvol_diff if bid_price_diff == 0 else bid_price_diff))
        current_ask_price = feat_dict['ask_price1'][-1]
        ask_price_diff = current_ask_price - feat_dict['ask_price1'][shift_(1)]
        current_ask_vol = feat_dict['ask_size1'][-1]
        avol_diff = current_ask_vol - feat_dict['ask_size1'][shift_(1)]
        ask_increment = current_ask_vol if ask_price_diff < 0 else (
            0 if ask_price_diff > 0 else (avol_diff if ask_price_diff == 0 else ask_price_diff))
        _ = bid_increment - ask_increment

        feat_dict['volume_order_imbalance'] = np.append(feat_dict['volume_order_imbalance'], _)
        sum_num_ = test_data_consistency(feat_dict['volume_order_imbalance'][-1], 'volume_order_imbalance',
                                         df, index, sum_num_, df_2d_, col_dict)

    # get_mid_price_change
    if len(feat_dict['ask_price1']) > 1:
        mid_last = (feat_dict['ask_price1'][-2] + feat_dict['bid_price1'][-2]) / 2
        mid = (feat_dict['ask_price1'][-1] + feat_dict['bid_price1'][-1]) / 2
        get_mid_price_change = mid / mid_last - 1
        feat_dict['get_mid_price_change'] = np.append(feat_dict['get_mid_price_change'], get_mid_price_change)
        sum_num_ = test_data_consistency(feat_dict['get_mid_price_change'][-1], 'get_mid_price_change',
                                         df, index, sum_num_, df_2d_, col_dict)

    return sum_num_


if __name__ == "__main__":
    # pass
    from minio import get_data_from_minio
    import time

    # minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
    # secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
    # symbol = 'btcusdt'
    symbol_list = 'ethusdt'
    platform = 'gate_swap_u'
    start_time = '2022-09-01-0'
    end_time = '2022-09-10-12'


    def get_data(platform, symbol_list, start_time, end_time):

        def cumsum(df):
            df['cum_size'] = np.cumsum(abs(df['size']))
            df['turnover'] = np.cumsum(df['price'] * abs(df['size']))
            return df

        depth = get_data_from_minio('gate_swap_u', symbol_list, 'datafile/tick/order_book_100ms/gate_swap_u',
                                    start_time=start_time, end_time=end_time)
        depth = depth.iloc[:, 2:-6]
        depth = depth.sort_values(by='closetime', ascending=True)

        trade = get_data_from_minio('gate_swap_u', symbol_list, 'datafile/tick/trade/gate_swap_u',
                                    index_name='timestamp',
                                    start_time=start_time, end_time=end_time)
        trade = trade.iloc[:, :-3]
        trade = trade.sort_values(by='dealid', ascending=True)
        trade = trade.rename({'timestamp': 'closetime'}, axis='columns')
        trade = trade.loc[:, ['closetime', 'price', 'size']]
        trade['datetime'] = pd.to_datetime(trade['closetime'] + 28800000, unit='ms')
        trade = trade.set_index('datetime').groupby(pd.Grouper(freq='1D')).apply(cumsum)
        trade = trade[
            (trade['closetime'] >= depth['closetime'].iloc[0]) & (trade['closetime'] <= depth['closetime'].iloc[-1])]
        trade = trade.reset_index(drop=True)
        data_merge = pd.merge(depth, trade, how='outer', on='closetime')
        data_merge.sort_values(by='closetime', ascending=True, inplace=True)
        data_merge['datetime'] = pd.to_datetime(data_merge['closetime'] + 28800000, unit='ms')
        data = data_merge.set_index('datetime').groupby(pd.Grouper(freq='1000ms')).apply('last')

        return data


    # tick_1s = get_data(platform=platform, symbol_list=symbol_list, start_time=start_time, end_time=end_time)
    data_agg = get_data_from_minio('gate_swap_u', symbol_list, 'datafile/feat/songhe/', index_name='closetime',
                                   start_time=start_time, end_time=end_time)

    data_agg.sort_values(by='closetime', ascending=True, inplace=True)
    data_agg.drop(['platform', 'year', 'month', 'symbol'], axis=1, inplace=True)

    # 这个是聚合的深度和订单流的数据
    ori_list = data_agg.columns.to_list()
    # 这个是聚合的深度和订单流的表头的列表
    col_dict_ = {key: i for i, key in enumerate(ori_list)}  # 将列表中的列名的下标作为值 列名做为键
    # 这个是因子的列表
    feat_dict_ = {i: np.array([]) for i in ori_list}
    agg_values = data_agg.values
    t1 = time.time()
    sum_num = 0
    df_2d = np.atleast_2d(np.zeros(66))
    for i in range(1000):
        sum_num = 0
        # test_columns = []
        # print(i)
        sum_num = factor_calculation(agg_values[i], i, feat_dict_, sum_num, df_2d, col_dict_)
    print(time.time() - t1)
