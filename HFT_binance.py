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
from tz_ctastrategy import (
    BarData,
)


sum_num = 0


cols_list = ['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1',
                          'ask_price2', 'ask_size2', 'bid_price2', 'bid_size2', 
                          'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3', 
                          'ask_price4', 'ask_size4', 'bid_price4', 'bid_size4', 
                          'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5', 
                          'ask_price6', 'ask_size6', 'bid_price6', 'bid_size6',
                          'ask_price7', 'ask_size7', 'bid_price7', 'bid_size7', 
                          'ask_price8', 'ask_size8', 'bid_price8', 'bid_size8', 
                          'ask_price9', 'ask_size9', 'bid_price9', 'bid_size9', 
                          'ask_price10', 'ask_size10','bid_price10','bid_size10',
                          'price', 'size', 'turnover', 'cum_size','ask_age',
                          'bid_age', 'inf_ratio', 'arrive_rate', 'depth_price_range', 'bp_rank',
                          'ap_rank', 'price_impact', 'depth_price_skew', 'depth_price_kurt',
                          'rolling_return', 'buy_increasing', 'sell_increasing', 'price_idxmax',
                          'center_deri_two', 'quasi', 'last_range','avg_spread', 'avg_turnover', 'abs_volume_kurt', 'abs_volume_skew',
                          'volume_kurt', 'volume_skew', 'price_kurt', 'price_skew','bv_divide_tn', 'av_divide_tn', 'weighted_price_to_mid',
                           'ask_withdraws', 'bid_withdraws', 'z_t', 'voi', 'voi2', 'wa', 'wb',
                           'slope', 'mpb', 'price_weighted_pressure', 'volume_order_imbalance','get_mid_price_change','mpb_500', 'positive_buying', 'positive_selling',
                           'buying_amplification_ratio', 'buying_amount_ratio', 'buying_willing', 'buying_willing_strength', 'buying_amount_strength', 'selling_ratio',
                           'buy_price_bias_level1', 'buy_amount_agg_ratio_level1', 'buy_price_bias_level2', 'buy_amount_agg_ratio_level2',
                           'sell_price_bias_level1', 'sell_amount_agg_ratio_level1','sell_price_bias_level2', 'sell_amount_agg_ratio_level2',
                           'posi_buy_cum_','caus_buy_cum_','posi_sell_cum_','caus_sell_cum_','amplify_biding','amplify_asking','posi_buy_turnover_','posi_sell_turnover_',
                           'buying_willing_strength_','buying_amount_strength_','bid_','ask_','buy_price_1','buy_amount_1','buy_price_2','buy_amount_2',
                           'sell_price_1','sell_amount_1','sell_price_2','sell_amount_2','vwap']





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
        
    


# def test_data_consistency(test_data: float, col_name: str, ori_data=np.array([]), sum_num_=0,
#                               df_2d_=np.atleast_2d(np.zeros(65))):
#         if test_data == 0:
#             if ori_data[col_dict[col_name]] == 0:
#                 dif_rate = 0
#             else:
#                 dif_rate = abs((test_data - ori_data[col_dict[col_name]]) / ori_data[col_dict[col_name]])
#         else:
#             # print('test_data:',test_data)
#             # print('ori_data:',ori_data[col_dict[col_name]])
#             dif_rate = abs((test_data - ori_data[col_dict[col_name]]) / test_data)
#         # if dif_rate > 0.0001 and round(test_data, 6) != round(ori_data[col_dict[col_name]], 6):
#         if dif_rate > 0.0001:
#             print('{}该数据批计算和实时流计算数值不一致---下标:{}---流数据:{}---批数据:{}'.format(col_name, index, test_data,
#                                                                          ori_data[col_dict[col_name]]))
#             # if col_name == 'depth_1s_buy_vwap_percentile_rolling_60':
#             #     print(col_name)

#         # df = df.replace(np.inf, 1)
#         # df = df.replace(-np.inf, -1)
        # if col_name in cols_list:
        #     col_name_index = cols_list.index(col_name)
        #     # 极大值和极小值强制赋值为 1 -1
        #     print('test_data-----------',test_data)
        #     print('col_name_index:',col_name_index)
        #     if np.isinf(test_data):
        #         df_2d_[0][col_name_index] = 1
        #     elif np.isneginf(test_data):
        #         df_2d_[0][col_name_index] = -1
        #     else:
        #         df_2d_[0][col_name_index] = test_data  # 组装入模的数据
        # test_columns.append(col_name)
        # return df_2d_

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


def _bid_withdraws_volume(l, n, levels=10):
    withdraws = 0
    for price_index in range(2, 2 + 4 * levels, 4):
        now_p = n[price_index]
        for price_last_index in range(2, 2 + 4 * levels, 4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

    return withdraws

def _ask_withdraws_volume(l, n, levels=10):
    withdraws = 0
    for price_index in range(0, 4 * levels, 4):
        now_p = n[price_index]
        for price_last_index in range(0, 4 * levels, 4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

    return withdraws




def factor_calculation(df: bytearray, feat_dict: dict, sum_num_: int, df_2d_, col_dict):
    
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




    def test_data_consistency(test_data: float, col_name: str, ori_data=np.array([]), sum_num_=0,
                              df_2d_=np.atleast_2d(np.zeros(65))):
        
#         if test_data == 0:
#             if ori_data[col_dict[col_name]] == 0:
#                 dif_rate = 0
#             else:
#                 dif_rate = abs((test_data - ori_data[col_dict[col_name]]) / ori_data[col_dict[col_name]])
#         else:
#             # print('test_data:',test_data)
#             # print('ori_data:',ori_data[col_dict[col_name]])
#             dif_rate = abs((test_data - ori_data[col_dict[col_name]]) / test_data)
#         # if dif_rate > 0.0001 and round(test_data, 6) != round(ori_data[col_dict[col_name]], 6):
#         if dif_rate > 0.0001:
#             print('{}该数据批计算和实时流计算数值不一致---下标:{}---流数据:{}---批数据:{}'.format(col_name, index, test_data,
#                                                                          ori_data[col_dict[col_name]]))
#             # if col_name == 'depth_1s_buy_vwap_percentile_rolling_60':
#             #     print(col_name)

#         # df = df.replace(np.inf, 1)
#         # df = df.replace(-np.inf, -1)
        if col_name in cols_list:
            col_name_index = cols_list.index(col_name)
            #极大值和极小值强制赋值为 1 -1
            # print('test_data-----------',test_data)
            # print('col_name_inde-------',col_name_index)
            # if np.isinf(test_data):
            #     df_2d_[0][col_name_index] = 1
            # elif np.isneginf(test_data):
            #     df_2d_[0][col_name_index] = -1
            # else:
            df_2d_[0][col_name_index] = test_data  # 组装入模的数据
        # test_columns.append(col_name)
        return df_2d_

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


    def _bid_withdraws_volume(l, n, levels=10):
        withdraws = 0
        for price_index in range(2, 2 + 4 * levels, 4):
            now_p = n[price_index]
            for price_last_index in range(2, 2 + 4 * levels, 4):
                if l[price_last_index] == now_p:
                    withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

        return withdraws

    def _ask_withdraws_volume(l, n, levels=10):
        withdraws = 0
        for price_index in range(0, 4 * levels, 4):
            now_p = n[price_index]
            for price_last_index in range(0, 4 * levels, 4):
                if l[price_last_index] == now_p:
                    withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

        return withdraws
    
    
    closetime = df[col_dict['closetime']]
    ask_price1 = df[col_dict['ask_price1']]
    ask_size1 = df[col_dict['ask_size1']]
    bid_price1 = df[col_dict['bid_price1']]
    bid_size1 = df[col_dict['bid_size1']]
    ask_price2 = df[col_dict['ask_price2']]
    ask_size2 = df[col_dict['ask_size2']]
    bid_price2 = df[col_dict['bid_price2']]
    bid_size2 = df[col_dict['bid_size2']]
    ask_price3 = df[col_dict['ask_price3']]
    ask_size3 = df[col_dict['ask_size3']]
    bid_price3 = df[col_dict['bid_price3']]
    bid_size3 = df[col_dict['bid_size3']]
    ask_price4 = df[col_dict['ask_price4']]
    ask_size4 = df[col_dict['ask_size4']]
    bid_price4 = df[col_dict['bid_price4']]
    bid_size4 = df[col_dict['bid_size4']]
    ask_price5 = df[col_dict['ask_price5']]
    ask_size5 = df[col_dict['ask_size5']]
    bid_price5 = df[col_dict['bid_price5']]
    bid_size5 = df[col_dict['bid_size5']]
    ask_price6 = df[col_dict['ask_price6']]
    ask_size6 = df[col_dict['ask_size6']]
    bid_price6 = df[col_dict['bid_price6']]
    bid_size6 = df[col_dict['bid_size6']]
    ask_price7 = df[col_dict['ask_price7']]
    ask_size7 = df[col_dict['ask_size7']]
    bid_price7 = df[col_dict['bid_price7']]
    bid_size7 = df[col_dict['bid_size7']]
    ask_price8= df[col_dict['ask_price8']]
    ask_size8 = df[col_dict['ask_size8']]
    bid_price8 = df[col_dict['bid_price8']]
    bid_size8 = df[col_dict['bid_size8']]
    ask_price9 = df[col_dict['ask_price9']]
    ask_size9 = df[col_dict['ask_size9']]
    bid_price9 = df[col_dict['bid_price9']]
    bid_size9 = df[col_dict['bid_size9']]
    ask_price10 = df[col_dict['ask_price10']]
    ask_size10 = df[col_dict['ask_size10']]
    bid_price10 = df[col_dict['bid_price10']]
    bid_size10 = df[col_dict['bid_size10']]
    price = df[col_dict['price']]
    size = df[col_dict['size']]
    cum_size = df[col_dict['cum_size']]
    turnover = df[col_dict['turnover']]
    
    feat_dict['closetime'] = np.append(feat_dict['closetime'], closetime)
    feat_dict['closetime'] = feat_dict['closetime'][-2000:]
    feat_dict['ask_price1'] = np.append(feat_dict['ask_price1'], ask_price1)
    feat_dict['ask_price1'] = feat_dict['ask_price1'][-2000:]
    feat_dict['bid_price1'] = np.append(feat_dict['bid_price1'], bid_price1)
    feat_dict['bid_price1'] = feat_dict['bid_price1'][-2000:]
    feat_dict['ask_price2'] = np.append(feat_dict['ask_price2'], ask_price2)
    feat_dict['ask_price2'] = feat_dict['ask_price2'][-2000:]
    feat_dict['bid_price2'] = np.append(feat_dict['bid_price2'], bid_price2)
    feat_dict['bid_price2'] = feat_dict['bid_price2'][-2000:]
    feat_dict['ask_price3'] = np.append(feat_dict['ask_price3'], ask_price3)
    feat_dict['ask_price3'] = feat_dict['ask_price3'][-2000:]
    feat_dict['bid_price3'] = np.append(feat_dict['bid_price3'], bid_price3)
    feat_dict['bid_price3'] = feat_dict['bid_price3'][-2000:]
    feat_dict['ask_price4'] = np.append(feat_dict['ask_price4'], ask_price4)
    feat_dict['ask_price4'] = feat_dict['ask_price4'][-2000:]
    feat_dict['bid_price4'] = np.append(feat_dict['bid_price4'], bid_price4)
    feat_dict['bid_price4'] = feat_dict['bid_price4'][-2000:]
    feat_dict['ask_price5'] = np.append(feat_dict['ask_price5'], ask_price5)
    feat_dict['ask_price5'] = feat_dict['ask_price5'][-2000:]
    feat_dict['bid_price5'] = np.append(feat_dict['bid_price5'], bid_price5)
    feat_dict['bid_price5'] = feat_dict['bid_price5'][-2000:]
    feat_dict['ask_price6'] = np.append(feat_dict['ask_price6'], ask_price6)
    feat_dict['ask_price6'] = feat_dict['ask_price6'][-2000:]
    feat_dict['bid_price6'] = np.append(feat_dict['bid_price6'], bid_price6)
    feat_dict['bid_price6'] = feat_dict['bid_price6'][-2000:]
    feat_dict['ask_price7'] = np.append(feat_dict['ask_price7'], ask_price7)
    feat_dict['ask_price7'] = feat_dict['ask_price7'][-2000:]
    feat_dict['bid_price7'] = np.append(feat_dict['bid_price7'], bid_price7)
    feat_dict['bid_price7'] = feat_dict['bid_price7'][-2000:]
    feat_dict['ask_price8'] = np.append(feat_dict['ask_price8'], ask_price8)
    feat_dict['ask_price8'] = feat_dict['ask_price8'][-2000:]
    feat_dict['bid_price8'] = np.append(feat_dict['bid_price8'], bid_price8)
    feat_dict['bid_price8'] = feat_dict['bid_price8'][-2000:]
    feat_dict['ask_price9'] = np.append(feat_dict['ask_price9'], ask_price9)
    feat_dict['ask_price9'] = feat_dict['ask_price9'][-2000:]
    feat_dict['bid_price9'] = np.append(feat_dict['bid_price9'], bid_price9)
    feat_dict['bid_price9'] = feat_dict['bid_price9'][-2000:]
    feat_dict['ask_price10'] = np.append(feat_dict['ask_price10'], ask_price10)
    feat_dict['ask_price10'] = feat_dict['ask_price10'][-2000:]
    feat_dict['bid_price10'] = np.append(feat_dict['bid_price10'], bid_price10)
    feat_dict['bid_price10'] = feat_dict['bid_price10'][-2000:]
    feat_dict['ask_size1'] = np.append(feat_dict['ask_size1'], ask_size1)
    feat_dict['ask_size1'] = feat_dict['ask_size1'][-2000:]
    feat_dict['bid_size1'] = np.append(feat_dict['bid_size1'], bid_size1)
    feat_dict['bid_size1'] = feat_dict['bid_size1'][-2000:]
    feat_dict['ask_size2'] = np.append(feat_dict['ask_size2'], ask_size2)
    feat_dict['ask_size2'] = feat_dict['ask_size2'][-2000:]
    feat_dict['bid_size2'] = np.append(feat_dict['bid_size2'], bid_size2)
    feat_dict['bid_size2'] = feat_dict['bid_size2'][-2000:]
    feat_dict['ask_size3'] = np.append(feat_dict['ask_size3'], ask_size3)
    feat_dict['ask_size3'] = feat_dict['ask_size3'][-2000:]
    feat_dict['bid_size3'] = np.append(feat_dict['bid_size3'], bid_size3)
    feat_dict['bid_size3'] = feat_dict['bid_size3'][-2000:]
    feat_dict['ask_size4'] = np.append(feat_dict['ask_size4'], ask_size4)
    feat_dict['ask_size4'] = feat_dict['ask_size4'][-2000:]
    feat_dict['bid_size4'] = np.append(feat_dict['bid_size4'], bid_size4)
    feat_dict['bid_size4'] = feat_dict['bid_size4'][-2000:]
    feat_dict['ask_size5'] = np.append(feat_dict['ask_size5'], ask_size5)
    feat_dict['ask_size5'] = feat_dict['ask_size5'][-2000:]
    feat_dict['bid_size5'] = np.append(feat_dict['bid_size5'], bid_size5)
    feat_dict['bid_size5'] = feat_dict['bid_size5'][-2000:]
    feat_dict['ask_size6'] = np.append(feat_dict['ask_size6'], ask_size6)
    feat_dict['ask_size6'] = feat_dict['ask_size6'][-2000:]
    feat_dict['bid_size6'] = np.append(feat_dict['bid_size6'], bid_size6)
    feat_dict['bid_size6'] =  feat_dict['bid_size6'][-2000:]
    feat_dict['ask_size7'] = np.append(feat_dict['ask_size7'], ask_size7)
    feat_dict['ask_size7'] = feat_dict['ask_size7'][-2000:]
    feat_dict['bid_size7'] = np.append(feat_dict['bid_size7'], bid_size7)
    feat_dict['bid_size7'] = feat_dict['bid_size7'][-2000:]
    feat_dict['ask_size8'] = np.append(feat_dict['ask_size8'], ask_size8)
    feat_dict['ask_size8'] = feat_dict['ask_size8'][-2000:]
    feat_dict['bid_size8'] = np.append(feat_dict['bid_size8'], bid_size8)
    feat_dict['bid_size8'] = feat_dict['bid_size8'][-2000:]
    feat_dict['ask_size9'] = np.append(feat_dict['ask_size9'], ask_size9)
    feat_dict['ask_size9'] = feat_dict['ask_size9'][-2000:]
    feat_dict['bid_size9'] = np.append(feat_dict['bid_size9'], bid_size9)
    feat_dict['bid_size9'] = feat_dict['bid_size9'][-2000:]
    feat_dict['ask_size10'] = np.append(feat_dict['ask_size10'], ask_size10)
    feat_dict['ask_size10'] = feat_dict['ask_size10'][-2000:]
    feat_dict['bid_size10'] = np.append(feat_dict['bid_size10'], bid_size10)
    feat_dict['bid_size10'] = feat_dict['bid_size10'][-2000:]
    feat_dict['price'] = np.append(feat_dict['price'], price)
    feat_dict['price'] = feat_dict['price'][-2000:]
    feat_dict['size'] = np.append(feat_dict['size'], size)
    feat_dict['size'] = feat_dict['size'][-2000:]
    feat_dict['cum_size'] = np.append(feat_dict['cum_size'], cum_size)
    feat_dict['cum_size'] = feat_dict['cum_size'][-2000:]
    feat_dict['turnover'] = np.append(feat_dict['turnover'], turnover)
    feat_dict['turnover'] = feat_dict['turnover'][-2000:]
    
    
    df_2d_ = test_data_consistency(feat_dict['closetime'][-1], 'closetime',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_price1'][-1], 'ask_price1',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_price1'][-1], 'bid_price1',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_price2'][-1], 'ask_price2',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_price2'][-1], 'bid_price2',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_price3'][-1], 'ask_price3',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_price3'][-1], 'bid_price3',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_price4'][-1], 'ask_price4',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_price4'][-1], 'bid_price4',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_price5'][-1], 'ask_price5',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_price5'][-1], 'bid_price5',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_price6'][-1], 'ask_price6',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_price6'][-1], 'bid_price6',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_price7'][-1], 'ask_price7',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_price7'][-1], 'bid_price7',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_price8'][-1], 'ask_price8',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_price8'][-1], 'bid_price8',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_price9'][-1], 'ask_price9',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_price9'][-1], 'bid_price9',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_price10'][-1], 'ask_price10',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_price10'][-1], 'bid_price10',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_size1'][-1], 'ask_size1',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_size1'][-1], 'bid_size1',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_size2'][-1], 'ask_size2',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_size2'][-1], 'bid_size2',
                                     df,  sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_size3'][-1], 'ask_size3',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_size3'][-1], 'bid_size3',
                                     df,  sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_size4'][-1], 'ask_size4',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_size4'][-1], 'bid_size4',
                                     df,  sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_size5'][-1], 'ask_size5',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_size5'][-1], 'bid_size5',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_size6'][-1], 'ask_size6',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_size6'][-1], 'bid_size6',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_size7'][-1], 'ask_size7',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_size7'][-1], 'bid_size7',
                                     df,  sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_size8'][-1], 'ask_size8',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_size8'][-1], 'bid_size8',
                                     df,  sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_size9'][-1], 'ask_size9',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_size9'][-1], 'bid_size9',
                                     df,  sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['ask_size10'][-1], 'ask_size10',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['bid_size10'][-1], 'bid_size10',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['price'][-1], 'price',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['size'][-1], 'size',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['cum_size'][-1], 'cum_size',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    df_2d_ = test_data_consistency(feat_dict['turnover'][-1], 'turnover',
                                     df, sum_num_, df_2d_)  # 数据一致性校验 pass
    # bid_age
    if len(feat_dict['bid_price1'])>10:
        # bp1_changes = bp1.rolling(rolling).apply(get_age, engine='numba', raw=True).fillna(0)
        bp1_changes = get_age(feat_dict['bid_price1'][-10:])
        if str(bp1_changes) == 'nan':
            feat_dict['bid_age'] = np.append(feat_dict['bid_age'],0)
        else:
            feat_dict['bid_age'] = np.append(feat_dict['bid_age'],bp1_changes)
        feat_dict['bid_age'] = feat_dict['bid_age'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['bid_age'][-1], 'bid_age',
                                      df, sum_num_, df_2d_)  # 数据一致性校验 pass
    
    # ask_age
    if len(feat_dict['ask_price1'])>10:
        
        ap1_changes = get_age(feat_dict['ask_price1'][-10:])
        if str(ap1_changes) == 'nan':
            feat_dict['ask_age'] = np.append(feat_dict['ask_age'],0)
        else:
            feat_dict['ask_age'] = np.append(feat_dict['ask_age'],ap1_changes)
        feat_dict['ask_age'] = feat_dict['ask_age'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['ask_age'][-1], 'ask_age',
                                      df, sum_num_, df_2d_)  # 数据一致性校验 pass
        
    # inf_ratio
    if len(feat_dict['price'])>100:
        inf_price = feat_dict['price'].copy()
        inf_price = np.where(np.isnan(inf_price),0,inf_price)
        quasi = np.nansum(np.abs(np.diff(inf_price)[-100:]))
        # quasi = np.nansum((np.abs(feat_dict['price'][-1]-feat_dict['price'][shift_(1)]))[-100:])
        # dif = np.abs(np.diff(inf_price[-100:]))
        dif = np.abs(inf_price[-1]-inf_price[shift_(100)])
        quasi = np.where(np.isnan(quasi),10,quasi)
        dif= np.where(np.isnan(dif),10,dif)
        feat_dict['inf_ratio'] = np.append(feat_dict['inf_ratio'], quasi/(dif+quasi))
        feat_dict['inf_ratio'] = feat_dict['inf_ratio'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['inf_ratio'][-1], 'inf_ratio',
                                      df, sum_num_, df_2d_)
        
    # depth_price_range
    if len(feat_dict['ask_price1'])>100:
        depth_price_range = ((np.max(feat_dict['ask_price1'][-100:]))/(np.min(feat_dict['ask_price1'][-100:])))-1
        if str(depth_price_range) == 'nan':
            feat_dict['depth_price_range'] = np.append(feat_dict['depth_price_range'], 0)
        else:
            feat_dict['depth_price_range'] = np.append(feat_dict['depth_price_range'], depth_price_range)
        feat_dict['depth_price_range'] = feat_dict['depth_price_range'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['depth_price_range'][-1], 'depth_price_range',
                                          df, sum_num_, df_2d_)
    
    # arrive_rate
    if len(feat_dict['closetime'])>300:
        res = feat_dict['closetime'][-1]-feat_dict['closetime'][shift_(300)]
        if str(res) == 'nan':
            res = 0
        else:
            res = res
        feat_dict['arrive_rate'] = np.append(feat_dict['arrive_rate'], res/300)
        feat_dict['arrive_rate'] = feat_dict['arrive_rate'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['arrive_rate'][-1], 'arrive_rate',
                                          df, sum_num_, df_2d_)
    
    # bp_rank
    if len(feat_dict['bid_price1'])>100:
        # bp_rank = feat_dict['bid_price1'][-100:].argsort().argsort()
        bp_rank = np.empty_like(feat_dict['bid_price1'])
        for i in range(len(feat_dict['bid_price1'])):
            b_window = feat_dict['bid_price1'][max(0, i-100+1):(i+1)]
            bp_rank[i] = (np.sum(b_window < feat_dict['bid_price1'][i]) + 0.5*np.sum(b_window == feat_dict['bid_price1'][i])) / len(b_window)
        if str(bp_rank) == 'nan' or abs(bp_rank[-1])< 0.00000001:
            feat_dict['bp_rank'] = np.append(feat_dict['bp_rank'], 0)
        else:
            # feat_dict['bp_rank'] = np.append(feat_dict['bp_rank'], bp_rank/100*2-1)
            feat_dict['bp_rank'] = np.append(feat_dict['bp_rank'], bp_rank[-1]*2-1+0.01)
        feat_dict['bp_rank'] = feat_dict['bp_rank'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['bp_rank'][-1], 'bp_rank',
                                      df, sum_num_, df_2d_)
    
    # ap_rank
    if len(feat_dict['ask_price1'])>100:
        # ap_rank = feat_dict['ask_price1'][-100:].argsort().argsort()
        ap_rank = np.empty_like(feat_dict['ask_price1'])
        for i in range(len(feat_dict['ask_price1'])):
            a_window = feat_dict['ask_price1'][max(0, i-100+1):(i+1)]
            ap_rank[i] = (np.sum(a_window < feat_dict['ask_price1'][i]) + 0.5*np.sum(a_window == feat_dict['ask_price1'][i])) / len(a_window)
        if str(ap_rank) == 'nan' or abs(ap_rank[-1])< 0.00000001:
            feat_dict['ap_rank'] = np.append(feat_dict['ap_rank'], 0)
        else:
            # feat_dict['ap_rank'] = np.append(feat_dict['ap_rank'], ap_rank/100*2-1)
            feat_dict['ap_rank'] = np.append(feat_dict['ap_rank'], ap_rank[-1]*2-1+0.01)
        feat_dict['ap_rank'] = feat_dict['ap_rank'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['ap_rank'][-1], 'ap_rank',
                                      df, sum_num_, df_2d_)
    
    # price_impact
    ask, bid, ask_v, bid_v = 0, 0, 0, 0
    # for i in range(1, 6):
    ask = df[col_dict['ask_price1']] * df[col_dict['ask_size1']] + df[col_dict['ask_price2']] * df[col_dict['ask_size2']] +\
          df[col_dict['ask_price3']] * df[col_dict['ask_size3']] + df[col_dict['ask_price4']] * df[col_dict['ask_size4']] +\
          df[col_dict['ask_price5']] * df[col_dict['ask_size5']] + df[col_dict['ask_price6']] * df[col_dict['ask_size6']] +\
          df[col_dict['ask_price7']] * df[col_dict['ask_size7']] + df[col_dict['ask_price8']] * df[col_dict['ask_size8']] +\
          df[col_dict['ask_price9']] * df[col_dict['ask_size9']] + df[col_dict['ask_price10']] * df[col_dict['ask_size10']]
    bid = df[col_dict['bid_price1']] * df[col_dict['bid_size1']] + df[col_dict['bid_price2']] * df[col_dict['bid_size2']] +\
          df[col_dict['bid_price3']] * df[col_dict['bid_size3']] + df[col_dict['bid_price4']] * df[col_dict['bid_size4']] +\
          df[col_dict['bid_price5']] * df[col_dict['bid_size5']] + df[col_dict['bid_price6']] * df[col_dict['bid_size6']] +\
          df[col_dict['bid_price7']] * df[col_dict['bid_size7']] + df[col_dict['bid_price8']] * df[col_dict['bid_size8']] +\
          df[col_dict['bid_price9']] * df[col_dict['bid_size9']] + df[col_dict['bid_price10']] * df[col_dict['bid_size10']]
    ask_v = df[col_dict['ask_size1']]+df[col_dict['ask_size2']]+df[col_dict['ask_size3']]+df[col_dict['ask_size4']]+df[col_dict['ask_size5']]+\
            df[col_dict['ask_size6']]+df[col_dict['ask_size7']]+df[col_dict['ask_size8']]+df[col_dict['ask_size9']]+df[col_dict['ask_size10']]
    bid_v = df[col_dict['bid_size1']]+df[col_dict['bid_size2']]+df[col_dict['bid_size3']]+df[col_dict['bid_size4']]+df[col_dict['bid_size5']] +\
            df[col_dict['bid_size6']]+df[col_dict['bid_size7']]+df[col_dict['bid_size8']]+df[col_dict['bid_size9']]+df[col_dict['bid_size10']]
    ask /= ask_v
    bid /= bid_v
    price_impact = -(df[col_dict['ask_price1']] - ask) / df[col_dict['ask_price1']] - (df[col_dict['bid_price1']]- bid)/df[col_dict['bid_price1']]
    feat_dict['price_impact'] = np.append(feat_dict['price_impact'], price_impact)
    feat_dict['price_impact'] = feat_dict['price_impact'][-2000:]
    df_2d_ = test_data_consistency(feat_dict['price_impact'][-1], 'price_impact',
                                      df, sum_num_, df_2d_)
    
    # depth_price_skew
    # if len(feat_dict['ask_price1'])>1:
    #     price_skew = np.hstack((feat_dict['bid_price10'][-1], feat_dict['bid_price9'][-1], feat_dict['bid_price8'][-1], feat_dict['bid_price7'][-1], feat_dict['bid_price6'][-1], feat_dict['bid_price5'][-1], feat_dict['bid_price4'][-1], feat_dict['bid_price3'][-1], feat_dict['bid_price2'][-1],feat_dict['bid_price1'][-1],
    #                             feat_dict['ask_price5'][-1], feat_dict['ask_price4'][-1], feat_dict['ask_price3'][-1], feat_dict['ask_price2'][-1],feat_dict['ask_price1'][-1])) 
    price_skew = np.hstack((df[col_dict['bid_price10']], df[col_dict['bid_price9']], df[col_dict['bid_price8']], 
                            df[col_dict['bid_price7']],df[col_dict['bid_price6']],
                            df[col_dict['bid_price5']], df[col_dict['bid_price4']], df[col_dict['bid_price3']], 
                            df[col_dict['bid_price2']],df[col_dict['bid_price1']],
                            df[col_dict['ask_price10']], df[col_dict['ask_price9']], df[col_dict['ask_price8']], 
                            df[col_dict['ask_price7']],df[col_dict['ask_price6']],
                            df[col_dict['ask_price5']], df[col_dict['ask_price4']], df[col_dict['ask_price3']], 
                            df[col_dict['ask_price2']],df[col_dict['ask_price1']])) 
    depth_price_skew = skew_(price_skew)
    feat_dict['depth_price_skew'] = np.append(feat_dict['depth_price_skew'], depth_price_skew)
    feat_dict['depth_price_skew'] = feat_dict['depth_price_skew'][-2000:]
    df_2d_ = test_data_consistency(feat_dict['depth_price_skew'][-1], 'depth_price_skew',
                                      df, sum_num_, df_2d_)
    
    # depth_price_kurt
    price_kurt = np.hstack((df[col_dict['bid_price10']], df[col_dict['bid_price9']], df[col_dict['bid_price8']], 
                            df[col_dict['bid_price7']],df[col_dict['bid_price6']],
                            df[col_dict['bid_price5']], df[col_dict['bid_price4']], df[col_dict['bid_price3']], 
                            df[col_dict['bid_price2']],df[col_dict['bid_price1']],
                            df[col_dict['ask_price10']], df[col_dict['ask_price9']], df[col_dict['ask_price8']], 
                            df[col_dict['ask_price7']],df[col_dict['ask_price6']],
                            df[col_dict['ask_price5']], df[col_dict['ask_price4']], df[col_dict['ask_price3']], 
                            df[col_dict['ask_price2']],df[col_dict['ask_price1']])) 
    depth_price_kurt = kurt_(price_kurt)
    feat_dict['depth_price_kurt'] = np.append(feat_dict['depth_price_kurt'], depth_price_kurt)
    feat_dict['depth_price_kurt'] = feat_dict['depth_price_kurt'][-2000:]
    df_2d_ = test_data_consistency(feat_dict['depth_price_kurt'][-1], 'depth_price_kurt',
                                      df, sum_num_, df_2d_)
    
    # rolling_return
    if len(feat_dict['ask_price1']) >100 and len(feat_dict['bid_price1'])>100:
        mp = (feat_dict['ask_price1'][-1]+feat_dict['bid_price1'][-1])/2
        mp_100 = (feat_dict['ask_price1'][shift_(100)]+feat_dict['bid_price1'][shift_(100)])/2
        rolling_return = (mp-mp_100)/mp
        if str(rolling_return) == 'nan':
            feat_dict['rolling_return'] = np.append(feat_dict['rolling_return'], 0)
        else:
            feat_dict['rolling_return'] = np.append(feat_dict['rolling_return'], rolling_return)
        feat_dict['rolling_return'] = feat_dict['rolling_return'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['rolling_return'][-1], 'rolling_return',
                                          df, sum_num_, df_2d_)
    
    # buy_increasing
    if len(feat_dict['size'])>200:
        b_v = feat_dict['size'].copy()
        b_v[b_v<0] = 0
        # if v[-1] <0:
        #     v[-1] = 0
        # else:
        #     v[-1] = v[-1]
        b_v = np.where(np.isnan(b_v),0,b_v)      
        # x = pd.DataFrame(b_v[-200:],len(feat_dict['size']))
        # x.to_csv('b_v.csv')
        buy_increasing = np.log1p((np.sum(b_v[-100*2:])+1)/(np.sum(b_v[-100:])+1))
        if str(buy_increasing) == 'nan':
            feat_dict['buy_increasing'] = np.append(feat_dict['buy_increasing'],1)
        else:
            feat_dict['buy_increasing'] = np.append(feat_dict['buy_increasing'], buy_increasing)
        feat_dict['buy_increasing'] = feat_dict['buy_increasing'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['buy_increasing'][-1], 'buy_increasing',
                                          df, sum_num_, df_2d_)
    
     # sell_increasing
    if len(feat_dict['size'])>100:
        s_v = feat_dict['size'].copy()
        s_v[s_v>0] = 0
        # if v[-1] >0:
        #     v[-1] = 0
        # else:
        #     v[-1] = v[-1]
        s_v = np.where(np.isnan(s_v),0,s_v)
        sell_increasing = np.log1p((np.sum(s_v[-100*2:])-1)/(np.sum(s_v[-100:])-1))
        if str(sell_increasing) == 'nan':
            feat_dict['sell_increasing'] = np.append(feat_dict['sell_increasing'],1)
        else:
            feat_dict['sell_increasing'] = np.append(feat_dict['sell_increasing'], sell_increasing)
        feat_dict['sell_increasing'] = feat_dict['sell_increasing'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['sell_increasing'][-1], 'sell_increasing',
                                          df, sum_num_, df_2d_)
    
    
    # price_idxmax
    if len(feat_dict['ask_price1'])>20:
        price_idxmax = first_location_of_maximum(feat_dict['ask_price1'][-20:])
        if str(price_idxmax) == 'nan':
            feat_dict['price_idxmax'] = np.append(feat_dict['price_idxmax'], 0)
        else:
            feat_dict['price_idxmax'] = np.append(feat_dict['price_idxmax'], price_idxmax)
        feat_dict['price_idxmax'] = feat_dict['price_idxmax'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['price_idxmax'][-1], 'price_idxmax',
                                      df, sum_num_, df_2d_)
    
    
    # center_deri_two
    if len(feat_dict['ask_price1'])>20:
        center_deri_two = mean_second_derivative_centra(feat_dict['ask_price1'][-20:])
        if str(center_deri_two) == 'nan':
            feat_dict['center_deri_two'] = np.append(feat_dict['center_deri_two'], 0)
        else:
            feat_dict['center_deri_two'] = np.append(feat_dict['center_deri_two'], center_deri_two)
        feat_dict['center_deri_two'] = feat_dict['center_deri_two'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['center_deri_two'][-1], 'center_deri_two',
                                      df, sum_num_, df_2d_)
    
    # quasi
    if len(feat_dict['ask_price1'])>100:
        quasi = np.sum(np.abs(np.diff(feat_dict['ask_price1'])[-100:]))
        if str(quasi) == 'nan':
            feat_dict['quasi'] = np.append(feat_dict['quasi'], 0)
        else:
            feat_dict['quasi'] = np.append(feat_dict['quasi'], quasi)
        feat_dict['quasi'] = feat_dict['quasi'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['quasi'][-1], 'quasi',
                                      df, sum_num_, df_2d_)
    
    # last_range
    if len(feat_dict['price'])>100:
        price = feat_dict['price'].copy()
        price = np.where(np.isnan(price),0,price)
        diff = abs(np.diff(price))
        last_range = np.nansum(diff[-100:])
        if str(last_range) == 'nan':
            feat_dict['last_range'] = np.append(feat_dict['last_range'], 0)
        else:
            feat_dict['last_range'] = np.append(feat_dict['last_range'], last_range)
        feat_dict['last_range'] = feat_dict['last_range'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['last_range'][-1], 'last_range',
                                      df, sum_num_, df_2d_)
    
    # avg_trade_volume
    # print('------------',np.abs(feat_dict['size'][::-1]))
    # if len(feat_dict['size'])>100:
    #     print('True:',np.nansum((np.abs(feat_dict['size'][::-1])[-100:])))
    #     avg_trade_voume = np.sum(np.abs(feat_dict['size'][::-1])[-100:])[shift_(-100+1)]
    #     if avg_trade_volume is None:
    #         feat_dict['avg_trade_volume'] = np.append(feat_dict['avg_trade_volume'], 0)
    #     else:
    #         feat_dict['avg_trade_volume'] = np.append(feat_dict['avg_trade_volume'], avg_trade_volume[::-1])
    #     sum_num_ = test_data_consistency(feat_dict['avg_trade_volume'][-1], 'avg_trade_volume',
    #                                   df, index, sum_num_, df_2d_, col_dict)
    
    # avg_spread
    if len(feat_dict['ask_price1']) >200 and len(feat_dict['bid_price1'])>200:
        avg_spread = np.mean((feat_dict['ask_price1']-feat_dict['bid_price1'])[-200:])
        if str(avg_spread) == 'nan':
            feat_dict['avg_spread'] = np.append(feat_dict['avg_spread'], 0)
        else:
            feat_dict['avg_spread'] = np.append(feat_dict['avg_spread'], avg_spread)
        feat_dict['avg_spread'] = feat_dict['avg_spread'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['avg_spread'][-1], 'avg_spread',
                                      df, sum_num_, df_2d_)
        
    # avg_turnover
    avg_turnover = np.sum(np.hstack((df[col_dict['bid_size10']], df[col_dict['bid_size9']], df[col_dict['bid_size8']], 
                                     df[col_dict['bid_size7']],df[col_dict['bid_size6']],
                                     df[col_dict['bid_size5']], df[col_dict['bid_size4']], df[col_dict['bid_size3']], 
                                     df[col_dict['bid_size2']],df[col_dict['bid_size1']],
                                     df[col_dict['ask_size10']], df[col_dict['ask_size9']], df[col_dict['ask_size8']],   
                                     df[col_dict['ask_size7']],df[col_dict['ask_size6']],
                                     df[col_dict['ask_size5']], df[col_dict['ask_size4']], df[col_dict['ask_size3']],   
                                     df[col_dict['ask_size2']],df[col_dict['ask_size1']]))) 
    feat_dict['avg_turnover'] = np.append(feat_dict['avg_turnover'], avg_turnover)
    feat_dict['avg_turnover'] = feat_dict['avg_turnover'][-2000:]
    df_2d_ = test_data_consistency(feat_dict['avg_turnover'][-1], 'avg_turnover',
                                      df, sum_num_, df_2d_)
    
    # abs_volume_kurt
    if len(feat_dict['size'])>500:
        avk_size = feat_dict['size'].copy()
        avk_size = np.where(np.isnan(avk_size),0,avk_size)
        abs_volume_kurt = kurt_(np.abs(avk_size)[-500:])
        if str(abs_volume_kurt) == 'nan':
            feat_dict['abs_volume_kurt'] = np.append(feat_dict['abs_volume_kurt'], 0)
        else:
            feat_dict['abs_volume_kurt'] = np.append(feat_dict['abs_volume_kurt'], abs_volume_kurt)
        feat_dict['abs_volume_kurt'] = feat_dict['abs_volume_kurt'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['abs_volume_kurt'][-1], 'abs_volume_kurt',
                                      df, sum_num_, df_2d_)
    
    # abs_volume_skew
    if len(feat_dict['size'])>500:
        avs_size = feat_dict['size'].copy()
        avs_size = np.where(np.isnan(avs_size),0,avs_size)
        abs_volume_skew = skew_(np.abs(avs_size)[-500:])
        if str(abs_volume_skew) == 'nan':
            feat_dict['abs_volume_skew'] = np.append(feat_dict['abs_volume_skew'], 0)
        else:
            feat_dict['abs_volume_skew'] = np.append(feat_dict['abs_volume_skew'], abs_volume_skew)
        feat_dict['abs_volume_skew'] = feat_dict['abs_volume_skew'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['abs_volume_skew'][-1], 'abs_volume_skew',
                                      df, sum_num_, df_2d_)
    
    # volume_kurt
    if len(feat_dict['size'])>500:
        vk_size = feat_dict['size'].copy()
        vk_size = np.where(np.isnan(vk_size),0,vk_size)
        volume_kurt = kurt_(vk_size[-500:])
        if str(volume_kurt) == 'nan':
            feat_dict['volume_kurt'] = np.append(feat_dict['volume_kurt'], 0)
        else:
            feat_dict['volume_kurt'] = np.append(feat_dict['volume_kurt'], volume_kurt)
        feat_dict['volume_kurt'] = feat_dict['volume_kurt'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['volume_kurt'][-1], 'volume_kurt',
                                      df, sum_num_, df_2d_)
    
    # volume_skew
    if len(feat_dict['size'])>500:
        vs_size = feat_dict['size'].copy()
        vs_size = np.where(np.isnan(vs_size),0,vs_size)
        volume_skew = skew_(vs_size[-500:])
        if str(volume_skew) == 'nan':
            feat_dict['volume_skew'] = np.append(feat_dict['volume_skew'], 0)
        else:
            feat_dict['volume_skew'] = np.append(feat_dict['volume_skew'], volume_skew)
        feat_dict['volume_skew'] = feat_dict['volume_skew'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['volume_skew'][-1], 'volume_skew',
                                      df, sum_num_, df_2d_)
    
    # price_kurt
    if len(feat_dict['price'])>500:
        pk_price = feat_dict['price'].copy()
        pk_price = np.where(np.isnan(pk_price),0,pk_price)
        price_kurt = kurt_(pk_price[-500:])
        if str(price_kurt) == 'nan':
            feat_dict['price_kurt'] = np.append(feat_dict['price_kurt'], 0)
        else:
            feat_dict['price_kurt'] = np.append(feat_dict['price_kurt'], price_kurt)
        feat_dict['price_kurt'] = feat_dict['price_kurt'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['price_kurt'][-1], 'price_kurt',
                                      df, sum_num_, df_2d_)
        
    # price_skew
    if len(feat_dict['price'])>500:
        ps_price = feat_dict['price'].copy()
        ps_price = np.where(np.isnan(ps_price),0,ps_price)
        price_skew = np.abs(skew_(ps_price[-500:]))
        if str(price_skew) == 'nan':
            feat_dict['price_skew'] = np.append(feat_dict['price_skew'], 0)
        else:
            feat_dict['price_skew'] = np.append(feat_dict['price_skew'], price_skew)
        feat_dict['price_skew'] = feat_dict['price_skew'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['price_skew'][-1], 'price_skew',
                                      df, sum_num_, df_2d_)
    
    # bv_divide_tn
    if len(feat_dict['size'])>10:
        bvs = feat_dict['bid_size1'][-1]+feat_dict['bid_size2'][-1]+feat_dict['bid_size3'][-1]+feat_dict['bid_size4'][-1]+feat_dict['bid_size5'][-1]+\
              feat_dict['bid_size6'][-1]+feat_dict['bid_size7'][-1]+feat_dict['bid_size8'][-1]+feat_dict['bid_size9'][-1]+feat_dict['bid_size10'][-1]
        bv = feat_dict['size'].copy()
        # if bv > 0:
        #     bv = 0
        # else:
            # bv[-1] = bv[-1]
        bv[bv>0] = 0
        # print(np.sum(bv[-10:]))
        bv_divide_tn = np.nansum(bv[-10:])/bvs
        # print('bv_divide_tn:',bv_divide_tn)
        if str(bv_divide_tn) == 'nan':
            feat_dict['bv_divide_tn'] = np.append(feat_dict['bv_divide_tn'], 0)
        else:
            feat_dict['bv_divide_tn'] = np.append(feat_dict['bv_divide_tn'], bv_divide_tn)
        feat_dict['bv_divide_tn'] = feat_dict['bv_divide_tn'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['bv_divide_tn'][-1], 'bv_divide_tn',
                                      df, sum_num_, df_2d_)
        
        
    # av_divide_tn
    if len(feat_dict['size'])>10:
        avs = feat_dict['ask_size1'][-1]+feat_dict['ask_size2'][-1]+feat_dict['ask_size3'][-1]+feat_dict['ask_size4'][-1]+feat_dict['ask_size5'][-1]+\
              feat_dict['ask_size6'][-1]+feat_dict['ask_size7'][-1]+feat_dict['ask_size8'][-1]+feat_dict['ask_size9'][-1]+feat_dict['ask_size10'][-1]
        av = feat_dict['size'].copy()
        # print('av_last:',av)
        # if av[-1] < 0:
        #     av[-1] = 0
        # else:
        #     av[-1] = av[-1]
        # print('av_now:',av)
        av[av<0] = 0
        av_divide_tn = np.nansum(av[-10:])/avs
        if str(av_divide_tn) == 'nan':
            feat_dict['av_divide_tn'] = np.append(feat_dict['av_divide_tn'], 0)
        else:
            feat_dict['av_divide_tn'] = np.append(feat_dict['bv_divide_tn'], av_divide_tn)
        feat_dict['av_divide_tn'] = feat_dict['av_divide_tn'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['av_divide_tn'][-1], 'av_divide_tn',
                                  df, sum_num_, df_2d_)
        
    
    # weighted_price_to_mid
    avs_aps ,bvs_bps, avs, bvs = 0, 0,0,0
    # for i in range(1, 6):
    avs_aps = df[col_dict['ask_price1']] * df[col_dict['ask_size1']] + df[col_dict['ask_price2']] * df[col_dict['ask_size2']] +\
              df[col_dict['ask_price3']] * df[col_dict['ask_size3']] + df[col_dict['ask_price4']] * df[col_dict['ask_size4']] +\
              df[col_dict['ask_price5']] * df[col_dict['ask_size5']] + df[col_dict['ask_price6']] * df[col_dict['ask_size6']] +\
              df[col_dict['ask_price7']] * df[col_dict['ask_size7']] + df[col_dict['ask_price8']] * df[col_dict['ask_size8']] +\
              df[col_dict['ask_price9']] * df[col_dict['ask_size9']] + df[col_dict['ask_price10']] * df[col_dict['ask_size10']]
    bvs_bps = df[col_dict['bid_price1']] * df[col_dict['bid_size1']] + df[col_dict['bid_price2']] * df[col_dict['bid_size2']] +\
              df[col_dict['bid_price3']] * df[col_dict['bid_size3']] + df[col_dict['bid_price4']] * df[col_dict['bid_size4']] +\
              df[col_dict['bid_price5']] * df[col_dict['bid_size5']] + df[col_dict['bid_price6']] * df[col_dict['bid_size6']] +\
              df[col_dict['bid_price7']] * df[col_dict['bid_size7']] + df[col_dict['bid_price8']] * df[col_dict['bid_size8']] +\
              df[col_dict['bid_price9']] * df[col_dict['bid_size9']] + df[col_dict['bid_price10']] * df[col_dict['bid_size10']]
    avs = df[col_dict['ask_size1']]+df[col_dict['ask_size2']]+df[col_dict['ask_size3']]+df[col_dict['ask_size4']]+df[col_dict['ask_size5']]+\
            df[col_dict['ask_size6']]+df[col_dict['ask_size7']]+df[col_dict['ask_size8']]+df[col_dict['ask_size9']]+df[col_dict['ask_size10']]    
    bvs = df[col_dict['bid_size1']]+df[col_dict['bid_size2']]+df[col_dict['bid_size3']]+df[col_dict['bid_size4']]+df[col_dict['bid_size5']] +\
            df[col_dict['bid_size6']]+df[col_dict['bid_size7']]+df[col_dict['bid_size8']]+df[col_dict['bid_size9']]+df[col_dict['bid_size10']]
    mp = (df[col_dict['ask_price1']]+df[col_dict['bid_price1']])/2
    weighted_price_to_mid = (avs_aps+bvs_bps)/(avs+bvs)-mp
    feat_dict['weighted_price_to_mid'] = np.append(feat_dict['weighted_price_to_mid'], weighted_price_to_mid)
    feat_dict['weighted_price_to_mid'] = feat_dict['weighted_price_to_mid'][-2000:]
    df_2d_ = test_data_consistency(feat_dict['weighted_price_to_mid'][-1], 'weighted_price_to_mid',
                                  df, sum_num_, df_2d_)
    
    # ask_withdraws
    # ask_withdraws = np.array([])
    if len(feat_dict['ask_price1'])>1:
        ask_ob_values_last = np.hstack((feat_dict['ask_price1'][-2], feat_dict['ask_size1'][-2], feat_dict['bid_price1'][-2], feat_dict['bid_size1'][-2],
                           feat_dict['ask_price2'][-2], feat_dict['ask_size2'][-2], feat_dict['bid_price2'][-2], feat_dict['bid_size2'][-2], 
                           feat_dict['ask_price3'][-2], feat_dict['ask_size3'][-2], feat_dict['bid_price3'][-2], feat_dict['bid_size3'][-2],
                           feat_dict['ask_price4'][-2], feat_dict['ask_size4'][-2], feat_dict['bid_price4'][-2], feat_dict['bid_size4'][-2],
                           feat_dict['ask_price5'][-2], feat_dict['ask_size5'][-2], feat_dict['bid_price5'][-2], feat_dict['bid_size5'][-2],
                           feat_dict['ask_price6'][-2], feat_dict['ask_size6'][-2], feat_dict['bid_price6'][-2], feat_dict['bid_size6'][-2],
                           feat_dict['ask_price7'][-2], feat_dict['ask_size7'][-2], feat_dict['bid_price7'][-2], feat_dict['bid_size7'][-2], 
                           feat_dict['ask_price8'][-2], feat_dict['ask_size8'][-2], feat_dict['bid_price8'][-2], feat_dict['bid_size8'][-2],
                           feat_dict['ask_price9'][-2], feat_dict['ask_size9'][-2], feat_dict['bid_price9'][-2], feat_dict['bid_size9'][-2],
                           feat_dict['ask_price10'][-2], feat_dict['ask_size10'][-2], feat_dict['bid_price10'][-2], feat_dict['bid_size10'][-2]))
        ask_ob_values_now = np.hstack((feat_dict['ask_price1'][-1], feat_dict['ask_size1'][-1], feat_dict['bid_price1'][-1], feat_dict['bid_size1'][-1],
                           feat_dict['ask_price2'][-1], feat_dict['ask_size2'][-1], feat_dict['bid_price2'][-1], feat_dict['bid_size2'][-1], 
                           feat_dict['ask_price3'][-1], feat_dict['ask_size3'][-1], feat_dict['bid_price3'][-1], feat_dict['bid_size3'][-1],
                           feat_dict['ask_price4'][-1], feat_dict['ask_size4'][-1], feat_dict['bid_price4'][-1], feat_dict['bid_size4'][-1],
                           feat_dict['ask_price5'][-1], feat_dict['ask_size5'][-1], feat_dict['bid_price5'][-1], feat_dict['bid_size5'][-1],
                           feat_dict['ask_price6'][-1], feat_dict['ask_size6'][-1], feat_dict['bid_price6'][-1], feat_dict['bid_size6'][-1],
                           feat_dict['ask_price7'][-1], feat_dict['ask_size7'][-1], feat_dict['bid_price7'][-1], feat_dict['bid_size7'][-1], 
                           feat_dict['ask_price8'][-1], feat_dict['ask_size8'][-1], feat_dict['bid_price8'][-1], feat_dict['bid_size8'][-1],
                           feat_dict['ask_price9'][-1], feat_dict['ask_size9'][-1], feat_dict['bid_price9'][-1], feat_dict['bid_size9'][-1],
                           feat_dict['ask_price10'][-1], feat_dict['ask_size10'][-1], feat_dict['bid_price10'][-1], feat_dict['bid_size10'][-1]))

        ask_withdraws = _ask_withdraws_volume(ask_ob_values_last, ask_ob_values_now)
        
        feat_dict['ask_withdraws'] = np.append(feat_dict['ask_withdraws'], ask_withdraws)
        feat_dict['ask_withdraws'] = feat_dict['ask_withdraws'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['ask_withdraws'][-1], 'ask_withdraws',
                                  df, sum_num_, df_2d_)
        
    # bid_withdraws
    # bid_withdraws = np.array([])
    if len(feat_dict['ask_price1'])>1:
        bid_ob_values_last = np.hstack((feat_dict['ask_price1'][-2], feat_dict['ask_size1'][-2], feat_dict['bid_price1'][-2], feat_dict['bid_size1'][-2],
                           feat_dict['ask_price2'][-2], feat_dict['ask_size2'][-2], feat_dict['bid_price2'][-2], feat_dict['bid_size2'][-2], 
                           feat_dict['ask_price3'][-2], feat_dict['ask_size3'][-2], feat_dict['bid_price3'][-2], feat_dict['bid_size3'][-2],
                           feat_dict['ask_price4'][-2], feat_dict['ask_size4'][-2], feat_dict['bid_price4'][-2], feat_dict['bid_size4'][-2],
                           feat_dict['ask_price5'][-2], feat_dict['ask_size5'][-2], feat_dict['bid_price5'][-2], feat_dict['bid_size5'][-2],
                           feat_dict['ask_price6'][-2], feat_dict['ask_size6'][-2], feat_dict['bid_price6'][-2], feat_dict['bid_size6'][-2],
                           feat_dict['ask_price7'][-2], feat_dict['ask_size7'][-2], feat_dict['bid_price7'][-2], feat_dict['bid_size7'][-2], 
                           feat_dict['ask_price8'][-2], feat_dict['ask_size8'][-2], feat_dict['bid_price8'][-2], feat_dict['bid_size8'][-2],
                           feat_dict['ask_price9'][-2], feat_dict['ask_size9'][-2], feat_dict['bid_price9'][-2], feat_dict['bid_size9'][-2],
                           feat_dict['ask_price10'][-2], feat_dict['ask_size10'][-2], feat_dict['bid_price10'][-2], feat_dict['bid_size10'][-2]))
        bid_ob_values_now = np.hstack((feat_dict['ask_price1'][-1], feat_dict['ask_size1'][-1], feat_dict['bid_price1'][-1], feat_dict['bid_size1'][-1],
                           feat_dict['ask_price2'][-1], feat_dict['ask_size2'][-1], feat_dict['bid_price2'][-1], feat_dict['bid_size2'][-1], 
                           feat_dict['ask_price3'][-1], feat_dict['ask_size3'][-1], feat_dict['bid_price3'][-1], feat_dict['bid_size3'][-1],
                           feat_dict['ask_price4'][-1], feat_dict['ask_size4'][-1], feat_dict['bid_price4'][-1], feat_dict['bid_size4'][-1],
                           feat_dict['ask_price5'][-1], feat_dict['ask_size5'][-1], feat_dict['bid_price5'][-1], feat_dict['bid_size5'][-1],
                           feat_dict['ask_price6'][-1], feat_dict['ask_size6'][-1], feat_dict['bid_price6'][-1], feat_dict['bid_size6'][-1],
                           feat_dict['ask_price7'][-1], feat_dict['ask_size7'][-1], feat_dict['bid_price7'][-1], feat_dict['bid_size7'][-1], 
                           feat_dict['ask_price8'][-1], feat_dict['ask_size8'][-1], feat_dict['bid_price8'][-1], feat_dict['bid_size8'][-1],
                           feat_dict['ask_price9'][-1], feat_dict['ask_size9'][-1], feat_dict['bid_price9'][-1], feat_dict['bid_size9'][-1],
                           feat_dict['ask_price10'][-1], feat_dict['ask_size10'][-1], feat_dict['bid_price10'][-1], feat_dict['bid_size10'][-1]))

        
        bid_withdraws = _bid_withdraws_volume(bid_ob_values_last, bid_ob_values_now)
        
        feat_dict['bid_withdraws'] = np.append(feat_dict['bid_withdraws'], bid_withdraws)
        feat_dict['bid_withdraws'] = feat_dict['bid_withdraws'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['bid_withdraws'][-1], 'bid_withdraws',
                                  df, sum_num_, df_2d_)
        
    # z_t
    tick_fac_data = np.log(df[col_dict['price']])-np.log((df[col_dict['ask_price1']]+df[col_dict['bid_price1']])/2)
    feat_dict['z_t'] = np.append(feat_dict['z_t'], tick_fac_data)
    feat_dict['z_t'] = feat_dict['z_t'][-2000:]
    df_2d_ = test_data_consistency(feat_dict['z_t'][-1], 'z_t',
                                  df, sum_num_, df_2d_)
    
    # voi
    if len(feat_dict['bid_price1'])>1:
        bid_sub_price = feat_dict['bid_price1'][-1] - feat_dict['bid_price1'][shift_(1)]
        ask_sub_price = feat_dict['ask_price1'][-1] - feat_dict['ask_price1'][shift_(1)]
        bid_sub_volume = feat_dict['bid_size1'][-1] - feat_dict['bid_size1'][shift_(1)]
        ask_sub_volume = feat_dict['ask_size1'][-1] - feat_dict['ask_size1'][shift_(1)]
        bid_volume_change = bid_sub_volume
        ask_volume_change = ask_sub_volume
        if bid_sub_price <0:
            bid_volume_change = 0
        if bid_sub_price >0:
            bid_volume_change = feat_dict['bid_size1'][-1]
        if ask_sub_price >0:
            ask_volume_change = 0
        if ask_sub_price <0:
            ask_volume_change = feat_dict['ask_size1'][-1]        
        voi = (bid_volume_change - ask_volume_change) / feat_dict['cum_size'][-1]
        feat_dict['voi'] = np.append(feat_dict['voi'], voi)
        feat_dict['voi'] = feat_dict['voi'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['voi'][-1], 'voi',
                                      df, sum_num_, df_2d_)

    # cal_weight_volume
    w = [1 - (i - 1) / 10 for i in range(1, 11)]
    w = np.array(w) / sum(w)
    wb = df[col_dict['bid_size1']]*w[0]+df[col_dict['bid_size2']]*w[1]+df[col_dict['bid_size3']]*w[2]+df[col_dict['bid_size4']]*w[3]+df[col_dict['bid_size5']]*w[4] +\
         df[col_dict['bid_size6']]*w[5]+df[col_dict['bid_size7']]*w[6]+df[col_dict['bid_size8']]*w[7]+df[col_dict['bid_size9']]*w[8]+df[col_dict['bid_size10']]*w[9]
    wa = df[col_dict['ask_size1']]*w[0]+df[col_dict['ask_size2']]*w[1]+df[col_dict['ask_size3']]*w[2]+df[col_dict['ask_size4']]*w[3]+df[col_dict['ask_size5']]*w[4] +\
         df[col_dict['ask_size6']]*w[5]+df[col_dict['ask_size7']]*w[6]+df[col_dict['ask_size8']]*w[7]+df[col_dict['ask_size9']]*w[8]+df[col_dict['ask_size10']]*w[9]
    feat_dict['wb'] = np.append(feat_dict['wb'], wa)
    feat_dict['wb'] = feat_dict['wb'][-2000:]
    feat_dict['wa'] = np.append(feat_dict['wa'], wb)
    feat_dict['wa'] = feat_dict['wa'][-2000:]
    df_2d_ = test_data_consistency(feat_dict['wb'][-1], 'wb',
                                  df, sum_num_, df_2d_)
    df_2d_ = test_data_consistency(feat_dict['wa'][-1], 'wa',
                                  df, sum_num_, df_2d_)
    
    
    #voi2
    if len(feat_dict['ask_price1'])>1:
        bid_sub_price_2 = feat_dict['bid_price1'][-1] - feat_dict['bid_price1'][shift_(1)]
        ask_sub_price_2 = feat_dict['ask_price1'][-1] - feat_dict['ask_price1'][shift_(1)]
        bid_sub_volume_2 = feat_dict['wa'][-1] - feat_dict['wa'][shift_(1)]
        ask_sub_volume_2 = feat_dict['wb'][-1] - feat_dict['wb'][shift_(1)]
        bid_volume_change_2 = bid_sub_volume_2
        ask_volume_change_2 = ask_sub_volume_2
        if bid_sub_price_2 <0:
            bid_volume_change_2 = 0
        if bid_sub_price_2 >0:
            bid_volume_change_2 = feat_dict['wa'][-1]
        if ask_sub_price_2 >0:
            ask_volume_change_2 = 0
        if ask_sub_price_2 <0:
            ask_volume_change_2 = feat_dict['wb'][-1]
        voi2 = (bid_volume_change_2 - ask_volume_change_2) / feat_dict['cum_size'][-1]
        feat_dict['voi2'] = np.append(feat_dict['voi2'], voi2)
        feat_dict['voi2'] = feat_dict['voi2'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['voi2'][-1], 'voi2',
                                      df, sum_num_, df_2d_)
    
    # mpb
    if len(feat_dict['ask_price1'])>3:
        # tp = feat_dict['turnover'][-1]/feat_dict['cum_size'][-1]
        tp = feat_dict['turnover']/feat_dict['cum_size']
        tp = np.where(np.isinf(tp),np.nan,tp)
        # print('tp---------',tp)
        tp = np.where(np.isnan(tp),np.where(np.isnan(tp[-2]),tp[-3],tp[-2]),tp)
        
        mid_last = (feat_dict['bid_price1'][-2]+feat_dict['ask_price1'][-2])/2
        mid = (feat_dict['bid_price1'][-1]+feat_dict['ask_price1'][-1])/2
        mpb = tp[-1] - (mid+mid_last)/1000/2
        feat_dict['mpb'] = np.append(feat_dict['mpb'], mpb)
        feat_dict['mpb'] = feat_dict['mpb'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['mpb'][-1], 'mpb',
                                      df, sum_num_, df_2d_)
    
    # slope
    slope = (df[col_dict['ask_price1']]-df[col_dict['bid_price1']])/(df[col_dict['ask_size1']]+df[col_dict['bid_size1']])*2
    feat_dict['slope'] = np.append(feat_dict['slope'], slope)
    feat_dict['slope'] = feat_dict['slope'][-2000:]
    df_2d_ = test_data_consistency(feat_dict['slope'][-1], 'slope',
                                      df, sum_num_, df_2d_)
    
    
    # price_weighted_pressure
    kws = {}
    n1 = kws.setdefault("n1", 1)
    n2 = kws.setdefault("n2", 10)

    bench = kws.setdefault("bench_type","MID")
    _ = np.arange(n1, n2 + 1)
    if bench == "MID":
        bench_prices = df[col_dict['ask_price1']]+df[col_dict['bid_price1']]
    elif bench == "SPECIFIC":
        bench_prices = kws.get("bench_price")
    else:
        raise Exception("")
    bid_d = [bench_prices/(bench_prices-df[col_dict['bid_price%s'%s]]) for s in _]
    # bid_d = [(bench_prices/(bench_prices-df[col_dict['bid_price1']][3])),(bench_prices/(bench_prices-df[col_dict['bid_price2']][7])),
             # (bench_prices/(bench_prices-df[col_dict['bid_price3']][11])),(bench_prices/(bench_prices-df[col_dict['bid_price4']][15])),
             # (bench_prices/(bench_prices-df[col_dict['bid_price5']][19]))]
    bid_denominator = np.sum(bid_d)
    # bid_weights = [(d / bid_denominator).replace(np.nan,1) for d in bid_d]
    bid_weights = np.array([])
    for d in bid_d:
        if d/bid_denominator==np.nan:
            bid_weights = np.append(bid_weights, 1)
        else:
            bid_weights = np.append(bid_weights, d/bid_denominator)
    press_buy = np.sum([df[col_dict["bid_size%s" % (i + 1)]] * w for i, w in enumerate(bid_weights)])
    ask_d = [bench_prices / (df[col_dict['ask_price%s' % s]] - bench_prices) for s in _]
    # ask_d = [(bench_prices/(bench_prices-df[col_dict['ask_price1']][1])),(bench_prices/(bench_prices-df[col_dict['ask_price2']][5])),
             # (bench_prices/(bench_prices-df[col_dict['ask_price3']][9])),(bench_prices/(bench_prices-df[col_dict['ask_price4']][13])),
             # (bench_prices/(bench_prices-df[col_dict['ask_price5']][17]))]
    ask_denominator = np.sum(ask_d)
    ask_weights = [d / ask_denominator for d in ask_d]
    press_sell = sum([df[col_dict['ask_size%s' % (i + 1)]] * w for i, w in enumerate(ask_weights)])
    price_weighted_pressure = np.log(press_buy) - np.log(press_sell)
    if price_weighted_pressure == np.inf or price_weighted_pressure == -np.inf:
        feat_dict['price_weighted_pressure']= np.append(feat_dict['price_weighted_pressure'], np.nan)
    else:
        feat_dict['price_weighted_pressure'] = np.append(feat_dict['price_weighted_pressure'], price_weighted_pressure)
    feat_dict['price_weighted_pressure'] = feat_dict['price_weighted_pressure'][-2000:]
    df_2d_ = test_data_consistency(feat_dict['price_weighted_pressure'][-1], 'price_weighted_pressure',
                                      df, sum_num_, df_2d_)
    
    # volume_order_imbalance
    if len(feat_dict['bid_price1'])>1:
        kws = {}
        drop_first = kws.setdefault("drop_first", True)
        current_bid_price = feat_dict['bid_price1'][-1]
        bid_price_diff = current_bid_price - feat_dict['bid_price1'][shift_(1)]
        current_bid_vol = feat_dict['bid_size1'][-1]
        bvol_diff = current_bid_vol - feat_dict['bid_size1'][shift_(1)]
        bid_increment = current_bid_vol if bid_price_diff > 0 else (0 if bid_price_diff < 0 else (bvol_diff if bid_price_diff == 0 else bid_price_diff))
        current_ask_price = feat_dict['ask_price1'][-1]
        ask_price_diff = current_ask_price - feat_dict['ask_price1'][shift_(1)]
        current_ask_vol = feat_dict['ask_size1'][-1]
        avol_diff = current_ask_vol - feat_dict['ask_size1'][shift_(1)]
        ask_increment = current_ask_vol if ask_price_diff < 0 else (0 if ask_price_diff > 0 else (avol_diff if ask_price_diff == 0 else ask_price_diff))
        _ = bid_increment - ask_increment
        
        feat_dict['volume_order_imbalance'] = np.append(feat_dict['volume_order_imbalance'], _)
        feat_dict['volume_order_imbalance'] = feat_dict['volume_order_imbalance'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['volume_order_imbalance'][-1], 'volume_order_imbalance',
                                      df, sum_num_, df_2d_)
    
    
    # get_mid_price_change
    if len(feat_dict['ask_price1'])>1:
        mid_last = (feat_dict['ask_price1'][-2]+feat_dict['bid_price1'][-2])/2
        mid = (feat_dict['ask_price1'][-1]+feat_dict['bid_price1'][-1])/2
        get_mid_price_change = mid/mid_last-1
        feat_dict['get_mid_price_change'] = np.append(feat_dict['get_mid_price_change'], get_mid_price_change)
        feat_dict['get_mid_price_change'] = feat_dict['get_mid_price_change'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['get_mid_price_change'][-1], 'get_mid_price_change',
                                      df, sum_num_, df_2d_)
    
    # mpb_500
    if len(feat_dict['ask_price1'])>500:
        tp_500 = feat_dict['turnover']/feat_dict['cum_size']
        tp_500 = np.where(np.isinf(tp_500),np.nan,tp_500)
        tp_500 = np.where(np.isnan(tp_500),np.where(np.isnan(tp_500[-2]),tp_500[-3],tp_500[-2]),tp_500)
        
        mid_last = (feat_dict['bid_price1'][shift_(500)]+feat_dict['ask_price1'][shift_(500)])/2
        mid = (feat_dict['bid_price1'][-1]+feat_dict['ask_price1'][-1])/2
        mpb_500 = tp_500[-1] - (mid+mid_last)/1000/2
        feat_dict['mpb_500'] = np.append(feat_dict['mpb_500'], mpb_500)
        feat_dict['mpb_500'] = feat_dict['mpb_500'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['mpb_500'][-1], 'mpb_500',
                                      df, sum_num_, df_2d_)
    
    # positive_buying
    if len(feat_dict['ask_price1'])>2:
        posi_buy_cum_ = np.where(feat_dict['price'][-1]>=feat_dict['ask_price1'][-2], feat_dict['cum_size'][-1], 0)
        feat_dict['posi_buy_cum_'] = np.append(feat_dict['posi_buy_cum_'], posi_buy_cum_)
        feat_dict['posi_buy_cum_'] = feat_dict['posi_buy_cum_'][-2000:]
        caus_buy_cum_ = np.where(feat_dict['price'][-1]<=feat_dict['bid_price1'][-2], feat_dict['cum_size'][-1], 0)
        feat_dict['caus_buy_cum_'] = np.append(feat_dict['caus_buy_cum_'], caus_buy_cum_)
        feat_dict['caus_buy_cum_'] = feat_dict['caus_buy_cum_'][-2000:]
        if len(feat_dict['caus_buy_cum_'])>1000:
            bm = np.sum(feat_dict['posi_buy_cum_'][-1000:])/np.sum(feat_dict['caus_buy_cum_'][-1000:])
            feat_dict['positive_buying'] = np.append(feat_dict['positive_buying'], bm)
            feat_dict['positive_buying'] = feat_dict['positive_buying'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['positive_buying'][-1], 'positive_buying',
                                          df, sum_num_, df_2d_)
    
    # positive_selling
    if len(feat_dict['ask_price1'])>2:
        posi_sell_cum_ = np.where(feat_dict['price'][-1]<=feat_dict['bid_price1'][-2], feat_dict['cum_size'][-1], 0)
        feat_dict['posi_sell_cum_'] = np.append(feat_dict['posi_sell_cum_'], posi_sell_cum_)
        feat_dict['posi_sell_cum_'] = feat_dict['posi_sell_cum_'][-2000:]
        caus_sell_cum_ = np.where(feat_dict['price'][-1]>=feat_dict['ask_price1'][-2], feat_dict['cum_size'][-1], 0)
        feat_dict['caus_sell_cum_'] = np.append(feat_dict['caus_sell_cum_'], caus_sell_cum_)
        feat_dict['caus_sell_cum_'] = feat_dict['caus_sell_cum_'][-2000:]
        if len(feat_dict['caus_sell_cum_'])>1000:
            sm = np.sum(feat_dict['posi_sell_cum_'][-1000:])/np.sum(feat_dict['caus_sell_cum_'][-1000:])
            feat_dict['positive_selling'] = np.append(feat_dict['positive_selling'], sm)
            feat_dict['positive_selling'] = feat_dict['positive_selling'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['positive_selling'][-1], 'positive_selling',
                                          df, sum_num_, df_2d_)
    # buying_amplification_ratio
    if len(feat_dict['ask_price1'])>2:
        asking_shift = feat_dict['ask_price1'][-2] * feat_dict['ask_size1'][-2] + feat_dict['ask_price2'][-2] * feat_dict['ask_size2'][-2] +\
                       feat_dict['ask_price3'][-2] * feat_dict['ask_size3'][-2] + feat_dict['ask_price4'][-2] * feat_dict['ask_size4'][-2] +\
                       feat_dict['ask_price5'][-2] * feat_dict['ask_size5'][-2]
        biding_shift = feat_dict['bid_price1'][-2] * feat_dict['bid_size1'][-2] + feat_dict['bid_price2'][-2] * feat_dict['bid_size2'][-2] +\
                       feat_dict['bid_price3'][-2] * feat_dict['bid_size3'][-2] + feat_dict['bid_price4'][-2] * feat_dict['bid_size4'][-2] +\
                       feat_dict['bid_price5'][-2] * feat_dict['bid_size5'][-2]
        asking = feat_dict['ask_price1'][-1] * feat_dict['ask_size1'][-1] + feat_dict['ask_price2'][-1] * feat_dict['ask_size2'][-1] +\
                 feat_dict['ask_price3'][-1] * feat_dict['ask_size3'][-1] + feat_dict['ask_price4'][-1] * feat_dict['ask_size4'][-1] +\
                 feat_dict['ask_price5'][-1] * feat_dict['ask_size5'][-1]
        biding = feat_dict['bid_price1'][-1] * feat_dict['bid_size1'][-1] + feat_dict['bid_price2'][-1] * feat_dict['bid_size2'][-1] +\
                 feat_dict['bid_price3'][-1] * feat_dict['bid_size3'][-1] + feat_dict['bid_price4'][-1] * feat_dict['bid_size4'][-1] +\
                 feat_dict['bid_price5'][-1] * feat_dict['bid_size5'][-1]
        amplify_biding = np.where(biding>biding_shift, biding-biding_shift,0)
        feat_dict['amplify_biding'] = np.append(feat_dict['amplify_biding'], amplify_biding)
        feat_dict['amplify_biding'] = feat_dict['amplify_biding'][-2000:]
        amplify_asking = np.where(asking>asking_shift, asking-asking_shift,0)
        feat_dict['amplify_asking'] = np.append(feat_dict['amplify_asking'], amplify_asking)
        feat_dict['amplify_asking'] = feat_dict['amplify_asking'][-2000:]
        diff = feat_dict['amplify_biding'] - feat_dict['amplify_asking']
        if len(feat_dict['amplify_biding'])>1000:     
            buying_ratio = np.sum(diff[-1000:])/feat_dict['turnover'][-1]/1000
            feat_dict['buying_amplification_ratio'] = np.append(feat_dict['buying_amplification_ratio'], buying_ratio)
            feat_dict['buying_amplification_ratio'] = feat_dict['buying_amplification_ratio'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['buying_amplification_ratio'][-1], 'buying_amplification_ratio',
                                          df, sum_num_, df_2d_)
    
    # buying_amount_ratio
    if len(feat_dict['ask_price1'])>2:
        posi_buy_turnover_ = np.where(feat_dict['price'][-1]>=feat_dict['ask_price1'][-2], feat_dict['turnover'][-1], 0)
        feat_dict['posi_buy_turnover_'] = np.append(feat_dict['posi_buy_turnover_'], posi_buy_turnover_)
        feat_dict['posi_buy_turnover_'] = feat_dict['posi_buy_turnover_'][-2000:]
        posi_sell_turnover_ = np.where(feat_dict['price'][-1]<=feat_dict['bid_price1'][-2], feat_dict['turnover'][-1], 0)
        feat_dict['posi_sell_turnover_'] = np.append(feat_dict['posi_sell_turnover_'], posi_sell_turnover_)
        feat_dict['posi_sell_turnover_'] = feat_dict['posi_sell_turnover_'][-2000:]
        diff_ = feat_dict['posi_buy_turnover_'] - feat_dict['posi_sell_turnover_']
        if len(feat_dict['posi_buy_turnover_'])>1000:
            buying_amount_ratio = (np.sum(diff_[-1000:])/np.sum(feat_dict['turnover'][-1000:]))/1000
            feat_dict['buying_amount_ratio'] = np.append(feat_dict['buying_amount_ratio'], buying_amount_ratio)
            feat_dict['buying_amount_ratio'] = feat_dict['buying_amount_ratio'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['buying_amount_ratio'][-1], 'buying_amount_ratio',
                                          df, sum_num_, df_2d_)
    # buying_willing
    if len(feat_dict['amplify_biding'])>1000:
        dif_ = (feat_dict['amplify_biding']-feat_dict['amplify_asking']) + (feat_dict['posi_buy_turnover_']-feat_dict['posi_sell_turnover_'])
        buying_willing = (np.sum(dif_[-1000:])/np.sum(feat_dict['turnover'][-1000:]))/1000
        feat_dict['buying_willing'] = np.append(feat_dict['buying_willing'], buying_willing)
        feat_dict['buying_willing'] = feat_dict['buying_willing'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['buying_willing'][-1], 'buying_willing',
                                      df, sum_num_, df_2d_)
    
    # buying_willing_strength
    if len(feat_dict['ask_price1'])>2:
        bid_ = feat_dict['bid_size1'][-1] + feat_dict['bid_size2'][-1] + feat_dict['bid_size3'][-1] + feat_dict['bid_size4'][-1] + feat_dict['bid_size5'][-1]
        feat_dict['bid_'] = np.append(feat_dict['bid_'] ,bid_)
        feat_dict['bid_'] = feat_dict['bid_'][-2000:]
        ask_ = feat_dict['ask_size1'][-1] + feat_dict['ask_size2'][-1] + feat_dict['ask_size3'][-1] + feat_dict['ask_size4'][-1] + feat_dict['ask_size5'][-1]
        feat_dict['ask_'] = np.append(feat_dict['ask_'] ,ask_)
        feat_dict['ask_'] = feat_dict['ask_'][-2000:]
        di_ = (feat_dict['bid_']-feat_dict['ask_']) + (feat_dict['posi_buy_turnover_'] - feat_dict['posi_sell_turnover_'])
        buying_willing_strength_ = np.mean(di_[-1000:])/np.std(di_[-1000:])
        feat_dict['buying_willing_strength_'] = np.append(feat_dict['buying_willing_strength_'], buying_willing_strength_)
        feat_dict['buying_willing_strength_'] = feat_dict['buying_willing_strength_'][-2000:]
        if len(feat_dict['buying_willing_strength_'])>1000:
            buying_willing_strength = np.std(feat_dict['buying_willing_strength_'][-1000:])/1000
            feat_dict['buying_willing_strength'] = np.append(feat_dict['buying_willing_strength'], buying_willing_strength)
            feat_dict['buying_willing_strength'] = feat_dict['buying_willing_strength'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['buying_willing_strength'][-1], 'buying_willing_strength',
                                          df, sum_num_, df_2d_)
    
    # buying_amount_strength
    if len(feat_dict['posi_buy_turnover_'])>1000:
        d_ = feat_dict['posi_buy_turnover_'] - feat_dict['posi_sell_turnover_']
        buying_amount_strength_ = np.mean(d_[-1000:])/np.std(d_[-1000:])
        feat_dict['buying_amount_strength_'] = np.append(feat_dict['buying_amount_strength_'], buying_amount_strength_)
        feat_dict['buying_amount_strength_'] = feat_dict['buying_amount_strength_'][-2000:]
        if len(feat_dict['buying_amount_strength_'])>1000:
            buying_amount_strength = np.std(feat_dict['buying_amount_strength_'][-1000:])/1000
            feat_dict['buying_amount_strength'] = np.append(feat_dict['buying_amount_strength'], buying_amount_strength)
            sum_num_ = test_data_consistency(feat_dict['buying_amount_strength'][-1], 'buying_amount_strength',
                                          df, sum_num_, df_2d_)      
      
    
    # selling_ratio
    if len(feat_dict['amplify_asking'])>1000:
        _diff = feat_dict['amplify_asking'] - feat_dict['amplify_biding']
        selling_ratio = np.sum(_diff[-1000:])/feat_dict['turnover'][-1]/1000
        feat_dict['selling_ratio'] = np.append(feat_dict['selling_ratio'], selling_ratio)
        feat_dict['selling_ratio'] = feat_dict['selling_ratio'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['selling_ratio'][-1], 'selling_ratio',
                                      df, sum_num_, df_2d_)
    
    # buy_order_aggressivenes_level1
    biding = feat_dict['bid_price1'] * feat_dict['bid_size1'] + feat_dict['bid_price2'] * feat_dict['bid_size2'] +\
             feat_dict['bid_price3'] * feat_dict['bid_size3'] + feat_dict['bid_price4'] * feat_dict['bid_size4'] +\
             feat_dict['bid_price5'] * feat_dict['bid_size5']
    if len(feat_dict['ask_price1'])>2:
        b_1_v_ = feat_dict['size'].copy()
        b_1_p_ = feat_dict['price'].copy()
        b_1_p_[b_1_v_<0]=0
        b_1_p_ = np.where(np.isnan(b_1_p_),0,b_1_p_)
        buy_price_1 = np.where((b_1_p_[-1]>=feat_dict['ask_price1'][-2])&(b_1_v_[-1]>=feat_dict['ask_size1'][-2]), b_1_p_[-1], 0)
        feat_dict['buy_price_1'] = np.append(feat_dict['buy_price_1'] ,buy_price_1)
        feat_dict['buy_price_1'] = feat_dict['buy_price_1'][-2000:]
        buy_amount_1 = np.where((b_1_p_[-1]>=feat_dict['ask_price1'][-2])&(b_1_v_[-1]>=feat_dict['ask_size1'][-2]), feat_dict['turnover'][-1]-feat_dict['turnover'][-2], np.nan)
        feat_dict['buy_amount_1'] = np.append(feat_dict['buy_amount_1'] ,buy_amount_1)
        feat_dict['buy_amount_1'] = feat_dict['buy_amount_1'][-2000:]
        if len(feat_dict['ask_price1'])>1000:
            mid_shift = (feat_dict['bid_price1'][-1000-1]+feat_dict['ask_price1'][-1000-1])/2
            buy_amount_agg_ratio_level1 = np.sum(biding[-1000:])/feat_dict['buy_amount_1'][-1]
            buy_price_bias_level1 = np.abs(feat_dict['buy_price_1'][-1] - mid_shift)/mid_shift
            feat_dict['buy_price_bias_level1'] = np.append(feat_dict['buy_price_bias_level1'], buy_price_bias_level1)
            feat_dict['buy_price_bias_level1'] = feat_dict['buy_price_bias_level1'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['buy_price_bias_level1'][-1], 'buy_price_bias_level1',
                                          df, sum_num_, df_2d_)
            feat_dict['buy_amount_agg_ratio_level1'] = np.append(feat_dict['buy_amount_agg_ratio_level1'], buy_amount_agg_ratio_level1)
            feat_dict['buy_amount_agg_ratio_level1'] = feat_dict['buy_amount_agg_ratio_level1'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['buy_amount_agg_ratio_level1'][-1], 'buy_amount_agg_ratio_level1',
                                          df, sum_num_, df_2d_)
    
    # buy_order_aggressivenes_level2
    if len(feat_dict['ask_price1'])>2:
        b_2_v_ = feat_dict['size'].copy()
        b_2_p_ = feat_dict['price'].copy()
        b_2_p_[b_2_v_<0]=0
        b_2_p_ = np.where(np.isnan(b_2_p_),0,b_2_p_)
        buy_price_2 = np.where((b_2_p_[-1]>=feat_dict['ask_price1'][-2])&(b_2_v_[-1]<=feat_dict['ask_size1'][-2]), b_2_p_[-1], 0)
        feat_dict['buy_price_2'] = np.append(feat_dict['buy_price_2'] ,buy_price_2)
        feat_dict['buy_price_2'] = feat_dict['buy_price_2'][-2000:]
        buy_amount_2 = np.where((b_2_p_[-1]>=feat_dict['ask_price1'][-2])&(b_2_v_[-1]<=feat_dict['ask_size1'][-2]), feat_dict['turnover'][-1]-feat_dict['turnover'][-2], np.nan)
        feat_dict['buy_amount_2'] = np.append(feat_dict['buy_amount_2'] ,buy_amount_2)
        feat_dict['buy_amount_2'] = feat_dict['buy_amount_2'][-2000:]
        if len(feat_dict['ask_price1'])>1000:
            mid_shift = (feat_dict['bid_price1'][-1000-1]+feat_dict['ask_price1'][-1000-1])/2
            buy_amount_agg_ratio_level2 = np.sum(biding[-1000:])/feat_dict['buy_amount_2'][-1]
            buy_price_bias_level2 = np.abs(feat_dict['buy_price_2'][-1] - mid_shift)/mid_shift
            feat_dict['buy_price_bias_level2'] = np.append(feat_dict['buy_price_bias_level2'], buy_price_bias_level2)
            feat_dict['buy_price_bias_level2'] = feat_dict['buy_price_bias_level2'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['buy_price_bias_level2'][-1], 'buy_price_bias_level2',
                                          df, sum_num_, df_2d_)
            feat_dict['buy_amount_agg_ratio_level2'] = np.append(feat_dict['buy_amount_agg_ratio_level2'], buy_amount_agg_ratio_level2)
            feat_dict['buy_amount_agg_ratio_level2'] = feat_dict['buy_amount_agg_ratio_level2'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['buy_amount_agg_ratio_level2'][-1], 'buy_amount_agg_ratio_level2',
                                          df, sum_num_, df_2d_)
    
    # sell_order_aggressivenes_level1
    asking = feat_dict['ask_price1'] * feat_dict['ask_size1'] + feat_dict['ask_price2'] * feat_dict['ask_size2'] +\
             feat_dict['ask_price3'] * feat_dict['ask_size3'] + feat_dict['ask_price4'] * feat_dict['ask_size4'] +\
             feat_dict['ask_price5'] * feat_dict['ask_size5']
    if len(feat_dict['bid_price1'])>2:
        s_1_v_ = feat_dict['size'].copy()
        s_1_p_ = feat_dict['price'].copy()
        s_1_p_[s_1_v_>0]=0
        sell_price_1 = np.where((s_1_p_[-1]<=feat_dict['bid_price1'][-2])&(abs(s_1_v_[-1])>=feat_dict['bid_size1'][-2]), s_1_p_[-1], 0)
        feat_dict['sell_price_1'] = np.append(feat_dict['sell_price_1'] ,sell_price_1)
        feat_dict['sell_price_1'] = feat_dict['sell_price_1'][-2000:]
        sell_amount_1 = np.where((s_1_p_[-1]<=feat_dict['bid_price1'][-2])&(abs(s_1_v_[-1])>=feat_dict['bid_size1'][-2]), feat_dict['turnover'][-1]-feat_dict['turnover'][-2], np.nan)
        feat_dict['sell_amount_1'] = np.append(feat_dict['sell_amount_1'] ,sell_amount_1)
        feat_dict['sell_amount_1'] = feat_dict['sell_amount_1'][-2000:]
        if len(feat_dict['bid_price1'])>1000:
            mid_shift = (feat_dict['bid_price1'][-1000-1]+feat_dict['ask_price1'][-1000-1])/2
            sell_amount_agg_ratio_level1 = np.sum(asking[-1000:])/feat_dict['sell_amount_1'][-1]
            sell_price_bias_level1 = np.abs(feat_dict['sell_price_1'][-1] - mid_shift)/mid_shift
            feat_dict['sell_amount_agg_ratio_level1'] = np.append(feat_dict['sell_amount_agg_ratio_level1'], sell_amount_agg_ratio_level1)
            feat_dict['sell_amount_agg_ratio_level1'] = feat_dict['sell_amount_agg_ratio_level1'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['sell_amount_agg_ratio_level1'][-1], 'sell_amount_agg_ratio_level1',
                                          df, sum_num_, df_2d_)
            feat_dict['sell_price_bias_level1'] = np.append(feat_dict['sell_price_bias_level1'], sell_price_bias_level1)
            feat_dict['sell_price_bias_level1'] = feat_dict['sell_price_bias_level1'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['sell_price_bias_level1'][-1], 'sell_price_bias_level1',
                                          df, sum_num_, df_2d_)
    
    # sell_order_aggressivenes_level2
    if len(feat_dict['bid_price1'])>2:
        s_2_v_ = feat_dict['size'].copy()
        s_2_p_ = feat_dict['price'].copy()
        s_2_p_[s_2_v_>0]=0
        sell_price_2 = np.where((s_2_p_[-1]<=feat_dict['bid_price1'][-2])&(abs(s_2_v_[-1])<=feat_dict['bid_size1'][-2]), s_2_p_[-1], 0)
        feat_dict['sell_price_2'] = np.append(feat_dict['sell_price_2'] ,sell_price_2)
        feat_dict['sell_price_2'] = feat_dict['sell_price_2'][-2000:]
        sell_amount_2 = np.where((s_2_p_[-1]<=feat_dict['bid_price1'][-2])&(abs(s_2_v_[-1])<=feat_dict['bid_size1'][-2]), feat_dict['turnover'][-1]-feat_dict['turnover'][-2], np.nan)
        feat_dict['sell_amount_2'] = np.append(feat_dict['sell_amount_2'] ,sell_amount_2)
        feat_dict['sell_amount_2'] = feat_dict['sell_amount_2'][-2000:]
        if len(feat_dict['bid_price1'])>1000:
            mid_shift = (feat_dict['bid_price1'][-1000-1]+feat_dict['ask_price1'][-1000-1])/2
            sell_amount_agg_ratio_level2 = np.sum(asking[-1000:])/feat_dict['sell_amount_2'][-1]
            sell_price_bias_level2 = np.abs(feat_dict['sell_price_2'][-1] - mid_shift)/mid_shift
            # print('mid:',mid_shift,'下标:',index)
            feat_dict['sell_amount_agg_ratio_level2'] = np.append(feat_dict['sell_amount_agg_ratio_level2'], sell_amount_agg_ratio_level2)
            feat_dict['sell_amount_agg_ratio_level2'] = feat_dict['sell_amount_agg_ratio_level2'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['sell_amount_agg_ratio_level2'][-1], 'sell_amount_agg_ratio_level2',
                                          df, sum_num_, df_2d_)
            feat_dict['sell_price_bias_level2'] = np.append(feat_dict['sell_price_bias_level2'], sell_price_bias_level2)
            feat_dict['sell_price_bias_level2'] = feat_dict['sell_price_bias_level2'][-2000:]
            df_2d_ = test_data_consistency(feat_dict['sell_price_bias_level2'][-1], 'sell_price_bias_level2',
                                          df, sum_num_, df_2d_)
    
    
    if len(feat_dict['price'])>120:
        p = feat_dict['price'].copy()
        s = feat_dict['size'].copy()
        p = np.where(np.isnan(p),0,p)
        s = np.where(np.isnan(s),0,s)
        vwap = np.sum(p[-120:] * np.abs(s[-120:]))/np.sum(np.abs(s[-120:]))
        feat_dict['vwap'] = np.append(feat_dict['vwap'], vwap)
        feat_dict['vwap'] = feat_dict['vwap'][-2000:]
        df_2d_ = test_data_consistency(feat_dict['vwap'][-1], 'vwap',
                                      df, sum_num_, df_2d_)
    # print('df_2d_-----------',df_2d_[0])
    
    return df_2d_


# if __name__ == "__main__":
#     # pass
#     from minio import get_data_from_minio
#     import time
#     # minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
#                         # secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
#     # symbol = 'btcusdt'
#     symbol_list = 'ethusdt'
#     platform = 'gate_swap_u'
#     start_time = '2022-09-01-0'
#     end_time = '2022-09-10-12'
    
    
    
#     def get_data(platform, symbol_list, start_time, end_time):
    
#         def cumsum(df):
#             df['cum_size'] = np.cumsum(abs(df['size']))
#             df['turnover'] = np.cumsum(df['price'] * abs(df['size']))
#             return df
    
#         depth = get_data_from_minio('gate_swap_u', symbol_list, 'datafile/tick/order_book_100ms/gate_swap_u',
#                                        start_time=start_time, end_time=end_time)
#         depth = depth.iloc[:, 2:-6]
#         depth = depth.sort_values(by='closetime', ascending=True)

#         trade = get_data_from_minio('gate_swap_u', symbol_list, 'datafile/tick/trade/gate_swap_u',index_name='timestamp',
#                                        start_time=start_time, end_time=end_time)
#         trade = trade.iloc[:, :-3]
#         trade = trade.sort_values(by='dealid', ascending=True)
#         trade = trade.rename({'timestamp': 'closetime'}, axis='columns')
#         trade = trade.loc[:, ['closetime', 'price', 'size']]
#         trade['datetime'] = pd.to_datetime(trade['closetime'] + 28800000, unit='ms')
#         trade = trade.set_index('datetime').groupby(pd.Grouper(freq='1D')).apply(cumsum)
#         trade = trade[(trade['closetime'] >= depth['closetime'].iloc[0]) & (trade['closetime'] <= depth['closetime'].iloc[-1])]
#         trade = trade.reset_index(drop=True)
#         data_merge = pd.merge(depth, trade, how='outer', on='closetime')
#         data_merge.sort_values(by='closetime', ascending=True, inplace=True)
#         data_merge['datetime'] = pd.to_datetime(data_merge['closetime'] + 28800000, unit='ms')
#         data = data_merge.set_index('datetime').groupby(pd.Grouper(freq='1000ms')).apply('last')
        
#         return data
    
#     # tick_1s = get_data(platform=platform, symbol_list=symbol_list, start_time=start_time, end_time=end_time)
#     data_agg = get_data_from_minio('gate_swap_u', symbol_list, 'datafile/feat/songhe/',index_name='closetime',
#                                        start_time=start_time, end_time=end_time)
    
#     data_agg.sort_values(by='closetime', ascending=True, inplace=True)
#     data_agg.drop(['platform','year', 'month', 'symbol'], axis=1, inplace=True)
    
#     # 这个是聚合的深度和订单流的数据
#     ori_list = data_agg.columns.to_list()
#     # 这个是聚合的深度和订单流的表头的列表
#     col_dict_ = {key: i for i, key in enumerate(ori_list)}  # 将列表中的列名的下标作为值 列名做为键
#     # 这个是因子的列表
#     feat_dict_ = {i: np.array([]) for i in ori_list}
#     agg_values = data_agg.values
#     t1 = time.time()
#     sum_num = 0
#     df_2d = np.atleast_2d(np.zeros(65))
#     for i in range(100):
#         sum_num = 0
#         # test_columns = []
#         # print(i)
#         sum_num = factor_calculation(agg_values[i], i, feat_dict_, sum_num, df_2d, col_dict_)
#     print(time.time() - t1)
        