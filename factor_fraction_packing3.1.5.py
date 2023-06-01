import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#%%  计算这一行基于bid和ask的wap
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def calc_wap3(df):
    wap = (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap4(df):
    wap = (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def calc_wap12(df):
    var1 = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    var2 = df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    den = df['bid_size1'] + df['ask_size1'] + df['bid_size2'] + df['ask_size2']
    return (var1+var2) / den

def calc_wap34(df):
    var1 = df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']
    var2 = df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']
    den = df['bid_size1'] + df['ask_size1'] + df['bid_size2'] + df['ask_size2']
    return (var1+var2) / den

def calc_swap1(df):
    return df['wap1'] - df['wap3']

def calc_swap12(df):
    return df['wap12'] - df['wap34']

def calc_tswap1(df):
    return -df['swap1'].diff()

def calc_tswap12(df):
    return -df['swap12'].diff()

def calc_wss12(df):
    ask = (df['ask_price1'] * df['ask_size1'] + df['ask_price2'] * df['ask_size2'])/(df['ask_size1']+df['ask_size2'])
    bid = (df['bid_price1'] * df['bid_size1'] + df['bid_price2'] * df['bid_size2'])/(df['bid_size1']+df['bid_size2'])
    mid = (df['ask_price1'] + df['bid_price1']) / 2
    return (ask - bid) / mid

def calc_tt1(df):
    p1 = df['ask_price1'] * df['ask_size1'] + df['bid_price1'] * df['bid_size1']
    p2 = df['ask_price2'] * df['ask_size2'] + df['bid_price2'] * df['bid_size2']
    return p2 - p1

def calc_price_impact(df):
    ask = (df['ask_price1'] * df['ask_size1'] + df['ask_price2'] * df['ask_size2'])/(df['ask_size1']+df['ask_size2'])
    bid = (df['bid_price1'] * df['bid_size1'] + df['bid_price2'] * df['bid_size2'])/(df['bid_size1']+df['bid_size2'])
    return (df['ask_price1'] - ask)/df['ask_price1'], (df['bid_price1'] - bid)/df['bid_price1']

# Calculate order book slope
def calc_slope(df):
    v0 = (df['bid_size1']+df['ask_size1'])/2
    p0 = (df['bid_price1']+df['ask_price1'])/2
    slope_bid = ((df['bid_size1']/v0)-1)/abs((df['bid_price1']/p0)-1)+(
                (df['bid_size2']/df['bid_size1'])-1)/abs((df['bid_price2']/df['bid_price1'])-1)
    slope_ask = ((df['ask_size1']/v0)-1)/abs((df['ask_price1']/p0)-1)+(
                (df['ask_size2']/df['ask_size1'])-1)/abs((df['ask_price2']/df['ask_price1'])-1)
    return (slope_bid+slope_ask)/2, abs(slope_bid-slope_ask)

# Calculate order book dispersion
def calc_dispersion(df):
    bspread = df['bid_price1'] - df['bid_price2']
    aspread = df['ask_price2'] - df['ask_price1']
    bmid = (df['bid_price1'] + df['ask_price1'])/2  - df['bid_price1']
    bmid2 = (df['bid_price1'] + df['ask_price1'])/2  - df['bid_price2']
    amid = df['ask_price1'] - (df['bid_price1'] + df['ask_price1'])/2
    amid2 = df['ask_price2'] - (df['bid_price1'] + df['ask_price1'])/2
    bdisp = (df['bid_size1']*bmid + df['bid_size2']*bspread)/(df['bid_size1']+df['bid_size2'])
    bdisp2 = (df['bid_size1']*bmid + df['bid_size2']*bmid2)/(df['bid_size1']+df['bid_size2'])
    adisp = (df['ask_size1']*amid + df['ask_size2']*aspread)/(df['ask_size1']+df['ask_size2'])
    adisp2 = (df['ask_size1']*amid + df['ask_size2']*amid2)/(df['ask_size1']+df['ask_size2'])
    return bspread, aspread, bmid, amid, bdisp, adisp, (bdisp + adisp)/2, (bdisp2 + adisp2)/2


# Calculate order book depth
def calc_depth(df):
    depth = df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1'] + df['bid_price2'] * df[
               'bid_size2'] + df['ask_price2'] * df['ask_size2']
    return depth

#  order flow imbalance
def calc_ofi(df):
    a = df['bid_size1']*np.where(df['bid_price1'].diff()>=0,1,0)
    b = df['bid_size1'].shift()*np.where(df['bid_price1'].diff()<=0,1,0)
    c = df['ask_size1']*np.where(df['ask_price1'].diff()<=0,1,0)
    d = df['ask_size1'].shift()*np.where(df['ask_price1'].diff()>=0,1,0)
    return a - b - c + d

# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(series):
    return np.log(series).diff()

# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))

def realized_quarticity(series):
    # return (np.sum(series**4)*series.shape[0]/3)
    return (series.count() / 3) * np.sum(series ** 4)

def reciprocal_transformation(series):
    return np.sqrt(1/series)*100000

def square_root_translation(series):
    return series**(1/2)

# Calculate the realized absolute variation
def realized_absvar(series):
    return np.sqrt(np.pi/(2*series.count()))*np.sum(np.abs(series))

# Calculate the realized skew
def realized_skew(series):
    return np.sqrt(series.count())*np.sum(series**3)/(realized_volatility(series)**3)

# Calculate the realized kurtosis
def realized_kurtosis(series):
    return series.count()*np.sum(series**4)/(realized_volatility(series)**4)

def book_preprocessor(data, rolling=60):

    df = data.loc[:,['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1',
       'ask_price2', 'ask_size2', 'bid_price2', 'bid_size2', 'ask_price3',
       'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4',
       'bid_price4', 'bid_size4']]

    # wap1 genetic functions
    df['wap1'] = calc_wap1(df)

    df['depth_1s_wap1_shift1_2_log_return'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(2))
    df['depth_1s_wap1_shift1_60_log_return'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(rolling))
    df['depth_1s_wap1_shift1_120_log_return'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(rolling*2))
    df['depth_1s_wap1_shift1_300_log_return'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(rolling*5))

    # realized volatility
    df['depth_1s_wap1_rolling_60_realized_volatility'] = df['wap1'].rolling(rolling).apply(realized_volatility)
    df['depth_1s_wap1_rolling_120_realized_volatility'] = df['wap1'].rolling(rolling * 2).apply(realized_volatility)
    df['depth_1s_wap1_rolling_300_realized_volatility'] = df['wap1'].rolling(rolling * 5).apply(realized_volatility)

    # realized absvar
    df['depth_1s_wap1_rolling_60_realized_absvar'] = df['wap1'].rolling(rolling).apply(realized_absvar)
    df['depth_1s_wap1_rolling_120_realized_absvar'] = df['wap1'].rolling(rolling * 2).apply(realized_absvar)
    df['depth_1s_wap1_rolling_300_realized_absvar'] = df['wap1'].rolling(rolling * 5).apply(realized_absvar)

    # realized skew
    df['depth_1s_wap1_rolling_60_realized_skew'] = df['wap1'].rolling(rolling).apply(realized_skew)
    df['depth_1s_wap1_rolling_120_realized_skew'] = df['wap1'].rolling(rolling * 2).apply(realized_skew)
    df['depth_1s_wap1_rolling_300_realized_skew'] = df['wap1'].rolling(rolling * 5).apply(realized_skew)

    # realized kurtosis
    df['depth_1s_wap1_rolling_60_realized_kurtosis'] = df['wap1'].rolling(rolling).apply(realized_kurtosis)
    df['depth_1s_wap1_rolling_120_realized_kurtosis'] = df['wap1'].rolling(rolling * 2).apply(realized_kurtosis)
    df['depth_1s_wap1_rolling_300_realized_kurtosis'] = df['wap1'].rolling(rolling * 5).apply(realized_kurtosis)

    df['depth_1s_wap1_shift1_2_diff'] = df['wap1'].shift(1) - df['wap1'].shift(2)
    df['depth_1s_wap1_shift1_60_diff'] = df['wap1'].shift(1) - df['wap1'].shift(rolling)
    df['depth_1s_wap1_shift1_120_diff'] = df['wap1'].shift(1) - df['wap1'].shift(rolling*2)
    df['depth_1s_wap1_shift1_300_diff'] = df['wap1'].shift(1) - df['wap1'].shift(rolling*5)

    # df['ewm_wap1_mean'] = pd.DataFrame.ewm(df['wap1'],span=rolling).mean()

    df['depth_1s_wap1_rolling_60_mean'] = df['wap1'].rolling(rolling).mean()
    df['depth_1s_wap1_rolling_60_sd'] = df['wap1'].rolling(rolling).std()
    df['depth_1s_wap1_rolling_60_min'] = df['wap1'].rolling(rolling).min()
    df['depth_1s_wap1_rolling_60_max'] = df['wap1'].rolling(rolling).max()
    df['depth_1s_wap1_rolling_60_skew'] = df['wap1'].rolling(rolling).skew()
    df['depth_1s_wap1_rolling_60_kurt'] = df['wap1'].rolling(rolling).kurt()

    df['depth_1s_wap1_rolling_60_quantile_25'] = df['wap1'].rolling(rolling).quantile(.25)
    df['depth_1s_wap1_rolling_120_quantile_25'] = df['wap1'].rolling(rolling*2).quantile(.25)
    df['depth_1s_wap1_rolling_60_quantile_75'] = df['wap1'].rolling(rolling).quantile(.75)
    df['depth_1s_wap1_rolling_120_quantile_75'] = df['wap1'].rolling(rolling*2).quantile(.75)


    # wap2
    df['wap2'] = calc_wap2(df)

    df['depth_1s_wap2_shift1_2_log_return'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(2))
    df['depth_1s_wap2_shift1_60_log_return'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(rolling))
    df['depth_1s_wap2_shift1_120_log_return'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(rolling*2))
    df['depth_1s_wap2_shift1_300_log_return'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(rolling*5))

    # realized volatility
    df['depth_1s_wap2_rolling_60_realized_volatility'] = df['wap2'].rolling(rolling).apply(realized_volatility)
    df['depth_1s_wap2_rolling_120_realized_volatility'] = df['wap2'].rolling(rolling * 2).apply(realized_volatility)
    df['depth_1s_wap2_rolling_300_realized_volatility'] = df['wap2'].rolling(rolling * 5).apply(realized_volatility)

    # realized absvar
    df['depth_1s_wap2_rolling_60_realized_absvar'] = df['wap2'].rolling(rolling).apply(realized_absvar)
    df['depth_1s_wap2_rolling_120_realized_absvar'] = df['wap2'].rolling(rolling * 2).apply(realized_absvar)
    df['depth_1s_wap2_rolling_300_realized_absvar'] = df['wap2'].rolling(rolling * 5).apply(realized_absvar)

    # realized skew
    df['depth_1s_wap2_rolling_60_realized_skew'] = df['wap2'].rolling(rolling).apply(realized_skew)
    df['depth_1s_wap2_rolling_120_realized_skew'] = df['wap2'].rolling(rolling * 2).apply(realized_skew)
    df['depth_1s_wap2_rolling_300_realized_skew'] = df['wap2'].rolling(rolling * 5).apply(realized_skew)

    # realized kurtosis
    df['depth_1s_wap2_rolling_60_realized_kurtosis'] = df['wap2'].rolling(rolling).apply(realized_kurtosis)
    df['depth_1s_wap2_rolling_120_realized_kurtosis'] = df['wap2'].rolling(rolling * 2).apply(realized_kurtosis)
    df['depth_1s_wap2_rolling_300_realized_kurtosis'] = df['wap2'].rolling(rolling * 5).apply(realized_kurtosis)

    df['depth_1s_wap2_shift1_2_diff'] = df['wap2'].shift(1) - df['wap2'].shift(2)
    df['depth_1s_wap2_shift1_60_diff'] = df['wap2'].shift(1) - df['wap2'].shift(rolling)
    df['depth_1s_wap2_shift1_120_diff'] = df['wap2'].shift(1) - df['wap2'].shift(rolling*2)
    df['depth_1s_wap2_shift1_300_diff'] = df['wap2'].shift(1) - df['wap2'].shift(rolling*5)

    # df['ewm_wap2_mean'] = pd.DataFrame.ewm(df['wap2'], span=rolling).mean()

    df['depth_1s_wap2_rolling_60_mean'] = df['wap2'].rolling(rolling).mean()
    df['depth_1s_wap2_rolling_60_sd'] = df['wap2'].rolling(rolling).std()
    df['depth_1s_wap2_rolling_60_min'] = df['wap2'].rolling(rolling).min()
    df['depth_1s_wap2_rolling_60_max'] = df['wap2'].rolling(rolling).max()
    df['depth_1s_wap2_rolling_60_skew'] = df['wap2'].rolling(rolling).skew()
    df['depth_1s_wap2_rolling_60_kurt'] = df['wap2'].rolling(rolling).kurt()

    df['depth_1s_wap2_rolling_60_quantile_25'] = df['wap2'].rolling(rolling).quantile(.25)
    df['depth_1s_wap2_rolling_120_quantile_25'] = df['wap2'].rolling(rolling * 2).quantile(.25)
    df['depth_1s_wap2_rolling_60_quantile_75'] = df['wap2'].rolling(rolling).quantile(.75)
    df['depth_1s_wap2_rolling_120_quantile_75'] = df['wap2'].rolling(rolling * 2).quantile(.75)


    # wap3 genetic functions
    df['wap3'] = calc_wap3(df)

    df['depth_1s_wap3_shift1_2_log_return'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(2))
    df['depth_1s_wap3_shift1_60_log_return'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(rolling))
    df['depth_1s_wap3_shift1_120_log_return'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(rolling*2))
    df['depth_1s_wap3_shift1_300_log_return'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(rolling*5))

    # realized volatility
    df['depth_1s_wap3_rolling_60_realized_volatility'] = df['wap3'].rolling(rolling).apply(realized_volatility)
    df['depth_1s_wap3_rolling_120_realized_volatility'] = df['wap3'].rolling(rolling * 2).apply(realized_volatility)
    df['depth_1s_wap3_rolling_300_realized_volatility'] = df['wap3'].rolling(rolling * 5).apply(realized_volatility)

    # realized absvar
    df['depth_1s_wap3_rolling_60_realized_absvar'] = df['wap3'].rolling(rolling).apply(realized_absvar)
    df['depth_1s_wap3_rolling_120_realized_absvar'] = df['wap3'].rolling(rolling * 2).apply(realized_absvar)
    df['depth_1s_wap3_rolling_300_realized_absvar'] = df['wap3'].rolling(rolling * 5).apply(realized_absvar)

    # realized skew
    df['depth_1s_wap3_rolling_60_realized_skew'] = df['wap3'].rolling(rolling).apply(realized_skew)
    df['depth_1s_wap3_rolling_120_realized_skew'] = df['wap3'].rolling(rolling * 2).apply(realized_skew)
    df['depth_1s_wap3_rolling_300_realized_skew'] = df['wap3'].rolling(rolling * 5).apply(realized_skew)

    # realized kurtosis
    df['depth_1s_wap3_rolling_60_realized_kurtosis'] = df['wap3'].rolling(rolling).apply(realized_kurtosis)
    df['depth_1s_wap3_rolling_120_realized_kurtosis'] = df['wap3'].rolling(rolling * 2).apply(realized_kurtosis)
    df['depth_1s_wap3_rolling_300_realized_kurtosis'] = df['wap3'].rolling(rolling * 5).apply(realized_kurtosis)

    df['depth_1s_wap3_shift1_2_diff'] = df['wap3'].shift(1) - df['wap3'].shift(2)
    df['depth_1s_wap3_shift1_60_diff'] = df['wap3'].shift(1) - df['wap3'].shift(rolling)
    df['depth_1s_wap3_shift1_120_diff'] = df['wap3'].shift(1) - df['wap3'].shift(rolling*2)
    df['depth_1s_wap3_shift1_300_diff'] = df['wap3'].shift(1) - df['wap3'].shift(rolling*5)

    # df['ewm_wap3_mean'] = pd.DataFrame.ewm(df['wap3'], span=rolling).mean()

    df['depth_1s_wap3_rolling_60_mean'] = df['wap3'].rolling(rolling).mean()
    df['depth_1s_wap3_rolling_60_sd'] = df['wap3'].rolling(rolling).std()
    df['depth_1s_wap3_rolling_60_min'] = df['wap3'].rolling(rolling).min()
    df['depth_1s_wap3_rolling_60_max'] = df['wap3'].rolling(rolling).max()
    df['depth_1s_wap3_rolling_60_skew'] = df['wap3'].rolling(rolling).skew()
    df['depth_1s_wap3_rolling_60_kurt'] = df['wap3'].rolling(rolling).kurt()

    df['depth_1s_wap3_rolling_60_quantile_25'] = df['wap3'].rolling(rolling).quantile(.25)
    df['depth_1s_wap3_rolling_120_quantile_25'] = df['wap3'].rolling(rolling * 2).quantile(.25)
    df['depth_1s_wap3_rolling_60_quantile_75'] = df['wap3'].rolling(rolling).quantile(.75)
    df['depth_1s_wap3_rolling_120_quantile_75'] = df['wap3'].rolling(rolling * 2).quantile(.75)


    # wap4 genetic functions
    df['wap4'] = calc_wap4(df)

    df['depth_1s_wap4_shift1_2_log_return'] = np.log(df['wap4'].shift(1) / df['wap4'].shift(2))
    df['depth_1s_wap4_shift1_60_log_return'] = np.log(df['wap4'].shift(1) / df['wap4'].shift(rolling))
    df['depth_1s_wap4_shift1_120_log_return'] = np.log(df['wap4'].shift(1) / df['wap4'].shift(rolling*2))
    df['depth_1s_wap4_shift1_300_log_return'] = np.log(df['wap4'].shift(1) / df['wap4'].shift(rolling*5))

    # realized volatility
    df['depth_1s_wap4_rolling_60_realized_volatility'] = df['wap4'].rolling(rolling).apply(realized_volatility)
    df['depth_1s_wap4_rolling_120_realized_volatility'] = df['wap4'].rolling(rolling * 2).apply(realized_volatility)
    df['depth_1s_wap4_rolling_300_realized_volatility'] = df['wap4'].rolling(rolling * 5).apply(realized_volatility)

    # realized absvar
    df['depth_1s_wap4_rolling_60_realized_absvar'] = df['wap4'].rolling(rolling).apply(realized_absvar)
    df['depth_1s_wap4_rolling_120_realized_absvar'] = df['wap4'].rolling(rolling * 2).apply(realized_absvar)
    df['depth_1s_wap4_rolling_300_realized_absvar'] = df['wap4'].rolling(rolling * 5).apply(realized_absvar)

    # realized skew
    df['depth_1s_wap4_rolling_60_realized_skew'] = df['wap4'].rolling(rolling).apply(realized_skew)
    df['depth_1s_wap4_rolling_120_realized_skew'] = df['wap4'].rolling(rolling * 2).apply(realized_skew)
    df['depth_1s_wap4_rolling_300_realized_skew'] = df['wap4'].rolling(rolling * 5).apply(realized_skew)

    # realized kurtosis
    df['depth_1s_wap4_rolling_60_realized_kurtosis'] = df['wap4'].rolling(rolling).apply(realized_kurtosis)
    df['depth_1s_wap4_rolling_120_realized_kurtosis'] = df['wap4'].rolling(rolling * 2).apply(realized_kurtosis)
    df['depth_1s_wap4_rolling_300_realized_kurtosis'] = df['wap4'].rolling(rolling * 5).apply(realized_kurtosis)

    df['depth_1s_wap4_shift1_2_diff'] = df['wap4'].shift(1) - df['wap4'].shift(2)
    df['depth_1s_wap4_shift1_60_diff'] = df['wap4'].shift(1) - df['wap4'].shift(rolling)
    df['depth_1s_wap4_shift1_120_diff'] = df['wap4'].shift(1) - df['wap4'].shift(rolling*2)
    df['depth_1s_wap4_shift1_300_diff'] = df['wap4'].shift(1) - df['wap4'].shift(rolling*5)

    # df['ewm_wap4_mean'] = pd.DataFrame.ewm(df['wap4'], span=rolling).mean()

    df['depth_1s_wap4_rolling_60_mean'] = df['wap4'].rolling(rolling).mean()
    df['depth_1s_wap4_rolling_60_sg'] = df['wap4'].rolling(rolling).std()
    df['depth_1s_wap4_rolling_60_min'] = df['wap4'].rolling(rolling).min()
    df['depth_1s_wap4_rolling_60_max'] = df['wap4'].rolling(rolling).max()
    df['depth_1s_wap4_rolling_60_skew'] = df['wap4'].rolling(rolling).skew()
    df['depth_1s_wap4_rolling_60_kurt'] = df['wap4'].rolling(rolling).kurt()

    df['depth_1s_wap4_rolling_60_quantile_25'] = df['wap4'].rolling(rolling).quantile(.25)
    df['depth_1s_wap4_rolling_120_quantile_25'] = df['wap4'].rolling(rolling * 2).quantile(.25)
    df['depth_1s_wap4_rolling_60_quantile_75'] = df['wap4'].rolling(rolling).quantile(.75)
    df['depth_1s_wap4_rolling_120_quantile_75'] = df['wap4'].rolling(rolling * 2).quantile(.75)


    df['wap_balance1'] = abs(df['wap1'] - df['wap2'])
    df['wap_balance2'] = abs(df['wap1'] - df['wap3'])
    df['wap_balance3'] = abs(df['wap2'] - df['wap3'])
    df['wap_balance4'] = abs(df['wap3'] - df['wap4'])


    df['wap12'] = calc_wap12(df)
    df['wap34'] = calc_wap34(df)

    df['swap1'] = calc_swap1(df)
    df['swap12'] = calc_swap12(df)

    df['depth_1s_swap1_shift_1_diff'] = calc_tswap1(df)
    df['depth_1s_swap12_shift_1_diff'] = calc_tswap12(df)

    df['wss12'] = calc_wss12(df)
    df['tt1'] = calc_tt1(df)

    df['price_impact1'], df['price_impact2'] = calc_price_impact(df)

    df['slope1'], df['slope2'] = calc_slope(df)

    df['bspread'] = df['bid_price1'] - df['bid_price2']
    df['aspread'] = df['ask_price2'] - df['ask_price1']
    df['bmid'] = (df['bid_price1'] + df['ask_price1']) / 2 - df['bid_price1']
    df['bmid2'] = (df['bid_price1'] + df['ask_price1']) / 2 - df['bid_price2']
    df['amid'] = df['ask_price1'] - (df['bid_price1'] + df['ask_price1']) / 2
    df['amid2'] = df['ask_price2'] - (df['bid_price1'] + df['ask_price1']) / 2
    df['bdisp'] = (df['bid_size1'] * df['bmid'] + df['bid_size2'] * df['bspread']) / (df['bid_size1'] + df['bid_size2'])
    df['bdisp2'] = (df['bid_size1'] * df['bmid'] + df['bid_size2'] * df['bmid2']) / (df['bid_size1'] + df['bid_size2'])
    df['adisp'] = (df['ask_size1'] * df['amid'] + df['ask_size2'] * df['aspread']) / (df['ask_size1'] + df['ask_size2'])
    df['adisp2'] = (df['ask_size1'] * df['amid'] + df['ask_size2'] * df['amid2']) / (df['ask_size1'] + df['ask_size2'])

    df['depth'] = calc_depth(df)

    df['ofi'] = calc_ofi(df)

    df['HR1'] = ((df['bid_price1'] - df['bid_price1'].shift(1)) - (df['ask_price1'] - df['ask_price1'].shift(1))) / (
                (df['bid_price1'] - df['bid_price1'].shift(1)) + (df['ask_price1'] - df['ask_price1'].shift(1)))

    df['depth_1s_HR1_rolling_60_mean'] = df['HR1'].rolling(rolling).mean()
    df['depth_1s_HR1_rolling_120_mean'] = df['HR1'].rolling(rolling * 2).mean()
    df['depth_1s_HR1_rolling_300_mean'] = df['HR1'].rolling(rolling * 5).mean()

    df['pre_vtA'] = np.where(df['ask_price1'] == df['ask_price1'].shift(1), df['ask_size1'] - df['ask_size1'].shift(1), 0)
    df['vtA'] = np.where(df['ask_price1'] > df['ask_price1'].shift(1), df['ask_size1'], df['pre_vtA'])
    df['pre_vtB'] = np.where(df['bid_price1'] == df['bid_price1'].shift(1), df['bid_size1'] - df['bid_size1'].shift(1), 0)
    df['vtB'] = np.where(df['bid_price1'] > df['bid_price1'].shift(1), df['bid_size1'], df['pre_vtB'])

    df['depth_1s_vtA_rolling_60_mean'] = df['vtA'].rolling(rolling).mean()
    df['depth_1s_vtA_rolling_120_mean'] = df['vtA'].rolling(rolling*2).mean()
    df['depth_1s_vtA_rolling_300_mean'] = df['vtA'].rolling(rolling*5).mean()

    df['depth_1s_vtB_rolling_60_mean'] = df['vtB'].rolling(rolling).mean()
    df['depth_1s_vtB_rolling_120_mean'] = df['vtB'].rolling(rolling * 2).mean()
    df['depth_1s_vtB_rolling_300_mean'] = df['vtB'].rolling(rolling * 5).mean()

    df['Oiab'] = df['vtB'] - df['vtA']
    df['Oiab_60'] = df['depth_1s_vtB_rolling_60_mean'] - df['depth_1s_vtA_rolling_60_mean']
    df['Oiab_120'] = df['depth_1s_vtB_rolling_120_mean'] - df['depth_1s_vtA_rolling_120_mean']
    df['Oiab_300'] = df['depth_1s_vtB_rolling_300_mean'] - df['depth_1s_vtA_rolling_300_mean']

    df['mid_price1'] = (df['ask_price1'] + df['bid_price1']) / 2
    df['depth_1s_mid_price1_rolling_60_mean'] = df['mid_price1'].rolling(rolling).mean()
    df['depth_1s_mid_price1_rolling_60_sd'] = df['mid_price1'].rolling(rolling).std()

    df['price_spread1'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['price_spread3'] = (df['ask_price3'] - df['bid_price3']) / ((df['ask_price3'] + df['bid_price3']) / 2)
    df['price_spread4'] = (df['ask_price4'] - df['bid_price4']) / ((df['ask_price4'] + df['bid_price4']) / 2)

    df['bid_ask_size1_minus'] = df['bid_size1'] - df['ask_size1']
    df['bid_ask_size1_plus'] = df['bid_size1'] + df['ask_size1']
    df['bid_ask_size2_minus'] = df['bid_size2'] - df['ask_size2']
    df['bid_ask_size2_plus'] = df['bid_size2'] + df['ask_size2']
    df['bid_ask_size3_minus'] = df['bid_size3'] - df['ask_size3']
    df['bid_ask_size3_plus'] = df['bid_size3'] + df['ask_size3']
    df['bid_ask_size4_minus'] = df['bid_size4'] - df['ask_size4']
    df['bid_ask_size4_plus'] = df['bid_size4'] + df['ask_size4']

    df['depth_1s_bid_size1_shift_1_diff'] = df['bid_size1'] - df['bid_size1'].shift()
    df['depth_1s_bid_size1_shift_120_diff'] = df['bid_size1'] - df['bid_size1'].shift(rolling*2)
    df['depth_1s_bid_size1_shift_300_diff'] = df['bid_size1'] - df['bid_size1'].shift(rolling*5)
    df['depth_1s_ask_size1_shift_1_diff'] = df['ask_size1'] - df['ask_size1'].shift()
    df['depth_1s_ask_size1_shift_120_diff'] = df['ask_size1'] - df['ask_size1'].shift(rolling*2)
    df['depth_1s_ask_size1_shift_300_diff'] = df['ask_size1'] - df['ask_size1'].shift(rolling*5)
    df['depth_1s_bid_size2_shift_1_diff'] = df['bid_size2'] - df['bid_size2'].shift()
    df['depth_1s_bid_size2_shift_120_diff'] = df['bid_size2'] - df['bid_size2'].shift(rolling*2)
    df['depth_1s_bid_size2_shift_300_diff'] = df['bid_size2'] - df['bid_size2'].shift(rolling*5)
    df['depth_1s_ask_size2_shift_1_diff'] = df['ask_size2'] - df['ask_size2'].shift()
    df['depth_1s_ask_size2_shift_12_diff'] = df['ask_size2'] - df['ask_size2'].shift(rolling*2)
    df['depth_1s_ask_size2_shift_300_diff'] = df['ask_size2'] - df['ask_size2'].shift(rolling*5)
    df['depth_1s_bid_size3_shift_1_diff'] = df['bid_size3'] - df['bid_size3'].shift()
    df['depth_1s_ask_size3_shift_1_diff'] = df['ask_size3'] - df['ask_size3'].shift()

    df['bid_ask_size1_spread'] = df['bid_ask_size1_minus'] / df['bid_ask_size1_plus']
    df['bid_ask_size2_spread'] = df['bid_ask_size2_minus'] / df['bid_ask_size2_plus']
    df['bid_ask_size3_spread'] = df['bid_ask_size3_minus'] / df['bid_ask_size3_plus']
    df['bid_ask_size4_spread'] = df['bid_ask_size4_minus'] / df['bid_ask_size4_plus']

    df['depth_1s_bid_ask_size1_minus_rolling_60_mean'] = df['bid_ask_size1_minus'].rolling(rolling).mean()
    df['depth_1s_bid_ask_size1_minus_rolling_120_mean'] = df['bid_ask_size1_minus'].rolling(rolling*2).mean()
    df['depth_1s_bid_ask_size2_minus_rolling_60_mean'] = df['bid_ask_size2_minus'].rolling(rolling).mean()
    df['depth_1s_bid_ask_size2_minus_rolling_120_mean'] = df['bid_ask_size2_minus'].rolling(rolling*2).mean()
    df['depth_1s_bid_ask_size3_minus_rolling_60_mean'] = df['bid_ask_size3_minus'].rolling(rolling).mean()
    df['depth_1s_bid_ask_size3_minus_rolling_120_mean'] = df['bid_ask_size3_minus'].rolling(rolling*2).mean()

    df['depth_1s_bid_size1_shift_1_diff_rolling_60_mean'] = df['depth_1s_bid_size1_shift_1_diff'].rolling(rolling).mean()
    df['depth_1s_bid_size1_shift_1_diff_rolling_120_mean'] = df['depth_1s_bid_size1_shift_120_diff'].rolling(2 * rolling).mean()
    df['depth_1s_bid_size1_shift_1_diff_rolling_300_mean'] = df['depth_1s_bid_size1_shift_300_diff'].rolling(5 * rolling).mean()
    df['depth_1s_ask_size1_shift_1_diff_rolling_60_mean'] = df['depth_1s_ask_size1_shift_1_diff'].rolling(rolling).mean()
    df['depth_1s_ask_size1_shift_1_diff_rolling_120_mean'] = df['depth_1s_ask_size1_shift_120_diff'].rolling(2 * rolling).mean()
    df['depth_1s_ask_size1_shift_1_diff_rolling_300_mean'] = df['depth_1s_ask_size1_shift_300_diff'].rolling(5 * rolling).mean()
    df['depth_1s_bid_ask_size1_spread_rolling_60_mean'] = df['bid_ask_size1_spread'].rolling(rolling).mean()
    df['depth_1s_bid_ask_size1_spread_rolling_120_mean'] = df['bid_ask_size1_spread'].rolling(2 * rolling).mean()
    df['depth_1s_bid_ask_size1_spread_rolling_300_mean'] = df['bid_ask_size1_spread'].rolling(5 * rolling).mean()


    print(df.columns)

    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    df = df.replace(np.inf, 1)
    df = df.replace(-np.inf, -1)

    return df

def trade_preprocessor(data, rolling=60):

    df = data.loc[:, ['closetime', 'last_price', 'size', 'sell_vwap', 'sell_size', 'buy_vwap', 'buy_size']]

    # last price genetic functions
    # log_return
    df['log_return'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(2))
    df['depth_1s_last_price_shift1_60_log_return'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(rolling))
    df['depth_1s_last_price_shift1_120_log_return'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(rolling * 2))
    # realized volatility
    df['depth_1s_log_return_rolling_60_realized_volatility'] = df['log_return'].rolling(rolling).apply(realized_volatility)
    df['depth_1s_log_return_rolling_120_realized_volatility'] = df['log_return'].rolling(rolling * 2).apply(realized_volatility)
    df['depth_1s_log_return_rolling_300_realized_volatility'] = df['log_return'].rolling(rolling * 5).apply(realized_volatility)

    # realized absvar
    df['depth_1s_log_return_rolling_60_realized_absvar'] = df['log_return'].rolling(rolling).apply(realized_absvar)
    df['depth_1s_log_return_rolling_120_realized_absvar0'] = df['log_return'].rolling(rolling * 2).apply(realized_absvar)
    df['depth_1s_log_return_rolling_300_realized_absvar'] = df['log_return'].rolling(rolling * 5).apply(realized_absvar)
    # realized skew
    df['depth_1s_log_return_rolling_60_realized_skew'] = df['log_return'].rolling(rolling).apply(realized_skew)
    df['depth_1s_log_return_rolling_120_realized_skew'] = df['log_return'].rolling(rolling * 2).apply(realized_skew)
    df['depth_1s_log_return_rolling_300_realized_skew'] = df['log_return'].rolling(rolling * 5).apply(realized_skew)

    df['depth_1s_last_price_rolling_60_quantile_25'] = df['last_price'].rolling(rolling).quantile(.25)
    df['depth_1s_last_price_rolling_120_quantile_25'] = df['last_price'].rolling(rolling*2).quantile(.25)
    df['depth_1s_last_price_rolling_300_quantile_25'] = df['last_price'].rolling(rolling*5).quantile(.25)

    df['depth_1s_last_price_rolling_60_quantile_75'] = df['last_price'].rolling(rolling).quantile(.75)
    df['depth_1s_last_price_rolling_120_quantile_75'] = df['last_price'].rolling(rolling*2).quantile(.75)
    df['depth_1s_last_price_rolling_300_quantile_75'] = df['last_price'].rolling(rolling*5).quantile(.75)

    df['depth_1s_last_price_percentile_rolling_60'] = df['depth_1s_last_price_rolling_60_quantile_75'] - df['depth_1s_last_price_rolling_60_quantile_25']
    df['depth_1s_last_price_percentile_rolling_60'] = df['depth_1s_last_price_rolling_120_quantile_75'] - df['depth_1s_last_price_rolling_120_quantile_25']
    df['depth_1s_last_price_percentile_rolling_60'] = df['depth_1s_last_price_rolling_300_quantile_75'] - df['depth_1s_last_price_rolling_300_quantile_25']

    df['depth_1s_last_price_shift1_2_diff'] = df['last_price'].shift(1) - df['last_price'].shift(2)
    df['depth_1s_last_price_shift1_120_diff'] = df['last_price'].shift(1) - df['last_price'].shift(rolling*2)
    df['depth_1s_last_price_shift1_300_diff'] = df['last_price'].shift(1) - df['last_price'].shift(rolling*5)


    # size genetic functions
    # realized volatility
    df['depth_1s_size_rolling_60_realized_volatility'] = df['size'].rolling(rolling).apply(realized_volatility)
    df['depth_1s_size_rolling_120_realized_volatility'] = df['size'].rolling(rolling * 2).apply(realized_volatility)
    df['depth_1s_size_rolling_300_realized_volatility'] = df['size'].rolling(rolling * 5).apply(realized_volatility)
    # realized skew
    df['depth_1s_size_rolling_60_realized_skew'] = df['size'].rolling(rolling).apply(realized_skew)
    df['depth_1s_size_rolling_120_realized_skew'] = df['size'].rolling(rolling * 2).apply(realized_skew)
    df['depth_1s_size_rolling_300_realized_skew'] = df['size'].rolling(rolling * 5).apply(realized_skew)
    # realized kurtosis
    df['depth_1s_size_rolling_60_realized_kurtosis'] = df['size'].rolling(rolling).apply(realized_kurtosis)
    df['depth_1s_size_rolling_120_realized_kurtosis'] = df['size'].rolling(rolling * 2).apply(realized_kurtosis)
    df['depth_1s_size_rolling_300_realized_kurtosis'] = df['size'].rolling(rolling * 5).apply(realized_kurtosis)
    # realized absvar
    df['depth_1s_size_rolling_60_realized_absvar'] = df['size'].rolling(rolling).apply(realized_absvar)
    df['depth_1s_size_rolling_120_realized_absvar'] = df['size'].rolling(rolling * 2).apply(realized_absvar)
    df['depth_1s_size_rolling_300_realized_absvar'] = df['size'].rolling(rolling * 5).apply(realized_absvar)

    df['depth_1s_size_rolling_60_mean'] = df['size'].rolling(rolling).mean()
    df['depth_1s_size_rolling_60_var'] = df['size'].rolling(rolling).var()
    df['depth_1s_size_rolling_60_sd'] = df['size'].rolling(rolling).std()
    df['depth_1s_size_rolling_60_sum'] = df['size'].rolling(rolling).sum()
    df['depth_1s_size_rolling_60_min'] = df['size'].rolling(rolling).min()
    df['depth_1s_size_rolling_60_max'] = df['size'].rolling(rolling).max()
    df['depth_1s_size_rolling_60_skew'] = df['size'].rolling(rolling).skew()
    df['depth_1s_size_rolling_60_kurt'] = df['size'].rolling(rolling).kurt()
    df['depth_1s_size_rolling_60_median'] = df['size'].rolling(rolling).median()

    # df['ewm_mean_size'] = pd.DataFrame.ewm(df['size'], span=rolling).mean()
    # df['ewm_std_size'] = pd.DataFrame.ewm(df['size'], span=rolling).std()

    df['depth_1s_size_rolling_60_quantile_25'] = df['size'].rolling(rolling).quantile(.25)
    df['depth_1s_size_rolling_120_quantile_25'] = df['size'].rolling(rolling*2).quantile(.25)
    df['depth_1s_size_rolling_300_quantile_25'] = df['size'].rolling(rolling*5).quantile(.25)
    df['depth_1s_size_rolling_60_quantile_75'] = df['size'].rolling(rolling).quantile(.75)
    df['depth_1s_size_rolling_120_quantile_75'] = df['size'].rolling(rolling*2).quantile(.75)
    df['depth_1s_size_rolling_300_quantile_75'] = df['size'].rolling(rolling*5).quantile(.75)
    df['depth_1s_size_percentile_rolling_60'] = df['depth_1s_size_rolling_60_quantile_75'] - df['depth_1s_size_rolling_60_quantile_25']
    df['depth_1s_size_percentile_rolling_120'] = df['depth_1s_size_rolling_120_quantile_75'] - df['depth_1s_size_rolling_120_quantile_25']
    df['depth_1s_size_percentile_rolling_300'] = df['depth_1s_size_rolling_300_quantile_75'] - df['depth_1s_size_rolling_300_quantile_25']


    # amount genetic functions
    df['amount'] = df['last_price'] * df['size']

    df['mid_price'] = np.where(df['size'] > 0, (df['amount'] - df['amount'].shift(1)) / df['size'], df['last_price'])
    df['depth_1s_mid_price_rolling_60_mean'] = df['mid_price'].rolling(rolling).mean()
    df['depth_1s_mid_price_rolling_60_sd'] = df['mid_price'].rolling(rolling).std()

    df['depth_1s_amount_rolling_60_mean'] = df['amount'].rolling(rolling).mean()
    df['depth_1s_amount_rolling_120_mean'] = df['amount'].rolling(rolling * 2).mean()
    df['depth_1s_amount_rolling_300_mean'] = df['amount'].rolling(rolling * 5).mean()
    df['depth_1s_amount_rolling_60_quantile_25'] = df['amount'].rolling(rolling).quantile(.25)
    df['depth_1s_amount_rolling_120_quantile_25'] = df['amount'].rolling(rolling * 2).quantile(.25)
    df['depth_1s_amount_rolling_300_quantile_25'] = df['amount'].rolling(rolling * 5).quantile(.25)
    df['depth_1s_amount_rolling_60_quantile_75'] = df['amount'].rolling(rolling).quantile(.75)
    df['depth_1s_amount_rolling_120_quantile_75'] = df['amount'].rolling(rolling * 2).quantile(.75)
    df['depth_1s_amount_rolling_300_quantile_75'] = df['amount'].rolling(rolling * 5).quantile(.75)


    # buy vwap genetic functions
    df['depth_1s_buy_vwap_shift1_2_diff'] = df['buy_vwap'].shift(1) - df['buy_vwap'].shift(2)
    df['depth_1s_buy_vwap_shift1_120_diff'] = df['buy_vwap'].shift(1) - df['buy_vwap'].shift(rolling * 2)
    df['depth_1s_buy_vwap_shift1_300_diff'] = df['buy_vwap'].shift(1) - df['buy_vwap'].shift(rolling * 5)
    df['log_return_buy_vwap'] = np.log(df['buy_vwap'].shift(1) / df['buy_vwap'].shift(2))
    df['depth_1s_buy_vwap_shift1_120_log_return'] = np.log(df['buy_vwap'].shift(1) / df['buy_vwap'].shift(rolling*2))
    df['depth_1s_buy_vwap_shift1_300_log_return'] = np.log(df['buy_vwap'].shift(1) / df['buy_vwap'].shift(rolling * 5))

    df['depth_1s_log_return_buy_vwap_rolling_60_realized_volatility'] = df['log_return_buy_vwap'].rolling(rolling).apply(realized_volatility)
    df['depth_1s_log_return_buy_vwap_rolling_120_realized_volatility'] = df['log_return_buy_vwap'].rolling(rolling * 2).apply(
        realized_volatility)
    df['depth_1s_log_return_buy_vwap_rolling_300_realized_volatility'] = df['log_return_buy_vwap'].rolling(rolling * 5).apply(
        realized_volatility)

    # realized absvar
    df['depth_1s_log_return_buy_vwap_rolling_60_realized_absvar'] = df['log_return_buy_vwap'].rolling(rolling).apply(realized_absvar)
    df['depth_1s_log_return_buy_vwap_rolling_120_realized_absvar'] = df['log_return_buy_vwap'].rolling(rolling * 2).apply(realized_absvar)
    df['depth_1s_log_return_buy_vwap_rolling_300_realized_absvar'] = df['log_return_buy_vwap'].rolling(rolling * 5).apply(realized_absvar)
    # realized skew
    df['depth_1s_log_return_buy_vwap_rolling_60_realized_skew'] = df['log_return_buy_vwap'].rolling(rolling).apply(realized_skew)
    df['depth_1s_log_return_buy_vwap_rolling_120_realized_skew'] = df['log_return_buy_vwap'].rolling(rolling * 2).apply(realized_skew)
    df['depth_1s_log_return_buy_vwap_rolling_300_realized_skew'] = df['log_return_buy_vwap'].rolling(rolling * 5).apply(realized_skew)

    df['depth_1s_buy_vwap_rolling_60_quantile_25'] = df['buy_vwap'].rolling(rolling).quantile(.25)
    df['depth_1s_buy_vwap_rolling_120_quantile_25'] = df['buy_vwap'].rolling(rolling * 2).quantile(.25)
    df['depth_1s_buy_vwap_rolling_300_quantile_25'] = df['buy_vwap'].rolling(rolling * 5).quantile(.25)

    df['depth_1s_buy_vwap_rolling_60_quantile_75'] = df['buy_vwap'].rolling(rolling).quantile(.75)
    df['depth_1s_buy_vwap_rolling_120_quantile_75'] = df['buy_vwap'].rolling(rolling * 2).quantile(.75)
    df['depth_1s_buy_vwap_rolling_300_quantile_75'] = df['buy_vwap'].rolling(rolling * 5).quantile(.75)

    df['depth_1s_buy_vwap_percentile_rolling_60'] = df['depth_1s_buy_vwap_rolling_60_quantile_75'] - df['depth_1s_buy_vwap_rolling_60_quantile_25']
    df['depth_1s_buy_vwap_percentile_rolling_120'] = df['depth_1s_buy_vwap_rolling_120_quantile_75'] - df['depth_1s_buy_vwap_rolling_120_quantile_25']
    df['depth_1s_buy_vwap_percentile_rolling_300'] = df['depth_1s_buy_vwap_rolling_300_quantile_75'] - df['depth_1s_buy_vwap_rolling_300_quantile_25']


    # sell vwap genetic functions
    df['depth_1s_sell_vwap_shift1_2_diff'] = df['sell_vwap'].shift(1) - df['sell_vwap'].shift(2)
    df['depth_1s_sell_vwap_shift1_120_diff'] = df['sell_vwap'].shift(1) - df['sell_vwap'].shift(rolling * 2)
    df['depth_1s_sell_vwap_shift1_300_diff'] = df['sell_vwap'].shift(1) - df['sell_vwap'].shift(rolling * 5)
    df['log_return_sell_vwap'] = np.log(df['sell_vwap'].shift(1) / df['sell_vwap'].shift(2))
    df['depth_1s_sell_vwap_shift1_120_log_return'] = np.log(df['sell_vwap'].shift(1) / df['sell_vwap'].shift(rolling*2))
    df['depth_1s_sell_vwap_shift1_300_log_return'] = np.log(df['sell_vwap'].shift(1) / df['sell_vwap'].shift(rolling * 5))

    df['depth_1s_log_return_sell_vwap_rolling_60_realized_volatility'] = df['log_return_sell_vwap'].rolling(rolling).apply(realized_volatility)
    df['depth_1s_log_return_sell_vwap_rolling_120_realized_volatility'] = df['log_return_sell_vwap'].rolling(rolling * 2).apply(realized_volatility)
    df['depth_1s_log_return_sell_vwap_rolling_300_realized_volatility'] = df['log_return_sell_vwap'].rolling(rolling * 5).apply(realized_volatility)

    # realized absvar
    df['depth_1s_log_return_sell_vwap_rolling_60_realized_absvar'] = df['log_return_sell_vwap'].rolling(rolling).apply(realized_absvar)
    df['depth_1s_log_return_sell_vwap_rolling_120_realized_absvar'] = df['log_return_sell_vwap'].rolling(rolling * 2).apply(realized_absvar)
    df['depth_1s_log_return_sell_vwap_rolling_300_realized_absvar'] = df['log_return_sell_vwap'].rolling(rolling * 5).apply(realized_absvar)
    # realized skew
    df['depth_1s_log_return_sell_vwap_rolling_60_realized_skew'] = df['log_return_sell_vwap'].rolling(rolling).apply(realized_skew)
    df['depth_1s_log_return_sell_vwap_rolling_120_realized_skew'] = df['log_return_sell_vwap'].rolling(rolling * 2).apply(realized_skew)
    df['depth_1s_log_return_sell_vwap_rolling_300_realized_skew'] = df['log_return_sell_vwap'].rolling(rolling * 5).apply(realized_skew)

    df['depth_1s_sell_vwap_rolling_60_quantile_25'] = df['sell_vwap'].rolling(rolling).quantile(.25)
    df['depth_1s_sell_vwap_rolling_120_quantile_25'] = df['sell_vwap'].rolling(rolling * 2).quantile(.25)
    df['depth_1s_sell_vwap_rolling_300_quantile_25'] = df['sell_vwap'].rolling(rolling * 5).quantile(.25)

    df['depth_1s_sell_vwap_rolling_60_quantile_75'] = df['sell_vwap'].rolling(rolling).quantile(.75)
    df['depth_1s_sell_vwap_rolling_120_quantile_75'] = df['sell_vwap'].rolling(rolling * 2).quantile(.75)
    df['depth_1s_sell_vwap_rolling_300_quantile_75'] = df['sell_vwap'].rolling(rolling * 5).quantile(.75)

    df['depth_1s_sell_vwap_percentile_rolling_60'] = df['depth_1s_sell_vwap_rolling_60_quantile_75'] - df['depth_1s_sell_vwap_rolling_60_quantile_25']
    df['depth_1s_sell_vwap_percentile_rolling_120'] = df['depth_1s_sell_vwap_rolling_120_quantile_75'] - df['depth_1s_sell_vwap_rolling_120_quantile_25']
    df['depth_1s_sell_vwap_percentile_rolling_300'] = df['depth_1s_sell_vwap_rolling_300_quantile_75'] - df['depth_1s_sell_vwap_rolling_300_quantile_25']

    # vwap mid price
    df['vwap_mid_price'] = (df['sell_vwap'] + df['buy_vwap']) / 2
    df['depth_1s_vwap_mid_price_rolling_60_mean'] = df['vwap_mid_price'].rolling(rolling).mean()
    df['depth_1s_vwap_mid_price_rolling_60_sd'] = df['vwap_mid_price'].rolling(rolling).std()

    df['trade_price_minus'] = df['sell_vwap'] - df['buy_vwap']
    df['trade_price_plus'] = df['sell_vwap'] + df['buy_vwap']
    df['vwap_spread'] = df['trade_price_minus'] / df['trade_price_plus']
    df['depth_1s_vwap_spread_rolling_60_mean'] = df['vwap_spread'].rolling(rolling).mean()
    df['depth_1s_vwap_spread_rolling_120_mean'] = df['vwap_spread'].rolling(rolling * 2).mean()
    df['depth_1s_vwap_spread_rolling_300_mean'] = df['vwap_spread'].rolling(rolling * 5).mean()


    # buy/sell amount
    df['sell_vwap_amount'] = df['sell_vwap']*df['sell_size']
    df['buy_vwap_amount'] = df['buy_vwap']*df['buy_size']

    df['depth_1s_sell_vwap_amount_rolling_60_mean'] = df['sell_vwap_amount'].rolling(rolling).mean()
    df['depth_1s_sell_vwap_amount_rolling_120_mean'] = df['sell_vwap_amount'].rolling(rolling * 2).mean()
    df['depth_1s_sell_vwap_amount_rolling_300_mean'] = df['sell_vwap_amount'].rolling(rolling * 5).mean()
    df['depth_1s_sell_vwap_amount_rolling_60_quantile_25'] = df['sell_vwap_amount'].rolling(rolling).quantile(.25)
    df['depth_1s_sell_vwap_amount_rolling_120_quantile_25'] = df['sell_vwap_amount'].rolling(rolling * 2).quantile(.25)
    df['depth_1s_sell_vwap_amount_rolling_300_quantile_25'] = df['sell_vwap_amount'].rolling(rolling * 5).quantile(.25)
    df['depth_1s_sell_vwap_amount_rolling_60_quantile_75'] = df['sell_vwap_amount'].rolling(rolling).quantile(.75)
    df['depth_1s_sell_vwap_amount_rolling_120_quantile_75'] = df['sell_vwap_amount'].rolling(rolling * 2).quantile(.75)
    df['depth_1s_sell_vwap_amount_rolling_300_quantile_75'] = df['sell_vwap_amount'].rolling(rolling * 5).quantile(.75)

    df['depth_1s_buy_vwap_amount_rolling_60_mean'] = df['buy_vwap_amount'].rolling(rolling).mean()
    df['depth_1s_buy_vwap_amount_rolling_120_mean'] = df['buy_vwap_amount'].rolling(rolling * 2).mean()
    df['depth_1s_buy_vwap_amount_rolling_300_mean'] = df['buy_vwap_amount'].rolling(rolling * 5).mean()
    df['depth_1s_buy_vwap_amount_rolling_60_quantile_25'] = df['buy_vwap_amount'].rolling(rolling).quantile(.25)
    df['depth_1s_buy_vwap_amount_rolling_120_quantile_25'] = df['buy_vwap_amount'].rolling(rolling * 2).quantile(.25)
    df['depth_1s_buy_vwap_amount_rolling_300_quantile_25'] = df['buy_vwap_amount'].rolling(rolling * 5).quantile(.25)
    df['depth_1s_buy_vwap_amount_rolling_60_quantile_75'] = df['buy_vwap_amount'].rolling(rolling).quantile(.75)
    df['depth_1s_buy_vwap_amount_rolling_120_quantile_75'] = df['buy_vwap_amount'].rolling(rolling * 2).quantile(.75)
    df['depth_1s_buy_vwap_amount_rolling_300_quantile_75'] = df['buy_vwap_amount'].rolling(rolling * 5).quantile(.75)

    df['buy_sell_vwap_amount_minus'] = df['sell_vwap_amount']-df['buy_vwap_amount']
    df['buy_sell_vwap_amount_plus'] = df['sell_vwap_amount'] + df['buy_vwap_amount']
    df['buy_sell_vwap_amount_spread'] = df['buy_sell_vwap_amount_minus'] / df['buy_sell_vwap_amount_plus']
    df['depth_1s_buy_sell_vwap_amount_spread_rolling_60_mean'] = df['buy_sell_vwap_amount_spread'].rolling(rolling).mean()
    df['depth_1s_buy_sell_vwap_amount_spread_rolling_120_mean'] = df['buy_sell_vwap_amount_spread'].rolling(rolling*2).mean()
    df['depth_1s_buy_sell_vwap_amount_spread_rolling_300_mean'] = df['buy_sell_vwap_amount_spread'].rolling(rolling*5).mean()

    df['ewm_mean_amount'] = pd.DataFrame.ewm(df['amount'], span=rolling).mean()

    print(df.columns)

    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    df = df.replace(np.inf, 1)
    df = df.replace(-np.inf, -1)

    return df

# #%% 计算输出因子
#
# book_data = book_preprocessor()
# trade_data = trade_preprocessor()
# book_data['datetime'] = pd.to_datetime(book_data['datetime'])
# trade_data['datetime'] = pd.to_datetime(trade_data['datetime'])
# #%% 计算这一分钟的vwap为target,且为挂单价格
# def get_vwap(data):
#     v = data['size']
#     p = data['last_price']
#     data['last_price_vwap'] = np.sum(p*v) / np.sum(v)
#     return data
#
# time_group = trade_data.set_index('datetime').groupby(pd.Grouper(freq='1min')).apply(get_vwap)
# # 通过trade产出的因子将数据进项聚合为1分钟的数据
# time_group_trade = time_group.groupby(pd.Grouper(freq='1min')).agg(np.mean)
# time_group_trade = time_group_trade.dropna(axis=0,how='all')
# time_group_trade = time_group_trade.reset_index()    #time_group为最终传入模型的因子
# #%% 通过book产出的因子聚合为1分钟的数据
# time_group_book = book_data.set_index('datetime').groupby(pd.Grouper(freq='1min')).agg(np.mean)
# time_group_book = time_group_book.dropna(axis=0,how='all')
# time_group_book = time_group_book.reset_index()