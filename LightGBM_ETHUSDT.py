#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import accuracy_score,roc_auc_score, classification_report
from sklearn import metrics
from tqdm import tqdm
import seaborn as sns
import warnings
import keras
import joblib
from scipy.stats import ks_2samp
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings("ignore")
import time
from functools import reduce
import pyarrow
from pyarrow import fs
import pyarrow.parquet as pq
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
from scipy.stats import pearsonr

# add_factor['vwap'] = (add_factor['price'].fillna(0)*abs(add_factor['size'].fillna(0))).rolling(120).sum()/abs(add_factor['size'].fillna(0)).rolling(120).sum()
# add_factor['vwap'] = (add_factor['wap1'].fillna(0)*abs(add_factor['size'].fillna(0))).rolling(120).sum()/abs(add_factor['size'].fillna(0)).rolling(120).sum()
#%% time bar
add_factor['datetime'] = pd.to_datetime(add_factor['closetime']+28800000, unit='ms')
data = add_factor.set_index('datetime').groupby(pd.Grouper(freq='5min')).apply('last')
data = data.dropna(axis=0)
#%% volume/dollar bar
def dollar_bars(df, dv_column, m):
    '''
    compute dollar bars

    # args
        df: pd.DataFrame()
        dv_column: name for dollar volume data
        m: int(), threshold value for dollars
    # returns
        idx: list of indices
    '''
    t = df[dv_column]
    ts = 0
    idx = []
    # for i, x in enumerate(tqdm(t)):
    for i in tqdm(range(1, len(t))):
        if t[i] - t[i - 1] >= m:
            # print(t[i])
            idx.append(i)
            continue
        # ts += x
        # if ts >= m:
        #     idx.append(i)
        #     ts = 0
        # continue
    return idx

def dollar_bar_df(df, dv_column, m):
    idx = dollar_bars(df, dv_column, m)
    # print(df.iloc[idx])
    return df.iloc[idx].drop_duplicates()
#
add_factor['turnover'] = add_factor['turnover'].fillna(method='ffill')
# data_80 = dollar_bar_df(add_factor, 'turnover',80_000)
# data_100 = dollar_bar_df(add_factor, 'turnover',100_000)
# data_120 = dollar_bar_df(add_factor, 'turnover',120_000)
# data_130 = dollar_bar_df(add_factor, 'turnover',130_000)
# data_1300 = dollar_bar_df(add_factor, 'turnover',1_300_000)
# data_110 = dollar_bar_df(add_factor, 'turnover',110_000)
# data_50 = dollar_bar_df(add_factor, 'turnover',50_000)
# data_60 = dollar_bar_df(add_factor, 'turnover',60_000)
# data_70 = dollar_bar_df(add_factor, 'turnover',70_000)
# del add_factor,
#%%
price_max_min = add_factor.set_index('datetime').groupby(pd.Grouper(freq='1min')).agg({'price':['min','max'], 'closetime':'last'})
price_max_min.columns = ['_'.join(col) for col in price_max_min.columns]
price_max_min = price_max_min.rename({'closetime_last':'closetime', 'price_max':'high', 'price_min':'low'},axis='columns')
price_max_min = price_max_min.dropna(axis=0)

data = pd.merge(data, price_max_min, on='closetime', how='left')
data['datetime'] = pd.to_datetime(data['closetime']+28800000, unit='ms')
data = data.set_index('datetime')

#%%
data['mid'] = (data['ask_price1']+data['bid_price1'])/2
#%%
df = data.iloc[:10,:]
# df['ask'] = df['ask_price1']*(1-0.00005)
# df['bid'] = df['bid_price1']*(1+0.00005)
df[['ask_price1', 'bid_price1','wap1','price']].plot()
plt.show()
#%%
# data['mid'] = (data['ask_price1']+data['bid_price1'])/2
data['buy_success'] = (data['bid_price1'].shift(1))*1.00003-data['price_min']
data['sell_success'] = data['price_max']-(data['ask_price1'].shift(1))*0.99997
data['buy_success'] = np.where(data['buy_success']>0, 1, data['buy_success'])
data['sell_success'] = np.where(data['sell_success']>0, 1, data['sell_success'])

data['target'] = data['buy_success']*data['sell_success']
print(len(data[data['target'].isnull().values==True])/len(data['target']))
print(len(data[data['target']==1])/len(data['target']))
#%%
def getDailyVol(close,span0=100):
    # daily vol, reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    a = df0 -1 #using a variable to avoid the error message.
    df0=pd.Series(close.index[a],
                  index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1
    # daily returns
    df0=df0.ewm(span=span0).std()
    return df0

def get_Daily_Volatility(close,span0=20):
    # simple percentage returns
    df0=close.pct_change()
    # 20 days, a month EWM's std as boundary
    df0=df0.ewm(span=span0).std()
    df0.dropna(inplace=True)
    return df0

#set the boundary of barriers, based on 15min EWM
daily_volatility = get_Daily_Volatility(data['price'], span0=15)
# how many times we hold the stock which set the vertical barrier
t_final = 15
#the up and low boundary multipliers
upper_lower_multipliers = [3, 1]
#allign the index
prices = data.loc[daily_volatility.index]

def get_3_barriers():
    #create a container
    barriers = pd.DataFrame(columns=['days_passed',
              'price', 'vert_barrier',
              'top_barrier', 'bottom_barrier'],
               index = daily_volatility.index)
    for day, vol in (tqdm(daily_volatility.iteritems())):
        days_passed = len(daily_volatility.loc[daily_volatility.index[0] : day])
        #set the vertical barrier
        if (days_passed + t_final < len(daily_volatility.index)and t_final != 0):
            vert_barrier = daily_volatility.index[days_passed + t_final]
        else:
            vert_barrier = np.nan
        #set the top barrier
        close = prices['price']
        if upper_lower_multipliers[0] > 0:
            top_barrier = close[day] + close[day] * upper_lower_multipliers[0] * vol
        else:
            #set it to NaNs
            top_barrier = pd.Series(index=prices.index)
        #set the bottom barrier
        if upper_lower_multipliers[1] > 0:
            bottom_barrier = close[day] - close[day] * upper_lower_multipliers[1] * vol
        else:
            #set it to NaNs
            bottom_barrier = pd.Series(index=prices.index)
        barriers.loc[day, ['days_passed', 'vert_barrier','top_barrier', 'bottom_barrier']] = days_passed, vert_barrier,top_barrier, bottom_barrier
    return barriers
barriers = get_3_barriers()
barriers['price'] = prices['price']
barriers['out'] = None

def get_labels(df):
    '''
    start: first day of the window
    end:last day of the window
    price_initial: first day stock price
    price_final:last day stock price
    top_barrier: profit taking limit
    bottom_barrier:stop loss limt
    condition_pt:top_barrier touching conditon
    condition_sl:bottom_barrier touching conditon
    '''
    barriers = df
    for i in range(len(barriers.index)):
        start = barriers.index[i]
        end = barriers.vert_barrier[i]
        if pd.notna(end):
            # assign the initial and final price
            price_initial = barriers.price[start]
            price_final = barriers.price[end]
            # assign the top and bottom barriers
            top_barrier = barriers.top_barrier[i]
            bottom_barrier = barriers.bottom_barrier[i]
            #set the profit taking and stop loss conditons
            condition_pt = (barriers.price[start: end] >= top_barrier).any()
            condition_sl = (barriers.price[start: end] <= bottom_barrier).any()
            #assign the labels
            if condition_pt:
                barriers['out'][i] = 1
            else:
                # condition_sl:
                barriers['out'][i] = 0
            # else:
                # barriers['out'][i] = 2
                # barriers['out'][i] = max([(price_final - price_initial)/(top_barrier - price_initial),
                #            (price_final - price_initial)/ (price_initial - bottom_barrier)],
                #             key=abs)
    return df

barriers2 = barriers.copy()
barriers2 = get_labels(barriers2)
print(barriers2.out.value_counts())

data['target'] = barriers2['out']
data['target'] = data['target'].astype('float64')
data = data.dropna(axis=0)
#%%
bar = 5
# data['target'] = data['bid_price1'].shift(1)-data['ask_price1']
data['target'] = np.log(data['wap1']/data['wap1'].shift(bar))
# data = data.drop(['buy_success', 'sell_success'], axis=1)
data['target'] = data['target'].shift(-bar)
def classify(y):

    if y < -0.0005:
        return 0
    if y > 0.0005:
        return 1
    else:
        return -1

print(data['target'].apply(lambda x:classify(x)).value_counts())
print(len(data[data['target'].apply(lambda x:classify(x))==-1])/len(data['target'].apply(lambda x:classify(x))))
#%%
def calcpearsonr(data,rolling):

    ic_list = []
    data = data.copy()
    # target = data['target']
    for column in list(data.columns[45:102]):

        ic = data[column].rolling(rolling).corr(data['target'])
        ic_mean = np.mean(ic)
        print(ic_mean)
        ic_list.append(ic_mean)
        IC = pd.DataFrame(ic_list)
        columns = pd.DataFrame(data.columns[45:102])
        IC_columns = pd.concat([IC, columns], axis=1)
        col = ['value', 'factor']
        IC_columns.columns = col
    return IC_columns
IC_columns = calcpearsonr(data,rolling=15)
#%%
time_1 = '2023-04-01 00:00:00'
time_2 = '2023-04-14 23:59:59'

cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data[data.index < time_1]
test = data[(data.index >= time_1)&(data.index <= time_2)]
print('测试集多空分布:',
      test['target'].apply(lambda x:classify(x)).value_counts())
train['target'] = train['target'].apply(lambda x:classify(x))
train = train[~train['target'].isin([-1])]
train_set = train[train_col]
train_set = train_set.iloc[:,45:102]
# train_set = train_set.iloc[:,45:85] #binance
train_target = train["target"]
test_set = test[train_col]
test_set = test_set.iloc[:,45:102]
# test_set = test_set.iloc[:,45:85]
test_target = test["target"]

X_train = np.array(train_set)
X_train_target = np.array(train_target)
X_test = np.array(test_set)
X_test_target = np.array(test_target)

del train_set, test_set, train_target, test_target

df = test
df['min'] = ((df['closetime']-df['closetime'].shift(bar))/1000)
print(df['min'].describe())
del df
#%%
# train_set = train[train.index < '2022-10-01 00:00:00']
# # train_set = train[(train.index >= '2022-05-15 00:00:00')&(train.index <= '2022-06-15 23:59:59')]
# test_set = train[(train.index >= '2022-10-01 00:00:00')&(train.index <= '2022-10-30 23:59:59')]
# train_target = target[train.index < '2022-10-01 00:00:00']
# # train_target = target[(train.index >= '2022-05-15 00:00:00')&(train.index <= '2022-06-15 23:59:59')]
# test_target = target[(train.index >= '2022-10-01 00:00:00')&(train.index <= '2022-10-30 23:59:59')]
# #%%
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
# train_set_scaled = sc.fit_transform(train_set)# 数据归一
# test_set_scaled = sc.transform(test_set)
# train_target = np.array(train_target)
# test_target = np.array(test_target)
#
# X_train = train_set_scaled
# X_train_target=train_target
# X_test = test_set_scaled
# X_test_target =test_target
#%%
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
over = SMOTE(random_state=2023)
# under = RandomUnderSampler(sampling_strategy=0.5)
X_train, X_train_target = over.fit_resample(X_train, X_train_target)
#%%
def custom_smooth_l1_loss_eval(y_pred, lgb_train):
    """
    Calculate loss value of the custom loss function
     Args:
        y_true : array-like of shape = [n_samples] The target values.
        y_pred : array-like of shape = [n_samples * n_classes] (for multi-class task)
    Returns:
        loss: loss value
        is_higher_better : bool, loss是越低越好，所以这个返回为False
        Is eval result higher better, e.g. AUC is ``is_higher_better``.
    """
    y_true = lgb_train.get_label()
    # y_pred = y_pred.get_label()
    y_pred = y_pred.reshape(len(y_true), len(y_pred) // len(y_true))
    y_pred = np.argmax(y_pred, axis=1)
    residual = (y_true - y_pred).astype("float")
    loss = np.where(np.abs(residual) < 1, (residual ** 2) * 0.5, np.abs(residual) - 0.5)
    return "custom_asymmetric_eval", np.mean(loss), False

def custom_smooth_l1_loss_train(y_pred, lgb_train):
    """Calculate smooth_l1_loss
    Args:
        y_true : array-like of shape = [n_samples]
        The target values. y_pred : array-like of shape = [n_samples * n_classes] (for multi-class task)
    Returns:
        grad: gradient, should be list, numpy 1-D array or pandas Series
        hess: matrix hessian value
    """
    y_true = lgb_train.get_label()
    y_pred = y_pred.reshape(len(y_true), len(y_pred) // len(y_true))
    y_pred = np.argmax(y_pred, axis=1)
    residual = (y_true - y_pred).astype("float")
    grad = np.where(np.abs(residual) < 1, residual, 1)
    hess = np.where(np.abs(residual) < 1, 1.0, 0.0)
    return grad, hess
#%% first model
def ic_lgbm(preds, train_data):
    """Custom IC eval metric for lightgbm"""
    is_higher_better = True
    return 'ic', pearsonr(preds, train_data.get_label())[0], is_higher_better
from sklearn.utils.class_weight import compute_class_weight
def LGB_bayesian(learning_rate, num_leaves, bagging_fraction, feature_fraction, min_child_weight, min_child_samples,
        min_split_gain, min_data_in_leaf, max_depth, reg_alpha, reg_lambda, n_estimators, colsample_bytree, subsample,):
    # LightGBM expects next three parameters need to be integer.
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)
    learning_rate = float(learning_rate)
    subsample = float(subsample)
    colsample_bytree = float(colsample_bytree)
    n_estimators = int(n_estimators)
    min_child_samples = float(min_child_samples)
    min_split_gain = float(min_split_gain)
    # scale_pos_weight = float(scale_pos_weight)
    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
    kf = TimeSeriesSplit(n_splits=5)
    X_train_pred = np.zeros(len(X_train_target))


    for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = X_train_target[train_index], X_train_target[val_index]
        # sample_x = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        # sample_x = [1 if i == 0 else 2 for i in y_train.tolist()]
        # sample_y = compute_class_weight(class_weight='balanced', classes=np.unique(y_val), y=y_val)
        # sample_y = [1 if i == 0 else 2 for i in y_val.tolist()]
        train_set = lgb.Dataset(x_train, label=y_train)
        val_set = lgb.Dataset(x_val, label=y_val)
        params = {
            'colsample_bytree': colsample_bytree,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'min_child_weight': min_child_weight,
            'min_child_samples': min_child_samples,
            'min_split_gain': min_split_gain,
            'bagging_fraction': bagging_fraction,
            'feature_fraction': feature_fraction,
            'subsample': subsample,
            'n_estimators': n_estimators,
            # 'learning_rate' : learning_rate,
            'max_depth': max_depth,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'objective': 'cross_entropy',
            # 'objective': 'multiclass',
            # 'num_class': '3',
            'save_binary': True,
            'seed': 2023,
            'feature_fraction_seed': 2023,
            'bagging_seed': 2023,
            'drop_seed': 2023,
            'data_random_seed': 2023,
            'boosting_type': 'gbdt',
            'verbose': 1,
            # 'is_unbalance': True,
            # 'scale_pos_weight': 2,
            'boost_from_average': True,
            'metric': {'cross_entropy','auc'},
            # 'metric': {'multi_logloss','auc'},
            'num_threads': 20}


        model = lgb.train(params, train_set=train_set, num_boost_round=5000, early_stopping_rounds=50,feval=ic_lgbm,
                          valid_sets=[val_set], verbose_eval=100) #fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)
        X_train_pred += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
        # fpr_train, tpr_train, thresholds_train = roc_auc_score(x_val, y_val)
        # gmeans_train = sqrt(tpr_train * (1 - fpr_train))
        # ix_train = argmax(gmeans_train)
        # print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
        #
        # thresholds_point_train = thresholds_train[ix_train]
        # x_val_thresholds = [1 if y > thresholds_point_train else 0 for y in x_val]
        score = roc_auc_score(X_train_target, X_train_pred)

        return score

bounds_LGB = {
    'colsample_bytree': (0.7, 1),
    'n_estimators': (500, 10000),
    'num_leaves': (31, 500),
    'min_data_in_leaf': (20, 200),
    'bagging_fraction' : (0.1, 0.9),
    'feature_fraction' : (0.1, 0.9),
    'learning_rate': (0.001, 0.3),
    'min_child_weight': (0.00001, 0.01),
    'min_child_samples': (2, 100),
    'min_split_gain': (0.1, 1),
    'subsample': (0.7, 1),
    'reg_alpha': (1, 2),
    'reg_lambda': (1, 2),
    'max_depth': (-1, 50),
    # 'scale_pos_weight':(0.5, 10)
}

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=2023)

init_points = 20
n_iter = 10
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

# LGB_BO.max['target']
# LGB_BO.max['params']

def lightgbm_model(X_train_target, X_test_target, LGB_BO, train):

    kf = TimeSeriesSplit(n_splits=5)
    y_pred = np.zeros(len(X_test_target))
    y_pred_train = np.zeros(len(X_train_target))
    importances = []
    model_list = []
    LGB_BO.max['params'] = LGB_BO.max['params']
    features = train.iloc[:,45:102].columns
    features = list(features)

    def plot_importance(importances, features, PLOT_TOP_N=20, figsize=(10, 10)):
        importance_df = pd.DataFrame(data=importances, columns=features)
        sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
        sorted_importance_df = importance_df.loc[:, sorted_indices]
        plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
        _, ax = plt.subplots(figsize=figsize)
        ax.grid()
        ax.set_xscale('log')
        ax.set_ylabel('Feature')
        ax.set_xlabel('Importance')
        sns.boxplot(data=sorted_importance_df[plot_cols],
                    orient='h',
                    ax=ax)
        plt.show()
    for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
        print('Model:',fold)
        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = X_train_target[train_index], X_train_target[val_index]
        # train_weight = [1 if i == 0 else 2 for i in y_train.tolist()]
        # test_weight = [1 if i == 0 else 2 for i in y_val.tolist()]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)


        params = {
            'boosting_type': 'gbdt',
            # 'metric': 'multi_logloss',
            # 'objective': 'multiclass',
            'metric': {'cross_entropy','auc','average_precision',},
            'objective': 'binary',  # regression,binary,multiclass
            # 'num_class': 3,
            'seed': 2023,
            'feature_fraction_seed': 2023,
            'bagging_seed': 2023,
            'drop_seed': 2023,
            'data_random_seed': 2023,
            'num_leaves': int(LGB_BO.max['params']['num_leaves']),
            'learning_rate': float(LGB_BO.max['params']['learning_rate']),
            'max_depth': int(LGB_BO.max['params']['max_depth']),
            'n_estimators': int(LGB_BO.max['params']['n_estimators']),
            'bagging_fraction': float(LGB_BO.max['params']['bagging_fraction']),
            'feature_fraction': float(LGB_BO.max['params']['feature_fraction']),
            'colsample_bytree': float(LGB_BO.max['params']['colsample_bytree']),
            'subsample': float(LGB_BO.max['params']['subsample']),
            'min_child_samples': int(LGB_BO.max['params']['min_child_samples']),
            'min_child_weight': float(LGB_BO.max['params']['min_child_weight']),
            'min_split_gain': float(LGB_BO.max['params']['min_split_gain']),
            'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
            'reg_alpha': float(LGB_BO.max['params']['reg_alpha']),
            'reg_lambda': float(LGB_BO.max['params']['reg_lambda']),
            # 'max_bin': 63,
            'save_binary': True,
            'verbose': 1,
            # 'is_unbalance': True,
            # 'scale_pos_weight': 2,
            'boost_from_average': True,
            # 'cross_entropy':'xentropy'
            'num_threads': 20
        }

        model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,feval=ic_lgbm,
                          valid_sets=[val_set], verbose_eval=100)#fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)

        y_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
        y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
        importances.append(model.feature_importance(importance_type='gain'))
        model_list.append(model)

        plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
        # lgb.plot_importance(model, max_num_features=30)
        # plt.show()
    return y_pred, y_pred_train, model_list

y_pred, y_pred_train, model_list = lightgbm_model(X_train_target=X_train_target, X_test_target=X_test_target, LGB_BO=LGB_BO, train=train)
#%%
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from numpy import sqrt,argmax
fpr_train, tpr_train, thresholds_train = roc_curve(X_train_target, y_pred_train)
gmeans_train = sqrt(tpr_train * (1-fpr_train))
ix_train = argmax(gmeans_train)
print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
thresholds_point_train = thresholds_train[ix_train]
yhat_train = [1 if y > thresholds_point_train else 0 for y in y_pred_train]
print("训练集表现：")
print(classification_report(yhat_train,X_train_target))
# print(metrics.confusion_matrix(yhat_train, X_train_target))
#%% roccurve
from sklearn.metrics import roc_curve,precision_recall_curve
from numpy import sqrt,argmax
fpr, tpr, thresholds = roc_curve(X_test_target, y_pred)
# fpr, tpr, thresholds = precision_recall_curve(X_test_target, y_pred)
gmeans = sqrt(tpr * (1-fpr))
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
thresholds_point = thresholds[ix]
# y_pred = model.predict(X_test, num_iteration=model.best_iteration)
# thresholds_point = thresholds_train[ix_train]
yhat = [1 if y > thresholds_point else 0 for y in y_pred]
# yhat = [1 if y > 0.55 else 0 for y in y_pred]
print("测试集表现：")
print(classification_report(yhat,X_test_target))
# print(metrics.confusion_matrix(yhat, X_test_target))
print('AUC:', metrics.roc_auc_score(yhat, X_test_target))
#%%
test_data = test
test_data = test_data.reset_index(drop=True)
predict = pd.DataFrame(y_pred,columns=['predict'])
predict['closetime'] = test_data['closetime']
# predict['vwap'] = test_data['vwap']
predict['price'] = test_data['price']
predict['target'] = test_data['target']
# predict['pctrank'] = predict.index.map(lambda x : predict.loc[:x].predict.rank(pct=True)[x])
def pctrank(x):
    n = len(x)
    temp = x.argsort()
    ranks = np.empty(n)
    ranks[temp] = (np.arange(n) + 1) / n
    return ranks[-1]
# predict['pctrank'] = predict['predict'].rolling(bar).apply(pctrank)
#
# df_1 = predict.loc[predict['pctrank']>0.9]
# df_0 = predict.loc[predict['pctrank']<0.1]
# print(len(df_1))
# print(len(df_0))

df_1 = predict.loc[predict['predict']>np.percentile(y_pred_train[-15000:], 90)]
df_0 = predict.loc[predict['predict']<np.percentile(y_pred_train[-15000:], 10)]
print(len(df_1))
print(len(df_0))
# print(np.percentile(y_pred_train[-15000:], 90))
# print(np.percentile(y_pred_train[-15000:], 10))
#
df_1['side'] = 'buy'
df_0['side'] = 'sell'
df = pd.concat([df_1, df_0], axis=0)
df = df.sort_values(by='closetime', ascending=True)
df = df.reset_index(drop=True)
print(df.loc[:,['target','predict']].corr())
# print(stats.jarque_bera(final_df['predict']))
#%%
signal = test.reset_index()
signal['predict'] = predict['predict']
signal_1 = signal[signal['predict']>=np.percentile(y_pred_train[-15000:], 90)]
signal_0 = signal[signal['predict']<=np.percentile(y_pred_train[-15000:], 10)]
# signal['pctrank'] = predict['pctrank']
# signal_1 = signal[signal['pctrank']>0.9]
# signal_0 = signal[signal['pctrank']<0.1]
signal_1['side'] = 'buy'
signal_0['side'] = 'sell'
signal_df = pd.concat([signal_1, signal_0],axis=0)
signal_df = signal_df.sort_values(by='closetime', ascending=True)
signal_df = signal_df.set_index('datetime')
#
def abs_classify(y):

    if y > 0.001:
        return 1
    else:
        return 0

train = data[data.index < time_1]
train_set = train[train_col]
train_set = train_set.iloc[:,45:102]
# train_set = train_set.iloc[:,45:85] #binance
train_target = abs(train['target']).apply(lambda x:abs_classify(x))
test_set = signal_df[train_col]
test_set = test_set.iloc[:,45:102]
# test_set = test_set.iloc[:,45:85]
test_target = abs(signal_df["target"]).apply(lambda x:abs_classify(x))
X_train = np.array(train_set)
X_train_target = np.array(train_target)
X_test = np.array(test_set)
X_test_target = np.array(test_target)

# secondary model
secondary_LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=2023)

init_points = 20
n_iter = 10
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    secondary_LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

def secondary_lightgbm_model(X_train_target, X_test_target, LGB_BO):

    kf = TimeSeriesSplit(n_splits=5)
    y_pred = np.zeros(len(X_test_target))
    y_pred_train = np.zeros(len(X_train_target))
    importances = []
    model_list = []
    LGB_BO.max['params'] = secondary_LGB_BO.max['params']

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
        print('Model:',fold)
        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = X_train_target[train_index], X_train_target[val_index]
        # train_weight = [1 if i == 0 else 2 for i in y_train.tolist()]
        # test_weight = [1 if i == 0 else 2 for i in y_val.tolist()]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)


        params = {
            'boosting_type': 'gbdt',
            # 'metric': 'multi_logloss',
            # 'objective': 'multiclass',
            'metric': {'cross_entropy','auc','average_precision',},
            'objective': 'binary',  # regression,binary,multiclass
            # 'num_class': 3,
            'seed': 2023,
            'feature_fraction_seed': 2023,
            'bagging_seed': 2023,
            'drop_seed': 2023,
            'data_random_seed': 2023,
            'num_leaves': int(LGB_BO.max['params']['num_leaves']),
            'learning_rate': float(LGB_BO.max['params']['learning_rate']),
            'max_depth': int(LGB_BO.max['params']['max_depth']),
            'n_estimators': int(LGB_BO.max['params']['n_estimators']),
            'bagging_fraction': float(LGB_BO.max['params']['bagging_fraction']),
            'feature_fraction': float(LGB_BO.max['params']['feature_fraction']),
            'colsample_bytree': float(LGB_BO.max['params']['colsample_bytree']),
            'subsample': float(LGB_BO.max['params']['subsample']),
            'min_child_samples': int(LGB_BO.max['params']['min_child_samples']),
            'min_child_weight': float(LGB_BO.max['params']['min_child_weight']),
            'min_split_gain': float(LGB_BO.max['params']['min_split_gain']),
            'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
            'reg_alpha': float(LGB_BO.max['params']['reg_alpha']),
            'reg_lambda': float(LGB_BO.max['params']['reg_lambda']),
            # 'max_bin': 63,
            'save_binary': True,
            'verbose': 1,
            # 'is_unbalance': True,
            # 'scale_pos_weight': 2,
            'boost_from_average': True,
            # 'cross_entropy':'xentropy'
            'num_threads': 20
        }

        model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,
                          valid_sets=[val_set], verbose_eval=100)#fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)

        y_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
        y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
        # importances.append(model.feature_importance(importance_type='gain'))
        model_list.append(model)

        # plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
        # lgb.plot_importance(model, max_num_features=20)
        # plt.show()
    return y_pred, y_pred_train, model_list

secondary_y_pred, secondary_y_pred_train, secondary_model_list = secondary_lightgbm_model(X_train_target=X_train_target, X_test_target=X_test_target, LGB_BO=secondary_LGB_BO)
#%% two model saving
def model_saveing(model_list,secondary_model_list,base_path, symbol):

    joblib.dump(model_list[0],'{}/{}_lightGBM_side_0.pkl'.format(base_path, symbol))
    joblib.dump(model_list[1],'{}/{}_lightGBM_side_1.pkl'.format(base_path,symbol))
    joblib.dump(model_list[2],'{}/{}_lightGBM_side_2.pkl'.format(base_path,symbol))
    joblib.dump(model_list[3],'{}/{}_lightGBM_side_3.pkl'.format(base_path,symbol))
    joblib.dump(model_list[4],'{}/{}_lightGBM_side_4.pkl'.format(base_path,symbol))
    joblib.dump(secondary_model_list[0],'{}/{}_lightGBM_out_0.pkl'.format(base_path,symbol))
    joblib.dump(secondary_model_list[1],'{}/{}_lightGBM_out_1.pkl'.format(base_path,symbol))
    joblib.dump(secondary_model_list[2],'{}/{}_lightGBM_out_2.pkl'.format(base_path,symbol))
    joblib.dump(secondary_model_list[3],'{}/{}_lightGBM_out_3.pkl'.format(base_path,symbol))
    joblib.dump(secondary_model_list[4],'{}/{}_lightGBM_out_4.pkl'.format(base_path,symbol))

    return
base_path = '/songhe/model_save/'
# base_path = '/songhe/solusdt/'
model_saveing(model_list, secondary_model_list, base_path, symbol)
#%%
def seondary_model_train_test(X_train_target, secondary_y_pred_train, X_test_target, secondary_y_pred):

    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
    from numpy import sqrt, argmax
    fpr_train, tpr_train, thresholds_train = roc_curve(X_train_target, secondary_y_pred_train)
    gmeans_train = sqrt(tpr_train * (1-fpr_train))
    ix_train = argmax(gmeans_train)
    # print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
    thresholds_point_train = thresholds_train[ix_train]
    secondary_yhat_train = [1 if y > thresholds_point_train else 0 for y in secondary_y_pred_train]
    # print("secondary_model训练集表现：")
    # print(classification_report(yhat_train,X_train_target))

    fpr, tpr, thresholds = roc_curve(X_test_target, secondary_y_pred)
    # fpr, tpr, thresholds = precision_recall_curve(X_test_target, y_pred)
    gmeans = sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # thresholds_point = thresholds_train[ix_train]
    secondary_yhat = [1 if y > thresholds[ix] else 0 for y in secondary_y_pred]
    # yhat = [1 if y > 0.55 else 0 for y in y_pred]
    # print("secondary_model测试集表现：")
    # print(classification_report(secondary_yhat,X_test_target))
    # print(metrics.confusion_matrix(yhat, X_test_target))
    # print('AUC:', metrics.roc_auc_score(secondary_yhat, X_test_target))
    return secondary_yhat_train, secondary_yhat
secondary_yhat_train, secondary_yhat = seondary_model_train_test(X_train_target, secondary_y_pred_train, X_test_target, secondary_y_pred)
print("secondary_model训练集表现：")
print(classification_report(secondary_yhat_train,X_train_target))
print("secondary_model测试集表现：")
print(classification_report(secondary_yhat,X_test_target))
#%%
out_threshold = 70
secondary_predict = pd.DataFrame(secondary_y_pred, columns=['out'])
# secondary_predict['out_pctrank'] = secondary_predict.index.map(lambda x : secondary_predict.loc[:x].out.rank(pct=True)[x])
# secondary_predict['out_pctrank'] = secondary_predict['out'].rolling(bar).apply(pctrank)
signal_df_ = signal_df.reset_index()
signal_df_['out'] = secondary_predict['out']
# signal_df_['out_pctrank'] = secondary_predict['out_pctrank']
final_df = signal_df_[signal_df_['out']>=np.percentile(secondary_y_pred_train[-15000:], out_threshold)]
# final_df = signal_df_[signal_df_['out']>=0.852]
print(len(final_df))
final_df = final_df.sort_values(by='closetime', ascending=True)
# final_df = final_df.loc[:,['datetime','closetime','vwap','price','predict','target','side','pctrank','out_pctrank']]
final_df = final_df.loc[:,['datetime','closetime','wap1','price','predict','target','side']]
print(final_df.loc[:,['target','predict']].corr())
final_df['symbol'] = symbol
final_df['platform'] = 'binance_swap_u'
# final_df['starttime'] = final_df['closetime']
# final_df['endtime'] = final_df['starttime']+60000*60*22
#%%
# final_df = final_df.dropna(axis=0)
final_df['datetime'] = pd.to_datetime(final_df['closetime']+28800000, unit='ms')
final_df = final_df.set_index('datetime')
# final_df = final_df[final_df.index>='2022-11-12 21:00:00']
final_df.to_csv('/songhe/{}/{}_20220401_0430_{}bar_vwap_ST2.0_20230531_filter_{}.csv'.format(symbol, symbol, bar, out_threshold))
#%%
df1 = pd.read_csv('/songhe/{}/{}_20221001_1030_{}bar_vwap_ST2.0_20230531_filter_{}.csv'.format(symbol, symbol, bar, out_threshold))
df2 = pd.read_csv('/songhe/{}/{}_20221101_1130_{}bar_vwap_ST2.0_20230531_filter_{}.csv'.format(symbol, symbol, bar, out_threshold))
df3 = pd.read_csv('/songhe/{}/{}_20221201_1230_{}bar_vwap_ST2.0_20230531_filter_{}.csv'.format(symbol, symbol, bar, out_threshold))
df4 = pd.read_csv('/songhe/{}/{}_20230101_0130_{}bar_vwap_ST2.0_20230531_filter_{}.csv'.format(symbol, symbol, bar, out_threshold))
df5 = pd.read_csv('/songhe/{}/{}_20230201_0228_{}bar_vwap_ST2.0_20230531_filter_{}.csv'.format(symbol, symbol, bar, out_threshold))
df6 = pd.read_csv('/songhe/{}/{}_20230301_0330_{}bar_vwap_ST2.0_20230531_filter_{}.csv'.format(symbol, symbol, bar, out_threshold))
df7 = pd.read_csv('/songhe/{}/{}_20230401_0430_{}bar_vwap_ST2.0_20230531_filter_{}.csv'.format(symbol, symbol, bar, out_threshold))

final_df = pd.concat([df1,df2,df3,df4,df5,df6,df7], axis=0)
final_df = final_df.sort_values(by='closetime', ascending=True)
final_df['datetime'] = pd.to_datetime(final_df['closetime']+28800000, unit='ms')
final_df = final_df.set_index('datetime')
# final_df = final_df[final_df.index>='2022-11-12 21:00:00']
del df1,df2,df3,df4,df5,df6,df7
final_df.to_csv('/songhe/{}/{}_20221001_0430_{}bar_vwap_ST2.0_20230531_filter_{}.csv'.format(symbol, symbol, bar,out_threshold))
#%% 精细回测
# symbol = 'ethusdt'
# bar = 40
# out_threshold = 90
final_df = pd.read_csv('/songhe/{}/{}_20221001_0430_{}bar_vwap_ST2.0_20230531_filter_{}.csv'.format(symbol, symbol, bar, out_threshold))
final_df['datetime'] = pd.to_datetime(final_df['closetime']+28800000, unit='ms')
#
final_df['time'] = None
from datetime import datetime
for i in range(len(final_df['datetime'])):
    final_df['time'].iloc[i] = ((final_df['datetime'].iloc[i].replace(second=59, microsecond=999999).timestamp()-28800)*1000)
    if final_df['time'].iloc[i] % 100 == 0:
        final_df['time'].iloc[i] = (final_df['time'].iloc[i])-1
final_df['closetime'] = final_df['time'].astype('int')
#
final_df = final_df.iloc[:,1:]
final_df['type'] = 'cur'
final_df['year'] = 2023
final_df['month'] = 5
# final_df = final_df.drop_duplicates(subset=['closetime'], keep='last')

# final_df = final_df.reset_index(drop=True)
from pyarrow import Table
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
dir_name = 'datafile/eval/songhe/{}'.format("cta_binance_{}bar_vwap_20221001_0430_20230531_{}_{}".format(bar, symbol, out_threshold))
pq.write_to_dataset(Table.from_pandas(df=final_df),
                    root_path=dir_name,
                    filesystem=minio, basename_template="part-{i}.parquet",
                    existing_data_behavior="overwrite_or_ignore")
#%%
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the time series data
ax.plot(T, X, label='Time series data')

# Plot the labeled trend regions
trend_up = T[trend_labels == 1]
trend_down = T[trend_labels == -1]
if len(trend_up) > 0:
    ax.axvspan(trend_up[0], trend_up[-1], color='green', alpha=0.2, label='Up trend')
if len(trend_down) > 0:
    ax.axvspan(trend_down[0], trend_down[-1], color='red', alpha=0.2, label='Down trend')

# Add labels and legend
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.set_title('Labeled Trend Regions')
ax.legend()

plt.show()

