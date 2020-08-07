from functionUtils import *
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook, tnrange
from workalendar.america import Brazil
from datetime import date, timedelta
import pandas as pd
import numpy as np
import warnings
import workalendar
import datetime
import gc

DATA_PATH = '../../input/'

print('read data...')
df_train = pd.read_csv(DATA_PATH + 'train.csv')
df_test = pd.read_csv(DATA_PATH + 'test.csv')
df_historical = pd.read_csv(DATA_PATH + 'historical_transactions.csv', dtype={'purchase_date': np.str}, low_memory=True)
df_new_merchant = pd.read_csv(DATA_PATH + 'new_merchant_transactions.csv', dtype={'purchase_date': np.str},
                              low_memory=True)

df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)
df_historical = reduce_mem_usage(df_historical)
df_new_merchant = reduce_mem_usage(df_new_merchant)

df_train['is_test'] = 0
df_test['is_test'] = 1
# 用户在历史和之后的记录中是否出现过
for df in [df_train, df_test]:
    df['is_in_new_monthlag1'] = df['card_id'].isin(df_new_merchant[df_new_merchant.month_lag == 1].card_id).astype(int)
    df['is_in_new_monthlag2'] = df['card_id'].isin(df_new_merchant[df_new_merchant.month_lag == 2].card_id).astype(int)
df_data = pd.concat([df_train, df_test])
df_data = pd.get_dummies(df_data, columns=['feature_1', 'feature_2', 'feature_3'],
                         prefix=['feature_1', 'feature_2', 'feature_3'])
df_transactions = pd.concat([df_historical, df_new_merchant])

df_transactions['purchase_amount'] = np.round(df_transactions['purchase_amount'] / 0.00150265118 + 497.06, 2)

## id类特征
print('id类特征')
df_temp = df_historical.card_id.value_counts().reset_index()
df_temp.rename(columns={'index': 'card_id', 'card_id': 'hist_card_id_counts'}, inplace=True)
df_temp['hist_card_id_ratio'] = df_temp['hist_card_id_counts'] / df_historical.shape[0]
df_data = df_data.merge(df_temp, on='card_id', how='left')

df_temp = df_new_merchant.card_id.value_counts().reset_index()
df_temp.rename(columns={'index': 'card_id', 'card_id': 'new_card_id_counts'}, inplace=True)
df_temp['new_card_id_ratio'] = df_temp['new_card_id_counts'] / df_new_merchant.shape[0]
df_data = df_data.merge(df_temp, on='card_id', how='left')

df_data['hist/new_card_counts_ratio'] = df_data['hist_card_id_counts'] / df_data['new_card_id_counts']

df_data[['hist_card_id_counts', 'hist_card_id_ratio', 'new_card_id_counts', 'new_card_id_ratio',
         'hist/new_card_counts_ratio']].fillna(0, inplace=True)

del df_temp
gc.collect()

dateHandle = pd.to_datetime(df_data['first_active_month'])
df_data['active_year'] = dateHandle.dt.year
df_data['active_month'] = dateHandle.dt.month
df_data['active_to_base_time'] = (datetime.date(2018, 2, 28) - dateHandle.dt.date).dt.days

print('刷卡时间...')
# hist第一次刷卡时间
df_temp = df_historical.groupby('card_id')['purchase_date'].min().reset_index()
df_temp.rename(columns={'purchase_date': 'hist_first_purchase_date'}, inplace=True)
df_data = df_data.merge(df_temp, on='card_id', how='left')
# 最后一次刷卡时间
df_temp = df_historical.groupby('card_id')['purchase_date'].max().reset_index()
df_temp.rename(columns={'purchase_date': 'hist_last_purchase_date'}, inplace=True)
df_data = df_data.merge(df_temp, on='card_id', how='left')

# new第一次刷卡时间
df_temp = df_new_merchant.groupby('card_id')['purchase_date'].min().reset_index()
df_temp.rename(columns={'purchase_date': 'new_first_purchase_date'}, inplace=True)
df_data = df_data.merge(df_temp, on='card_id', how='left')
# 最后一次刷卡时间
df_temp = df_new_merchant.groupby('card_id')['purchase_date'].max().reset_index()
df_temp.rename(columns={'purchase_date': 'new_last_purchase_date'}, inplace=True)
df_data = df_data.merge(df_temp, on='card_id', how='left')

# hist-new 购买时长
df_data['hist_first_to_base_time'] = (
        pd.to_datetime(df_data['hist_first_purchase_date']).dt.date - datetime.date(2018, 2, 28)).dt.days
df_data['hist_last_to_base_time'] = (
        pd.to_datetime(df_data['hist_last_purchase_date']).dt.date - datetime.date(2018, 2, 28)).dt.days
df_data['hist_last_to_first_time'] = (pd.to_datetime(df_data['hist_last_purchase_date']).dt.date - pd.to_datetime(
    df_data['hist_first_purchase_date']).dt.date).dt.days
df_data['hist_first_active_time'] = (pd.to_datetime(df_data['hist_first_purchase_date']).dt.date - pd.to_datetime(
    df_data['first_active_month']).dt.date).dt.days
df_data['hist_last_active_time'] = (pd.to_datetime(df_data['hist_last_purchase_date']).dt.date - pd.to_datetime(
    df_data['first_active_month']).dt.date).dt.days

df_data['new_first_to_base_time'] = (
        pd.to_datetime(df_data['new_first_purchase_date']).dt.date - datetime.date(2018, 2, 28)).dt.days
df_data['new_last_to_base_time'] = (
        pd.to_datetime(df_data['new_last_purchase_date']).dt.date - datetime.date(2018, 2, 28)).dt.days
df_data['new_first_to_active_time'] = (pd.to_datetime(df_data['new_first_purchase_date']).dt.date - pd.to_datetime(
    df_data['first_active_month']).dt.date).dt.days
df_data['new_last_to_active_time'] = (pd.to_datetime(df_data['new_last_purchase_date']).dt.date - pd.to_datetime(
    df_data['first_active_month']).dt.date).dt.days
df_data['new_last_to_first_time'] = (pd.to_datetime(df_data['new_last_purchase_date']).dt.date - pd.to_datetime(
    df_data['new_first_purchase_date']).dt.date).dt.days

# hist最后一次购买距离new第一次购买的时间差
df_data['new_to_hist_time'] = (pd.to_datetime(df_data['new_first_purchase_date']).dt.date -
                               pd.to_datetime(df_data['hist_last_purchase_date']).dt.date).dt.days
# 单位时间内的刷卡次数
df_data['hist_per_time_purchaseCounts'] = df_data['hist_card_id_counts'] / df_data['hist_last_to_first_time']
df_data['new_per_time_purchaseCounts'] = df_data['new_card_id_counts'] / df_data['new_last_to_first_time']

df_data.drop(columns=['hist_last_purchase_date', 'hist_first_purchase_date', 'new_first_purchase_date',
                      'new_last_purchase_date'], inplace=True)
df_data.replace([np.inf, -np.inf], 0, inplace=True)
df_data.fillna(-1, inplace=True)
del df_train, df_test, df_historical, df_new_merchant, df_temp
gc.collect()


def dateUtils(df=None, timeCol='purchase_date'):
    dateHandle = pd.to_datetime(df[timeCol])
    df['week'] = dateHandle.dt.week
    df['year'] = dateHandle.dt.year
    df['month'] = dateHandle.dt.month
    df['dayofweek'] = dateHandle.dt.dayofweek
    df['weekend'] = (dateHandle.dt.weekday >= 5).astype(int)
    df['hour'] = dateHandle.dt.hour
    df['month_gap'] = (dateHandle.dt.date - datetime.date(2018, 2, 28)).dt.days // 30
    df['day_gap'] = (dateHandle.dt.date - datetime.date(2018, 2, 28)).dt.days
    # cardid用户连续购买之间的时间差
    df['purchase_time'] = dateHandle
    df['purchase_time_diff_days'] = df.groupby('card_id')['purchase_time'].apply(lambda series: series.diff(1)).dt.days
    df['purchase_time_diff_days'].fillna(df.purchase_time_diff_days.mean(), inplace=True)
    df['purchase_time_diff_seconds'] = df.groupby('card_id')['purchase_time'].apply(
        lambda series: series.diff(1)).dt.seconds
    df['purchase_time_diff_seconds'].fillna(df.purchase_time_diff_seconds.mean(), inplace=True)
    df.drop(columns=['purchase_time'], inplace=True)

    def getdate2Holiday(d):
        cal = Brazil()
        holiday = cal.holidays(d.year)
        dis, flag = 0, 0
        year, month, day = d.year, d.month, d.day
        d = datetime.date(year, month, day)
        for i, h in enumerate(holiday):
            if holiday[i][0] > d:
                flag = 1
                dis = (holiday[i][0] - d).days
                return dis
        if flag == 0:
            dis = (holiday[-1][0] - d).days
        return dis

    df['date2Holiday'] = df['purchase_date'].apply(lambda x: getdate2Holiday(pd.to_datetime(x)))

    return df


# 缺失值处理
# C2:区域；C3分期等级；C1为-1城市编码对应Y
df_transactions['category_2'].fillna(6, inplace=True)
df_transactions['category_3'].fillna('B', inplace=True)
df_transactions['merchant_id'].fillna(df_transactions['merchant_id'].value_counts().index[0], inplace=True)
df_transactions['installments'].replace(-1, 1, inplace=True)
df_transactions['installments'].replace(999, 0, inplace=True)

# 交互特征
df_transactions['city_C1'] = ((df_transactions['city_id'] < 0) + 0).astype(np.str) + '_' + df_transactions[
    'category_1'].astype(np.str)
df_transactions['C2_state'] = df_transactions['category_2'].astype(np.str) + '_' + df_transactions['state_id'].astype(
    np.str)
df_transactions['C2_state_subsector'] = df_transactions['category_2'].astype(np.str) + '_' + df_transactions[
    'state_id'].astype(np.str) + '_' + df_transactions['subsector_id'].astype(np.str)
df_transactions['subsector_city'] = df_transactions['subsector_id'].astype(np.str) + '_' + (
        (df_transactions['city_id'] < 0) + 0).astype(np.str)
df_transactions['auth_C3'] = df_transactions['authorized_flag'].astype(np.str) + '_' + df_transactions[
    'category_3'].astype(np.str)

cateCols = ['city_C1', 'C2_state', 'C2_state_subsector', 'subsector_city', 'auth_C3']
print('label_encoding...')
df_transactions = label_encoding(df_transactions, cateCols)

print('category_3...')
for cate in ['A', 'B', 'C']:
    df_transactions['category_3_%s' % cate] = (df_transactions['category_3'] == cate) + 0
print('category_2...')
for cate in [1, 2, 3, 4, 5, 6]:
    df_transactions['category_2_%s' % cate] = (df_transactions['category_2'] == cate) + 0
print('authorized_flag...')
for cate in ['Y', 'N']:
    df_transactions['authorized_flag_%s' % cate] = (df_transactions['authorized_flag'] == cate) + 0

print('map...')
df_transactions['category_1'] = df_transactions['category_1'].map({'Y': 1, 'N': 0})
df_transactions['category_3'] = df_transactions['category_3'].map({'A': 0, 'B': 1, 'C': 2})
df_transactions['authorized_flag'] = df_transactions['authorized_flag'].map({'Y': 1, 'N': 0})

df_transactions.sort_values(by=['card_id', 'purchase_date'], ascending=True, inplace=True)
print('date utils...')
df_transactions = dateUtils(df_transactions, timeCol='purchase_date')
df_transactions = reduce_mem_usage(df_transactions)

df_transactions.sort_values(by=['card_id', 'purchase_date'], ascending=True, inplace=True)
# df_transactions['purchase_diff'] = df_transactions.groupby('card_id')['purchase_amount'].apply(lambda series:series.diff(1)).values
# df_transactions['purchase_diff'].fillna(0,inplace=True)
df_hist_transactions = df_transactions[df_transactions.month_lag <= 0]
df_new_transactions = df_transactions[df_transactions.month_lag > 0]
for df in [df_hist_transactions, df_new_transactions]:
    df['purchase_diff'] = df.groupby('card_id')['purchase_amount'].apply(lambda series: series.diff(1)).values
    df['purchase_diff'].fillna(0, inplace=True)
df_hist_auth_Y_transactions = df_hist_transactions[df_hist_transactions.authorized_flag == 1]
df_hist_auth_N_transactions = df_hist_transactions[df_hist_transactions.authorized_flag == 0]


# 用户月消费记录
def getMonthPurchase(df_data, df_feature, group='month_gap', fea='purchase_amount', name='hist'):
    df_purchase = df_feature.groupby(['card_id', 'month_gap'])['purchase_amount'].sum().reset_index()
    df_purchase.rename(columns={'purchase_amount': 'purchase_amount_sum_month'}, inplace=True)
    df_purchase.sort_values(by=['month_gap'], inplace=True, ascending=True)

    df_temp = df_purchase[['card_id', 'month_gap', 'purchase_amount_sum_month']]
    df_temp.index = df_temp.card_id
    df_temp = df_temp.set_index(['month_gap'], append=True)
    df_temp = pd.Series(df_temp['purchase_amount_sum_month'].values.reshape(len(df_temp['purchase_amount_sum_month'])),
                        index=df_temp.index)
    df_temp = df_temp.unstack()
    df_temp.reset_index(inplace=True)
    cols = ['card_id']
    for index in list(df_purchase['month_gap'].unique()):
        cols.append('%s_month%s_purchase' % (name, index))
    df_temp.columns = cols
    df_temp.fillna(0, inplace=True)
    df_data = df_data.merge(df_temp, on='card_id', how='left')

    del df_purchase, df_temp, cols
    gc.collect()

    return df_data


# 周消费记录
def getWeekPurchase(df_data, df_feature, group='week', fea='purchase_amount', name='hist'):
    df_purchase = df_feature.groupby(['card_id', 'week'])['purchase_amount'].sum().reset_index()
    df_purchase.rename(columns={'purchase_amount': 'purchase_sum_week'}, inplace=True)
    df_purchase.sort_values(by=['week'], inplace=True, ascending=True)
    df_count = df_purchase.groupby(['card_id'])['purchase_sum_week'].count().reset_index()
    df_count.rename(columns={'purchase_sum_week': '%s_purchase_counts' % name}, inplace=True)

    df_temp = df_purchase[['card_id', 'week', 'purchase_sum_week']]
    df_temp.index = df_temp.card_id
    df_temp = df_temp.set_index(['week'], append=True)
    df_temp = pd.Series(df_temp['purchase_sum_week'].values.reshape(len(df_temp['purchase_sum_week'])),
                        index=df_temp.index)
    df_temp = df_temp.unstack()
    df_temp.reset_index(inplace=True)
    cols = ['card_id']
    for index in list(df_purchase['week'].unique()):
        cols.append('%s_week%s_purchase' % (name, index))
    df_temp.columns = cols
    df_temp = df_temp.merge(df_count, on='card_id', how='left')
    df_temp.fillna(0, inplace=True)
    df_data = df_data.merge(df_temp, on='card_id', how='left')
    return df_data


# 用户月分期记录
def getMonthInstallments(df_data, df_feature, group='month_gap', fea='installments', name='hist'):
    df_purchase = df_feature.groupby(['card_id', 'month_gap'])['installments'].sum().reset_index()
    df_purchase.rename(columns={'installments': 'installments_sum_month'}, inplace=True)
    df_purchase.sort_values(by=['month_gap'], inplace=True, ascending=True)

    df_temp = df_purchase[['card_id', 'month_gap', 'installments_sum_month']]
    df_temp.index = df_temp.card_id
    df_temp = df_temp.set_index(['month_gap'], append=True)
    df_temp = pd.Series(df_temp['installments_sum_month'].values.reshape(len(df_temp['installments_sum_month'])),
                        index=df_temp.index)
    df_temp = df_temp.unstack()
    df_temp.reset_index(inplace=True)
    cols = ['card_id']
    for index in list(df_purchase['month_gap'].unique()):
        cols.append('%s_month%s_installments' % (name, index))
    df_temp.columns = cols
    df_temp.fillna(0, inplace=True)
    df_data = df_data.merge(df_temp, on='card_id', how='left')
    del df_purchase, df_temp, cols
    gc.collect()
    return df_data


# 用户周分期记录
def getWeekInstallments(df_data, df_feature, group='week', fea='installments', name='hist'):
    df_purchase = df_feature.groupby(['card_id', 'week'])['installments'].sum().reset_index()
    df_purchase.rename(columns={'installments': 'installments_sum_week'}, inplace=True)
    df_purchase.sort_values(by=['week'], inplace=True, ascending=True)

    df_temp = df_purchase[['card_id', 'week', 'installments_sum_week']]
    df_temp.index = df_temp.card_id
    df_temp = df_temp.set_index(['week'], append=True)
    df_temp = pd.Series(df_temp['installments_sum_week'].values.reshape(len(df_temp['installments_sum_week'])),
                        index=df_temp.index)
    df_temp = df_temp.unstack()
    df_temp.reset_index(inplace=True)
    cols = ['card_id']
    for index in list(df_purchase['week'].unique()):
        cols.append('%s_week%s_installments' % (name, index))
    df_temp.columns = cols
    df_temp.fillna(0, inplace=True)
    df_data = df_data.merge(df_temp, on='card_id', how='left')
    del df_purchase, df_temp, cols
    gc.collect()
    return df_data


df_data = getMonthPurchase(df_data, df_hist_auth_Y_transactions, name='hist_auth_Y')
df_data = getMonthPurchase(df_data, df_hist_auth_N_transactions, name='hist_auth_N')
df_data = getMonthPurchase(df_data, df_new_transactions, name='new')
# df_data = getMonthInstallments(df_data,df_hist_transactions,name='hist')
# df_data = getMonthInstallments(df_data,df_new_transactions,name='new')


# df_data = getWeekPurchase(df_data,df_hist_auth_Y_transactions,name='hist_auth_Y')
# df_data = getWeekPurchase(df_data,df_hist_auth_N_transactions,name='hist_auth_N')
# df_data = getWeekPurchase(df_data,df_new_transactions,name='new')
# df_data = getWeekInstallments(df_data,df_hist_transactions,name='hist')
# df_data = getWeekInstallments(df_data,df_new_transactions,name='new')

df_data.replace([np.inf, -np.inf], np.nan, inplace=True)
df_data.fillna(0, inplace=True)

### tmp01 to_csv
# df_data.to_csv('tmp01_df_data.csv',index=False)
# df_transactions.to_csv('tmp01_df_transactions.csv',index=False)

originalCols = list(df_data.columns)


###持卡人不同授权类型的购买总量
def getAuthorizedPurchase(df_data, df_features, name='hist'):
    df_temp = df_features.groupby(['card_id', 'authorized_flag'])['purchase_amount'].sum().reset_index()
    df_temp = df_temp.pivot(index='card_id', columns='authorized_flag', values='purchase_amount')
    cols = []
    for col in df_temp.columns:
        cols.append(name + df_temp.columns.name + '_' + np.str(col) + '_purchasesum')
    df_temp.columns = cols
    df_temp.columns.name = None
    df_temp.reset_index(inplace=True)
    df_temp.fillna(0, inplace=True)
    if name == 'hist':
        df_temp['%s_Auth_purchase_sum' % name] = df_temp[cols[0]] + df_temp[cols[1]]
        df_temp[cols[0] + '_ratio'] = df_temp[cols[0]] / df_temp['%s_Auth_purchase_sum' % name]
        df_temp[cols[1] + '_ratio'] = df_temp[cols[1]] / df_temp['%s_Auth_purchase_sum' % name]
        df_temp.drop(columns=['%s_Auth_purchase_sum' % name], inplace=True)
    df_data = df_data.merge(df_temp, on='card_id', how='left')
    del df_temp
    gc.collect()
    return df_data


#### 用户对商家的回购率
def getUserRepurchaseRatio(df_data, df_fea, group=['card_id', 'merchant_id'], fea='purchase_amount',
                           name='repurchase_mechant_ratio'):
    df_temp = df_features.groupby(group)[fea].count().reset_index().rename(
        columns={fea: 'purchase_amount_counts'})
    df_temp['repurchase_flag'] = (df_temp['purchase_amount_counts'] > 1) + 0
    df = df_temp.groupby(['card_id'])['purchase_amount_counts'].sum().reset_index().rename(
        columns={'purchase_amount_counts': 'purchase_records'})
    df_temp['repurchase_counts'] = df_temp['repurchase_flag'] * df_temp['purchase_amount_counts']
    df_temp = df_temp.merge(df, on='card_id', how='left')
    df_temp['repurchase_ratio'] = df_temp['repurchase_counts'] / df_temp['purchase_records']
    df_temp = df_temp.groupby('card_id')['repurchase_ratio'].max().reset_index().rename(
        columns={'repurchase_ratio': name})
    df_data = df_data.merge(df_temp, on='card_id', how='left')
    del df, df_temp
    gc.collect()
    return df_data


for flag, df_features in zip(['hist_auth_Y', 'hist_auth_N', 'new'],
                             [df_hist_auth_Y_transactions, df_hist_auth_N_transactions, df_new_transactions]):
    print(".................%s......................" % flag)
    if 'hist' in flag:
        month_lag = [-1, -3, -5, -7, -13]
    else:
        month_lag = [1, 2]
    for index in month_lag:
        print('month_lag:', index)
        df = df_features[df_features.month_lag >= index]
        # 各个阶段的purchase_amount(强特)
        print('purchase_amount...')
        df_data = getMeanStaticsFeatures(df_data, df, ['card_id'], 'purchase_amount',
                                         name='%s_purchaseAmountMean_%s' % (flag, index))
        df_data = getMaxStaticsFeatures(df_data, df, ['card_id'], 'purchase_amount',
                                        name='%s_purchaseAmountMax_%s' % (flag, index))
        df_data = getMedianStaticsFeatures(df_data, df, ['card_id'], 'purchase_amount',
                                           name='%s_purchaseAmountMedian_%s' % (flag, index))
        df_data = getSumStaticsFeatures(df_data, df, ['card_id'], 'purchase_amount',
                                        name='%s_purchaseAmountSum_%s' % (flag, index))
        df_data = getStdStaticsFeatures(df_data, df, ['card_id'], 'purchase_amount',
                                        name='%s_purcahseAmountStd_%s' % (flag, index))
        df_data = getCountsStaticsFeatures(df_data, df, ['card_id'], 'purchase_amount',
                                           name='%s_purcahseAmountCount_%s' % (flag, index))

        df_data['%s_per_time_purchaseAmountSum_%s' % (flag, index)] = df_data[
                                                                          '%s_purchaseAmountSum_%s' % (flag, index)] / (
                                                                          np.abs(index))
        df_data['%s_per_time_purcahseAmountCount_%s' % (flag, index)] = df_data['%s_purcahseAmountCount_%s' % (
            flag, index)] / (np.abs(index))

        # 分期数
        print('installments...')
        df_data = getMeanStaticsFeatures(df_data, df, ['card_id'], 'installments',
                                         name='%s_installmentsMean_%s' % (flag, index))
        df_data = getMaxStaticsFeatures(df_data, df, ['card_id'], 'installments',
                                        name='%s_installmentsMax_%s' % (flag, index))
        df_data = getSumStaticsFeatures(df_data, df, ['card_id'], 'installments',
                                        name='%s_installmentsSum_%s' % (flag, index))
        df_data = getCountsStaticsFeatures(df_data, df, ['card_id'], 'installments',
                                           name='%s_installmentsCount_%s' % (flag, index))

        df_data['%s_per_time_installmentsSum_%s' % (flag, index)] = df_data['%s_installmentsSum_%s' % (flag, index)] / (
            np.abs(index))
        df_data['%s_per_time_installmentsCount_%s' % (flag, index)] = df_data[
                                                                          '%s_installmentsCount_%s' % (flag, index)] / (
                                                                          np.abs(index))

        # purchase_diff(强特)
        print('purchase_diff...')
        df_data = getMeanStaticsFeatures(df_data, df, ['card_id'], 'purchase_diff',
                                         name='%s_purchaseDiffMean_%s' % (flag, index))
        df_data = getMaxStaticsFeatures(df_data, df, ['card_id'], 'purchase_diff',
                                        name='%s_purchaseDiffMax_%s' % (flag, index))
        df_data = getMedianStaticsFeatures(df_data, df, ['card_id'], 'purchase_diff',
                                           name='%s_purchaseDiffMedian_%s' % (flag, index))
        df_data = getMinStaticsFeatures(df_data, df, ['card_id'], 'purchase_diff',
                                        name='%s_purchaseDiffMin_%s' % (flag, index))
        df_data = getStdStaticsFeatures(df_data, df, ['card_id'], 'purchase_diff',
                                        name='%s_purcahseDiffStd_%s' % (flag, index))

        # day_diff
        #         df_data = getMaxStaticsFeatures(df_data,df_features,['card_id'],'day_diff',name='%s_dayDiffMax_%s'%(flag,index))
        #         df_data = getMinStaticsFeatures(df_data,df_features,['card_id'],'day_diff',name='%s_dayDiffMin_%s'%(flag,index))
        #         df_data = getCategoryFrequenceMax(df_data,df_features,['card_id'],'day_diff',name='%s_dayDiffFrequenceMax_%s'%(flag,index))

        # 信用卡分期类别特征
        print('category_3...')
        for cate in ['category_3_A', 'category_3_B', 'category_3_C']:
            df_data = getSumStaticsFeatures(df_data, df, ['card_id'], cate, name='%s_%s_sum_%s' % (flag, cate, index))

    if 'hist' in flag:
        histflag = 'hist'
    else:
        histflag = 'new'
    # 信用卡授权统计
    print('authorized_flag...')
    for cate in ['Y', 'N']:
        df_data = getSumStaticsFeatures(df_data, df_features, ['card_id'], 'authorized_flag_%s' % cate,
                                        name='%s_authorized_flag_%s_sum_%s' % (flag, cate, index))
        df_data['%s_per_time_authorized_flag_%s_sum_%s' % (flag, cate, index)] = (
                df_data['%s_authorized_flag_%s_sum_%s' % (flag, cate, index)] / df_data[
            '%s_last_to_first_time' % histflag])

    ##date2Holiday
    print('date2Holiday...')
    df_data = getMaxStaticsFeatures(df_data, df_features, ['card_id'], 'date2Holiday', name='%s_date2HolidayMax' % flag)
    df_data = getMinStaticsFeatures(df_data, df_features, ['card_id'], 'date2Holiday', name='%s_date2HolidayMin' % flag)
    df_data = getMeanStaticsFeatures(df_data, df_features, ['card_id'], 'date2Holiday', name='%s_date2Holiday' % flag)

    # AuthorizedPurchase
    print('AuthorizedPurchase...')
    df_data = getAuthorizedPurchase(df_data, df_features, name='%s_' % flag)
    # purchase_is_outlier
    #     df_data = getSumStaticsFeatures(df_data,df_features,['card_id'],'purchase_is_outlier',name='%s_purchaseOutlier_sum'%flag)
    # month
    df_data = getCategoryCounts(df_data, df_features, ['card_id'], 'month', name='%s_activeMonthCount' % flag)
    df_data = getCategoryFrequenceMax(df_data, df_features, ['card_id'], 'month', name='%s_activeMonthMax' % flag)
    df_data = getCategoryFrequenceMaxRatio(df_data, df_features, ['card_id'], 'month',
                                           name='%s_activeMonthMaxRatio' % flag)
    df_data['%s_per_time_activeMonthCount' % flag] = (df_data['%s_activeMonthCount' % flag] /
                                                      df_data['%s_last_to_first_time' % histflag])
    # month_lag
    print('month_lag...')
    df_data = getMinStaticsFeatures(df_data, df_features, ['card_id'], 'month_lag', name='%s_monthLagMin' % flag)
    df_data = getMaxStaticsFeatures(df_data, df_features, ['card_id'], 'month_lag', name='%s_monthLagMax' % flag)
    # month_gap
    print('month_gap...')
    df_data = getCategoryCounts(df_data, df_features, ['card_id'], 'month_gap',
                                name='%s_monthGap_categoryCounts' % flag)
    df_data = getCategoryCountsRatio(df_data, df_features, ['card_id'], 'month_gap',
                                     name='%s_monthGap_categoryCountsRatio' % flag)

    # day_gap
    df_data = getMaxStaticsFeatures(df_data, df_features, ['card_id'], 'day_gap', name='%s_dayGapMax' % flag)
    df_data = getMinStaticsFeatures(df_data, df_features, ['card_id'], 'day_gap', name='%s_dayGapMin' % flag)
    # 回购率
    df_data = getUserRepurchaseRatio(df_data, df_features, name='%s_repurchase_merchant_ratio' % flag)
    df_data = getUserRepurchaseRatio(df_data, df_features, group=['card_id', 'merchant_category_id'],
                                     name='%s_repurchase_merCate_ratio' % flag)
    # purchase shift比例
    print('purchase shift...')
    df_features['%s_purchase_shift' % flag] = df_features.groupby(['card_id'])['purchase_amount'].apply(
        lambda series: series.shift(1)).values
    df_features['%s_purchase_shift' % flag].fillna(0, inplace=True)
    df_features['%s_purchase_shift_ratio' % flag] = df_features['%s_purchase_shift' % flag] / df_features[
        'purchase_amount']
    df_temp = df_features.groupby(['card_id'])['%s_purchase_shift_ratio' % flag].max().reset_index()
    df_data = df_data.merge(df_temp, on='card_id', how='left')
    # hour
    print('hour...')
    df_data = getMeanStaticsFeatures(df_data, df_features, ['card_id'], 'hour', name='%s_HourMean_' % (flag))
    df_data = getSumStaticsFeatures(df_data, df_features, ['card_id'], 'hour', name='%s_HourSum_' % (flag))
    df_data = getCountsStaticsFeatures(df_data, df_features, ['card_id'], 'hour', name='%s_HourCounts_' % (flag))
    # weekend
    df_data = getSumStaticsFeatures(df_data, df_features, ['card_id'], 'weekend', name='%s_weekendSum_' % (flag))
    # dayofweek
    print('dayofweek...')
    df_data = getMeanStaticsFeatures(df_data, df_features, ['card_id'], 'dayofweek', name='%s_dayofweekMean_' % flag)
    df_data = getCategoryCounts(df_data, df_features, ['card_id'], 'dayofweek', name='%s_dayofWeekCounts_' % flag)

    for cate in [1, 2, 3, 4, 5, 6]:
        df_data = getSumStaticsFeatures(df_data, df_features, ['card_id'], 'category_2_%s' % cate,
                                        name='%s_category_2_%s_sum' % (flag, cate))
        df_data['%s_per_time_category_2_%s_sum' % (flag, cate)] = (
                df_data['%s_category_2_%s_sum' % (flag, cate)] / df_data['%s_last_to_first_time' % histflag])

    categoryCols = cateCols + ['authorized_flag', 'merchant_id', 'city_id', 'category_1', 'category_2', 'category_3',
                               'merchant_category_id', 'subsector_id']
    # purchase_time_diff
    print('purchase_time_diff...')
    df_data = getMinStaticsFeatures(df_data, df_features, ['card_id'], 'purchase_time_diff_days',
                                    name='%s_purchase_time_diff_days_min_' % (flag))
    df_data = getMinStaticsFeatures(df_data, df_features, ['card_id'], 'purchase_time_diff_seconds',
                                    name='%s_purchase_time_diff_seconds_min_' % (flag))

    for fea in ['purchase_time_diff_seconds', 'purchase_time_diff_days']:
        print(fea)
        df_data = getCategoryFrequenceMax(df_data, df_features, ['card_id'], fea,
                                          name='%s_%s_frequenceMax' % (flag, fea))
        df_data = getCategoryCounts(df_data, df_features, ['card_id'], fea,
                                    name='%s_%s_categoryCounts' % (flag, fea))
        df_data = getCategoryFrequenceMaxRatio(df_data, df_features, ['card_id'], fea,
                                               name='%s_%s_frequenceMaxRatio' % (flag, fea))
        df_data = getCategoryCountsRatio(df_data, df_features, ['card_id'], fea,
                                         name='%s_%s_categoryCountsRatio' % (flag, fea))
    for fea in categoryCols:
        print('categoryCols:', fea)
        df_data = getCategoryFrequenceMax(df_data, df_features, ['card_id'], fea,
                                          name='%s_%s_frequenceMax' % (flag, fea))
        df_data = getCategoryCounts(df_data, df_features, ['card_id'], fea,
                                    name='%s_%s_categoryCounts' % (flag, fea))
        df_data = getCategoryFrequenceMaxRatio(df_data, df_features, ['card_id'], fea,
                                               name='%s_%s_frequenceMaxRatio' % (flag, fea))
        df_data = getCategoryCountsRatio(df_data, df_features, ['card_id'], fea,
                                         name='%s_%s_categoryCountsRatio' % (flag, fea))

# month_lag=0的消费平均消费水平
print('month_lag=0的消费平均消费水平。。。')
df_temp = df_hist_transactions[df_hist_transactions.month_lag == 0]
df_temp = df_temp.groupby(['card_id'])['purchase_amount'].mean().reset_index().rename(
    columns={'purchase_amount': 'hist_purchaseAmountMean_0'})
df_data = df_data.merge(df_temp, on='card_id', how='left')

del df_temp, df_hist_transactions, df_new_transactions
gc.collect()

df_data['avg_purchase_ratio_-1'] = df_data['hist_auth_Y_purchaseAmountMean_-1'] / df_data['hist_purchaseAmountMean_0']
df_data['avg_purchase_ratio_-3'] = df_data['hist_auth_Y_purchaseAmountMean_-3'] / df_data['hist_purchaseAmountMean_0']
df_data['avg_purchase_ratio_-5'] = df_data['hist_auth_Y_purchaseAmountMean_-5'] / df_data['hist_purchaseAmountMean_0']
df_data['avg_purchase_ratio_-7'] = df_data['hist_auth_Y_purchaseAmountMean_-7'] / df_data['hist_purchaseAmountMean_0']
df_data['avg_purchase_ratio_-13'] = df_data['hist_auth_Y_purchaseAmountMean_-13'] / df_data['hist_purchaseAmountMean_0']
df_data['avg_purchase_ratio_1'] = df_data['new_purchaseAmountMean_1'] / df_data['hist_purchaseAmountMean_0']
df_data['avg_purchase_ratio_2'] = df_data['new_purchaseAmountMean_2'] / df_data['hist_purchaseAmountMean_0']

df_data.replace([np.inf, -np.inf], np.nan, inplace=True)
df_data.fillna(0, inplace=True)

# 保存用户统计特征
df_data = reduce_mem_usage(df_data)
df_transactions = reduce_mem_usage(df_transactions)

### tmp02 to_csv
print('save to csv...')
df_data.to_csv('tmp02_df_data.csv', index=False)
df_transactions.to_csv('tmp02_df_transactions.csv', index=False)
