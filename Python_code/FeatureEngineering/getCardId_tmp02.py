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

print('start...')
df_data = pd.read_csv('tmp02_df_data.csv')
df_transactions = pd.read_csv('tmp02_df_transactions.csv')
# df_uid = pd.read_csv('tmp02_df_data.csv', usecols=['card_id'])

print('类别特征处理...')
# 类别特征label_encoding
cateCols = ['city_C1', 'C2_state', 'C2_state_subsector', 'subsector_city', 'auth_C3']
categoryCols = cateCols + ['authorized_flag', 'merchant_id', 'city_id', 'category_1', 'category_2', 'category_3',
                           'merchant_category_id', 'subsector_id']

df_transactions = label_encoding(df_transactions, ['merchant_id'])
df_hist_transactions = df_transactions[df_transactions.month_lag <= 0]
df_new_transactions = df_transactions[df_transactions.month_lag > 0]

'''
# 类别特征编码后的统计特征
for flag, df_features in zip(['hist', 'new'], [df_hist_transactions, df_new_transactions]):
    print(".................%s......................" % flag)
    # categoryCols
    for fea in categoryCols:
        print(fea)
        df_uid = getMeanStaticsFeatures(df_uid, df_features, ['card_id'], fea, name='%s_%s_Mean' % (flag, fea))
        df_uid = getMaxStaticsFeatures(df_uid, df_features, ['card_id'], fea, name='%s_%s_Max' % (flag, fea))
        df_uid = getMedianStaticsFeatures(df_uid, df_features, ['card_id'], fea, name='%s_%s_Median' % (flag, fea))
        df_uid = getSumStaticsFeatures(df_uid, df_features, ['card_id'], fea, name='%s_%s_Sum' % (flag, fea))
        df_uid = getStdStaticsFeatures(df_uid, df_features, ['card_id'], fea, name='%s_%s_Std' % (flag, fea))
        df_uid = getCountsStaticsFeatures(df_uid, df_features, ['card_id'], fea, name='%s_%s_Count' % (flag, fea))

print('save to csv：tmp03_df_cate_statics.csv')
df_uid.to_csv('tmp03_df_cate_statics.csv', index=False)
del df_uid
gc.collect()

# 节假日交易金额特征
print('节假日交易金额特征')
df_hist = df_hist_transactions.groupby(['card_id', 'date2Holiday'])['purchase_amount'].sum().reset_index()
df_hist = df_hist.groupby('card_id')['purchase_amount'].max().reset_index().rename(
    columns={'purchase_amount': 'hist_holiday_purchase_Max'})

df_new = df_new_transactions.groupby(['card_id', 'date2Holiday'])['purchase_amount'].sum().reset_index()
df_new = df_new.groupby('card_id')['purchase_amount'].max().reset_index().rename(
    columns={'purchase_amount': 'new_holiday_purchase_Max'})

df_temp = pd.merge(df_hist, df_new, on='card_id', how='left')
df_temp['new/hist_holiday_purchase_Max_ratio'] = df_temp['new_holiday_purchase_Max'] / df_temp[
    'hist_holiday_purchase_Max']
print('save to csv：tmp03_df_new_hist_holiday_purchase_Max_ratio.csv')
df_temp.to_csv('tmp03_df_new_hist_holiday_purchase_Max_ratio.csv', index=False)

# new/hist
print('new/hist ratio...')
# id类特征new/hist比例
for col in ['merchant_id', 'merchant_category_id', 'city_id', 'subsector_id']:
    print(col)
    df_hist_temp = df_hist_transactions.groupby('card_id')[col].unique().apply(lambda x: len(x)).reset_index().rename(
        columns={col: 'hist_%s_counts' % col})
    df_new_temp = df_new_transactions.groupby('card_id')[col].unique().apply(lambda x: len(x)).reset_index().rename(
        columns={col: 'new_%s_counts' % col})
    df_temp = pd.merge(df_hist_temp, df_new_temp, on='card_id', how='left')
    df_temp['new/hist_%s_counts_ratio' % col] = df_temp['new_%s_counts' % col] / df_temp['hist_%s_counts' % col]
    df_temp.drop(columns=['hist_%s_counts' % col, 'new_%s_counts' % col], inplace=True)
    df_temp.fillna(0, inplace=True)
    print('save to csv：tmp03_df_new_hist_%s_counts_ratio.csv' % col)
    df_temp.to_csv('tmp03_df_new_hist_%s_counts_ratio.csv' % col, index=False)

# 分期数、失败成功数统new/hist比例
for col in ['installments', 'authorized_flag']:
    df_hist_temp = df_hist_transactions.groupby('card_id')[col].sum().reset_index().rename(
        columns={col: 'hist_%s_sum' % col})
    df_new_temp = df_new_transactions.groupby('card_id')[col].sum().reset_index().rename(
        columns={col: 'new_%s_sum' % col})
    df_temp = pd.merge(df_hist_temp, df_new_temp, on='card_id', how='left')
    df_temp['new/hist_%s_sum_ratio' % col] = df_temp['new_%s_sum' % col] / df_temp['hist_%s_sum' % col]
    df_temp.drop(columns=['hist_%s_sum' % col, 'new_%s_sum' % col], inplace=True)
    df_temp.fillna(0, inplace=True)
    print('save to csv：tmp03_df_new_hist_%s_sum_ratio.csv' % col)
    df_temp.to_csv('tmp03_df_new_hist_%s_sum_ratio.csv' % col, index=False)

del df_hist_temp, df_new_temp, df_temp
gc.collect()

# merchant_category_id交易金额new/hist比例mean
df_hist = df_hist_transactions.groupby(['card_id', 'merchant_category_id'])['purchase_amount'].mean().reset_index()
df_hist = df_hist.groupby('card_id')['purchase_amount'].max().reset_index().rename(
    columns={'purchase_amount': 'hist_card_and_merchant_purchase_mean'})
df_new = df_new_transactions.groupby(['card_id', 'merchant_id'])['purchase_amount'].mean().reset_index()
df_new = df_new.groupby('card_id')['purchase_amount'].max().reset_index().rename(
    columns={'purchase_amount': 'new_card_and_merchant_purchase_mean'})
df_temp = pd.merge(df_hist, df_new, on='card_id', how='left')
df_temp['new/hist_card_and_merchant_category_id_purchase_mean_raio'] = df_temp['new_card_and_merchant_purchase_mean'] / \
                                                                       df_temp['hist_card_and_merchant_purchase_mean']
df_temp.drop(columns=['hist_card_and_merchant_purchase_mean', 'new_card_and_merchant_purchase_mean'], inplace=True)
df_temp.fillna(0, inplace=True)
print('save to csv：tmp03_df_new_and_merchant_category_id_purchase_mean_ratio.csv')
df_temp.to_csv('tmp03_df_new_and_merchant_category_id_purchase_mean_ratio.csv', index=False)
# merchant_category_id交易金额new/hist比例min
df_hist = df_hist_transactions.groupby(['card_id', 'merchant_category_id'])['purchase_amount'].mean().reset_index()
df_hist = df_hist.groupby('card_id')['purchase_amount'].min().reset_index().rename(
    columns={'purchase_amount': 'hist_card_and_merchant_purchase_mean'})
df_new = df_new_transactions.groupby(['card_id', 'merchant_id'])['purchase_amount'].mean().reset_index()
df_new = df_new.groupby('card_id')['purchase_amount'].min().reset_index().rename(
    columns={'purchase_amount': 'new_card_and_merchant_purchase_mean'})
df_temp = pd.merge(df_hist, df_new, on='card_id', how='left')
df_temp['new/hist_card_and_merchant_category_id_purchase_min_raio'] = df_temp['new_card_and_merchant_purchase_mean'] / \
                                                                      df_temp['hist_card_and_merchant_purchase_mean']
df_temp.drop(columns=['hist_card_and_merchant_purchase_mean', 'new_card_and_merchant_purchase_mean'], inplace=True)
df_temp.fillna(0, inplace=True)
print('save to csv：tmp03_df_new_and_merchant_category_id_purchase_min_ratio.csv')
df_temp.to_csv('tmp03_df_new_and_merchant_category_id_purchase_min_ratio.csv', index=False)
# merchant_category_id交易金额new/hist比例medium
df_hist = df_hist_transactions.groupby(['card_id', 'merchant_category_id'])['purchase_amount'].mean().reset_index()
df_hist = df_hist.groupby('card_id')['purchase_amount'].median().reset_index().rename(
    columns={'purchase_amount': 'hist_card_and_merchant_purchase_mean'})
df_new = df_new_transactions.groupby(['card_id', 'merchant_id'])['purchase_amount'].mean().reset_index()
df_new = df_new.groupby('card_id')['purchase_amount'].median().reset_index().rename(
    columns={'purchase_amount': 'new_card_and_merchant_purchase_mean'})
df_temp = pd.merge(df_hist, df_new, on='card_id', how='left')
df_temp['new/hist_card_and_merchant_category_id_purchase_median_raio'] = df_temp[
                                                                             'new_card_and_merchant_purchase_mean'] / \
                                                                         df_temp['hist_card_and_merchant_purchase_mean']
df_temp.drop(columns=['hist_card_and_merchant_purchase_mean', 'new_card_and_merchant_purchase_mean'], inplace=True)
df_temp.fillna(0, inplace=True)
print('save to csv：tmp03_df_new_and_merchant_category_id_purchase_median_ratio.csv')
df_temp.to_csv('tmp03_df_new_and_merchant_category_id_purchase_median_ratio.csv', index=False)

# new/hist (card,merchant) ratio mean
df_hist = df_hist_transactions.groupby(['card_id', 'merchant_id'])['purchase_amount'].mean().reset_index()
df_hist = df_hist.groupby('card_id')['purchase_amount'].max().reset_index().rename(
    columns={'purchase_amount': 'hist_card_and_merchant_purchase_mean'})
df_new = df_new_transactions.groupby(['card_id', 'merchant_id'])['purchase_amount'].mean().reset_index()
df_new = df_new.groupby('card_id')['purchase_amount'].max().reset_index().rename(
    columns={'purchase_amount': 'new_card_and_merchant_purchase_mean'})
df_temp = pd.merge(df_hist, df_new, on='card_id', how='left')
df_temp['new/hist_card_and_merchant_purchase_mean_raio'] = df_temp['new_card_and_merchant_purchase_mean'] / df_temp[
    'hist_card_and_merchant_purchase_mean']
df_temp.drop(columns=['hist_card_and_merchant_purchase_mean', 'new_card_and_merchant_purchase_mean'], inplace=True)
df_temp.fillna(0, inplace=True)
print('save to csv：tmp03_df_new_and_merchant_purchase_mean_ratio.csv')
df_temp.to_csv('tmp03_df_new_and_merchant_purchase_mean_ratio.csv', index=False)
# new/hist (card,merchant) ratio min
df_hist = df_hist_transactions.groupby(['card_id', 'merchant_id'])['purchase_amount'].mean().reset_index()
df_hist = df_hist.groupby('card_id')['purchase_amount'].min().reset_index().rename(
    columns={'purchase_amount': 'hist_card_and_merchant_purchase_mean'})
df_new = df_new_transactions.groupby(['card_id', 'merchant_id'])['purchase_amount'].mean().reset_index()
df_new = df_new.groupby('card_id')['purchase_amount'].min().reset_index().rename(
    columns={'purchase_amount': 'new_card_and_merchant_purchase_mean'})
df_temp = pd.merge(df_hist, df_new, on='card_id', how='left')
df_temp['new/hist_card_and_merchant_purchase_min_raio'] = df_temp['new_card_and_merchant_purchase_mean'] / df_temp[
    'hist_card_and_merchant_purchase_mean']
df_temp.drop(columns=['hist_card_and_merchant_purchase_mean', 'new_card_and_merchant_purchase_mean'], inplace=True)
df_temp.fillna(0, inplace=True)
print('save to csv：tmp03_df_new_and_merchant_purchase_min_ratio.csv')
df_temp.to_csv('tmp03_df_new_and_merchant_purchase_min_ratio.csv', index=False)
# new/hist (card,merchant) ratio median
df_hist = df_hist_transactions.groupby(['card_id', 'merchant_id'])['purchase_amount'].mean().reset_index()
df_hist = df_hist.groupby('card_id')['purchase_amount'].median().reset_index().rename(
    columns={'purchase_amount': 'hist_card_and_merchant_purchase_mean'})
df_new = df_new_transactions.groupby(['card_id', 'merchant_id'])['purchase_amount'].mean().reset_index()
df_new = df_new.groupby('card_id')['purchase_amount'].median().reset_index().rename(
    columns={'purchase_amount': 'new_card_and_merchant_purchase_mean'})
df_temp = pd.merge(df_hist, df_new, on='card_id', how='left')
df_temp['new/hist_card_and_merchant_purchase_median_raio'] = df_temp['new_card_and_merchant_purchase_mean'] / df_temp[
    'hist_card_and_merchant_purchase_mean']
df_temp.drop(columns=['hist_card_and_merchant_purchase_mean', 'new_card_and_merchant_purchase_mean'], inplace=True)
df_temp.fillna(0, inplace=True)
print('save to csv：tmp03_df_new_and_merchant_purchase_median_ratio.csv')
df_temp.to_csv('tmp03_df_new_and_merchant_purchase_median_ratio.csv', index=False)
'''

# month_lag new/hist ratio
print('month_lag new/hist ratio...')
month_lag = [0, -1, -2, -3, -4, -13]
for lag in month_lag:
    print('month_lag:', lag)
    df_feature = df_hist_transactions[df_hist_transactions.month_lag >= lag]
    df_hist_temp = df_feature.groupby('card_id')['purchase_amount'].sum().reset_index().rename(
        columns={'purchase_amount': 'hist_purchase_amount_sum'})
    df_new_temp = df_new_transactions.groupby('card_id')['purchase_amount'].sum().reset_index().rename(
        columns={'purchase_amount': 'new_purchase_amount_sum'})
    df_temp = pd.merge(df_hist_temp, df_new_temp, on='card_id', how='left')
    df_temp['new/hist_purchase_amount_sum_ratio_%s' % lag] = df_temp['new_purchase_amount_sum'] / df_temp[
        'hist_purchase_amount_sum']
    df_temp.drop(columns=['hist_purchase_amount_sum', 'new_purchase_amount_sum'], inplace=True)
    df_data = df_data.merge(df_temp, on='card_id', how='left')
    df_data['new/hist_purchase_amount_sum_ratio_%s' % lag].fillna(0, inplace=True)

# 不同分期付款购买的金额
print('installments-purchase_amount...')
for flag, df_features in zip(['hist', 'new'], [df_hist_transactions, df_new_transactions]):
    print(".................%s......................" % flag)
    df_features['install_flag'] = (df_features['installments'] > 0).astype(np.int)
    df_temp = df_features.groupby(['card_id', 'install_flag'])['purchase_amount'].sum().reset_index()
    df_temp = df_temp.pivot(index='card_id', columns='install_flag', values='purchase_amount')
    df_temp.columns.name = None
    df_temp.columns = ['%s_installments_0' % flag, '%s_installments_1' % flag]
    df_temp.reset_index(inplace=True)
    df_temp.fillna(0, inplace=True)
    df_temp['purchase_amount'] = df_temp['%s_installments_0' % flag] + df_temp['%s_installments_1' % flag]
    df_temp['%s_purchase_install_0_ratio' % flag] = df_temp['%s_installments_0' % flag] / df_temp['purchase_amount']
    df_temp['%s_purchase_install_1_ratio' % flag] = df_temp['%s_installments_1' % flag] / df_temp['purchase_amount']
    df_temp.drop(columns=['purchase_amount'], inplace=True)

    df_data = df_data.merge(df_temp, on='card_id', how='left')

# 不同分期等级购买的金额
print('category_3-purchase_amount...')
for flag, df_features in zip(['hist', 'new'], [df_hist_transactions, df_new_transactions]):
    print(".................%s......................" % flag)
    df_temp = df_features.groupby(['card_id', 'category_3'])['purchase_amount'].sum().reset_index()
    df_temp = df_temp.pivot(index='card_id', columns='category_3', values='purchase_amount')
    df_temp.columns.name = None
    cols = []
    for col in df_temp.columns:
        cols.append('%s_category_3_' % flag + np.str(col))
    df_temp.columns = cols
    df_temp.fillna(0, inplace=True)
    df_temp['purchase_amount'] = 0
    for col in cols:
        df_temp['purchase_amount'] += df_temp[col]
    for col in cols:
        df_temp[col + '_ratio'] = df_temp[col] / df_temp['purchase_amount']
    df_temp.reset_index(inplace=True)
    df_temp.drop(columns=['purchase_amount'], inplace=True)
    df_data = df_data.merge(df_temp, on='card_id', how='left')

# 不同category_1购买的金额
print('category_1-purchase_amount...')
for flag, df_features in zip(['hist', 'new'], [df_hist_transactions, df_new_transactions]):
    print(".................%s......................" % flag)
    df_temp = df_features.groupby(['card_id', 'category_1'])['purchase_amount'].sum().reset_index()
    df_temp = df_temp.pivot(index='card_id', columns='category_1', values='purchase_amount')
    df_temp.columns.name = None
    cols = []
    for col in df_temp.columns:
        cols.append('%s_category_1_' % flag + np.str(col))
    df_temp.columns = cols
    df_temp['purchase_amount'] = 0
    df_temp.fillna(0, inplace=True)
    for col in cols:
        df_temp['purchase_amount'] += df_temp[col]
    for col in cols:
        df_temp[col + '_ratio'] = df_temp[col] / df_temp['purchase_amount']

    df_temp.reset_index(inplace=True)
    df_temp.drop(columns=['purchase_amount'], inplace=True)

    df_data = df_data.merge(df_temp, on='card_id', how='left')

# 不同category_2购买的金额
print('category_2-purchase_amount...')
for flag, df_features in zip(['hist', 'new'], [df_hist_transactions, df_new_transactions]):
    print(".................%s......................" % flag)
    df_temp = df_features.groupby(['card_id', 'category_2'])['purchase_amount'].sum().reset_index()
    df_temp = df_temp.pivot(index='card_id', columns='category_2', values='purchase_amount')
    df_temp.columns.name = None
    cols = []
    for col in df_temp.columns:
        cols.append('%s_category_2_' % flag + np.str(col))
    df_temp.columns = cols
    df_temp['purchase_amount'] = 0
    df_temp.fillna(0, inplace=True)
    for col in cols:
        df_temp['purchase_amount'] += df_temp[col]
    for col in cols:
        df_temp[col + '_ratio'] = df_temp[col] / df_temp['purchase_amount']

    df_temp.reset_index(inplace=True)
    df_temp.drop(columns=['purchase_amount'], inplace=True)

    df_data = df_data.merge(df_temp, on='card_id', how='left')

# 不同subsector_id购买的金额
print('subsector_id-purchase_amount...')
for flag, df_features in zip(['hist', 'new'], [df_hist_transactions, df_new_transactions]):
    print(".................%s......................" % flag)
    df_temp = df_features.groupby(['card_id', 'subsector_id'])['purchase_amount'].sum().reset_index()
    df_temp = df_temp.pivot(index='card_id', columns='subsector_id', values='purchase_amount')
    df_temp.columns.name = None
    cols = []
    for col in df_temp.columns:
        cols.append('%s_subsector_id_' % flag + np.str(col))
    df_temp.columns = cols
    df_temp['purchase_amount'] = 0
    df_temp.fillna(0, inplace=True)
    for col in cols:
        df_temp['purchase_amount'] += df_temp[col]
    for col in cols:
        df_temp[col + '_ratio'] = df_temp[col] / df_temp['purchase_amount']

    df_temp.reset_index(inplace=True)
    df_temp.drop(columns=['purchase_amount'], inplace=True)

    df_data = df_data.merge(df_temp, on='card_id', how='left')

df_transactions = pd.read_csv('tmp02_df_transactions.csv')
df_merchant = pd.read_csv(DATA_PATH + 'merchants.csv', usecols=['category_4', 'merchant_id'])
df_transactions = df_transactions.merge(df_merchant, on='merchant_id', how='left')
df_hist_transactions = df_transactions[df_transactions.month_lag <= 0]
df_new_transactions = df_transactions[df_transactions.month_lag > 0]

# 不同category_4购买的金额
print('category_4-purchase_amount...')
for flag, df_features in zip(['hist', 'new'], [df_hist_transactions, df_new_transactions]):
    print(".................%s......................" % flag)
    df_temp = df_features.groupby(['card_id', 'category_4'])['purchase_amount'].sum().reset_index()
    df_temp = df_temp.pivot(index='card_id', columns='category_4', values='purchase_amount')
    df_temp.columns.name = None
    cols = []
    for col in df_temp.columns:
        cols.append('%s_category_4_' % flag + np.str(col))
    df_temp.columns = cols
    df_temp['purchase_amount'] = 0
    df_temp.fillna(0, inplace=True)
    for col in cols:
        df_temp['purchase_amount'] += df_temp[col]
    for col in cols:
        df_temp[col + '_ratio'] = df_temp[col] / df_temp['purchase_amount']

    df_temp.reset_index(inplace=True)
    df_temp.drop(columns=['purchase_amount'], inplace=True)

    df_data = df_data.merge(df_temp, on='card_id', how='left')

df_merchant = pd.read_csv(DATA_PATH + 'merchants.csv', usecols=['merchant_group_id', 'merchant_id'])
df_transactions = df_transactions.merge(df_merchant, on='merchant_id', how='left')
df_hist_transactions = df_transactions[df_transactions.month_lag <= 0]
df_new_transactions = df_transactions[df_transactions.month_lag > 0]

# 获取词向量
from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser


def getWord2Vec(df_temp=None, fea=None, embedding_size=10, name=None):
    corpus = df_temp[fea].values
    model = Word2Vec(corpus, size=embedding_size, window=3, min_count=1, workers=16)
    Q_VEC = np.zeros((len(corpus), embedding_size))
    cols = []
    for i in range(embedding_size):
        cols.append(name + '_vec_%s' % i)
    for i in range(df_temp.shape[0]):
        Q_VEC[i, :] = np.mean(model.wv[corpus[i]], axis=0)
    df_vec = pd.DataFrame(data=Q_VEC, columns=cols)
    df_vec['card_id'] = df_temp['card_id'].values
    return df_vec


# 购买次数序列构成的embedding
df_temp = df_transactions.groupby(['card_id', 'month'])['purchase_amount'].count().reset_index()
df_temp['purchase_amount'] = df_temp['purchase_amount'].astype(np.str)
df_temp = df_temp.groupby(['card_id'])['purchase_amount'].apply(lambda series: list(series)).reset_index()
df_temp.rename(columns={'purchase_amount': 'purchase_sequence'}, inplace=True)
df_vec = getWord2Vec(df_temp, fea='purchase_sequence', name='purchase_sequence')

df_data = df_data.merge(df_vec, on='card_id', how='left')

print('getWord2Vec...')
for flag, df_features in zip(['hist', 'new'], [df_hist_transactions, df_new_transactions]):
    print(".................%s......................" % flag)
    df_temp = df_features.groupby(['card_id', 'month'])['merchant_group_id'].count().reset_index()
    df_temp['merchant_group_id'] = df_temp['merchant_group_id'].astype(np.str)
    df_temp = df_temp.groupby(['card_id'])['merchant_group_id'].apply(lambda series: list(series)).reset_index()
    df_temp.rename(columns={'merchant_group_id': 'merchant_group_id_sequence'}, inplace=True)
    df_vec = getWord2Vec(df_temp, fea='merchant_group_id_sequence', name='%s_merchant_group_id_sequence' % flag)
    df_data = df_data.merge(df_vec, on='card_id', how='left')

df_data = reduce_mem_usage(df_data)

print('save to csv：tmp03_df_data.csv')
df_data.to_csv('tmp03_df_data.csv', index=False)

##additional features
print('additional features...')
cols = ['card_id', 'first_active_month', 'feature_1', 'feature_2', 'feature_3']
df_train = pd.read_csv(DATA_PATH + 'train.csv', usecols=cols + ['target'])
df_test = pd.read_csv(DATA_PATH + 'test.csv', usecols=cols)
df = pd.concat([df_train, df_test])
df['first_active_month'] = pd.to_datetime(df['first_active_month'])
df['quarter'] = df['first_active_month'].dt.quarter
df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

df['days_feature1'] = df['elapsed_time'] * df['feature_1']
df['days_feature2'] = df['elapsed_time'] * df['feature_2']
df['days_feature3'] = df['elapsed_time'] * df['feature_3']

df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']
df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']
df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']

df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
df['feature_mean'] = df['feature_sum'] / 3
df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

df.drop(columns=['first_active_month', 'target'], inplace=True)
print('save to csv：tmp03_df_train_test_features_additional.csv')
df.to_csv('tmp03_df_train_test_features_additional.csv', index=False)


def historical_transactions(num_rows=None):
    print('......historical......')
    # load csv
    hist_df = pd.read_csv(DATA_PATH + 'historical_transactions.csv', nrows=num_rows)

    # fillna
    hist_df['category_2'].fillna(1.0, inplace=True)
    hist_df['category_3'].fillna('A', inplace=True)
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
    hist_df['installments'].replace(-1, np.nan, inplace=True)
    hist_df['installments'].replace(999, np.nan, inplace=True)

    # trim
    hist_df['purchase_amount'] = hist_df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_3'] = hist_df['category_3'].map({'A': 0, 'B': 1, 'C': 2})
    # 交互特征
    hist_df['city_C1'] = ((hist_df['city_id'] < 0) + 0).astype(np.str) + '_' + hist_df['category_1'].astype(np.str)
    hist_df['C2_state'] = hist_df['category_2'].astype(np.str) + '_' + hist_df['state_id'].astype(np.str)
    hist_df['C2_state_subsector'] = hist_df['category_2'].astype(np.str) + '_' + hist_df['state_id'].astype(
        np.str) + '_' + hist_df['subsector_id'].astype(np.str)
    hist_df['subsector_city'] = hist_df['subsector_id'].astype(np.str) + '_' + ((hist_df['city_id'] < 0) + 0).astype(
        np.str)
    hist_df['auth_C3'] = hist_df['authorized_flag'].astype(np.str) + '_' + hist_df['category_3'].astype(np.str)

    cateCols = ['city_C1', 'C2_state', 'C2_state_subsector', 'subsector_city', 'auth_C3']
    hist_df = label_encoding(hist_df, cateCols)

    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    hist_df['month'] = hist_df['purchase_date'].dt.month
    hist_df['day'] = hist_df['purchase_date'].dt.day
    hist_df['hour'] = hist_df['purchase_date'].dt.hour
    hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
    hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
    hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >= 5).astype(int)

    # Christmas : December 25 2017
    hist_df['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Mothers Day: May 14 2017
    hist_df['Mothers_Day_2017'] = (pd.to_datetime('2017-06-04') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # fathers day: August 13 2017
    hist_df['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Childrens day: October 12 2017
    hist_df['Children_day_2017'] = (pd.to_datetime('2017-10-12') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Valentine's Day : 12th June, 2017
    hist_df['Valentine_Day_2017'] = (pd.to_datetime('2017-06-12') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Black Friday : 24th November 2017
    hist_df['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)

    # 2018
    # Mothers Day: May 13 2018
    hist_df['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - hist_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)

    hist_df['month_diff'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days) // 30
    hist_df['month_diff'] += hist_df['month_lag']

    # additional features
    df_temp = hist_df.groupby(['card_id', 'merchant_id'])['purchase_amount'].count().reset_index()
    df_temp.rename(columns={'purchase_amount': 'card_merchant_counts_totals'}, inplace=True)
    hist_df = hist_df.merge(df_temp, on=['card_id', 'merchant_id'], how='left')

    hist_df['duration'] = hist_df['purchase_amount'] * hist_df['month_diff']
    hist_df['duration/Visitcounts'] = hist_df['duration'] / hist_df['card_merchant_counts_totals']
    hist_df['duration/sqrtVisits'] = hist_df['duration'] / np.sqrt(hist_df['card_merchant_counts_totals'])
    hist_df['durations_log1p_visits'] = hist_df['duration'] * np.log1p(hist_df['card_merchant_counts_totals'])

    hist_df['purchase/Visitcounts'] = hist_df['purchase_amount'] / hist_df['card_merchant_counts_totals']
    hist_df['purchase/sqrtVisits'] = hist_df['purchase_amount'] / np.sqrt(hist_df['card_merchant_counts_totals'])
    hist_df['purchase_log1p_visits'] = hist_df['purchase_amount'] * np.log1p(hist_df['card_merchant_counts_totals'])

    hist_df['purchase_amount_installments'] = hist_df['purchase_amount'] * hist_df['installments']
    hist_df['price'] = hist_df['purchase_amount'] / hist_df['installments']
    hist_df['purchase_amount_authorized_flag'] = hist_df['purchase_amount'] * hist_df['authorized_flag']
    hist_df['purchase_amount_category_1'] = hist_df['purchase_amount'] * hist_df['category_1']
    hist_df['purchase_amount_category_2'] = hist_df['purchase_amount'] * hist_df['category_2']

    hist_df['amount_month_ratio'] = hist_df['purchase_amount'] / hist_df['month_diff']

    hist_df['purchase_lag'] = hist_df['purchase_amount'] * hist_df['month_lag']
    hist_df['purchase/month_lag'] = hist_df['purchase_amount'] / hist_df['month_lag']

    hist_df['installments_month_diff'] = hist_df['installments'] * hist_df['month_diff']
    hist_df['installments_month_diff_ratio'] = hist_df['installments'] / hist_df['month_diff']

    hist_df['purchase_amount_weekend'] = hist_df['purchase_amount'] * hist_df['weekend']

    df_temp = hist_df.groupby(['card_id'])['purchase_amount'].count().reset_index()
    df_temp.rename(columns={'purchase_amount': 'card_visit_counts'}, inplace=True)
    hist_df = hist_df.merge(df_temp, on='card_id', how='left')

    hist_df['duration/CardVisitcounts'] = hist_df['duration'] / hist_df['card_visit_counts']
    hist_df['duration/sqrtCardVisits'] = hist_df['duration'] / np.sqrt(hist_df['card_visit_counts'])
    hist_df['durations_log1p_Card_visits'] = hist_df['duration'] * np.log1p(hist_df['card_visit_counts'])

    hist_df['purchase/CardVisitcounts'] = hist_df['purchase_amount'] / hist_df['card_visit_counts']
    hist_df['purchase/sqrtCardVisits'] = hist_df['purchase_amount'] / np.sqrt(hist_df['card_visit_counts'])
    hist_df['purchase_log1p_Card_visits'] = hist_df['purchase_amount'] * np.log1p(hist_df['card_visit_counts'])

    hist_df['month_diff_installments_card_merchant_counts'] = hist_df['month_diff'] * hist_df['installments'] * hist_df[
        'card_merchant_counts_totals']
    hist_df['month_diff_installments_card_merchant_counts_ratio'] = hist_df['month_diff'] * hist_df['installments'] / \
                                                                    hist_df['card_merchant_counts_totals']

    hist_df['month_diff_installments_card_counts'] = hist_df['month_diff'] * hist_df['card_visit_counts'] * hist_df[
        'installments']
    hist_df['month_diff_installments_card_counts_ratio'] = hist_df['month_diff'] * hist_df['installments'] / hist_df[
        'card_visit_counts']

    ##加入month_gap
    hist_df['month_gap'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days) // 30
    hist_df['purchase_multi_month_gap'] = hist_df['purchase_amount'] * hist_df['month_gap']
    hist_df['purchase_visitcounts_ratio'] = hist_df['purchase_multi_month_gap'] / hist_df['card_merchant_counts_totals']
    hist_df['purchase_sqrtvisits_ratio'] = hist_df['purchase_multi_month_gap'] / hist_df['card_merchant_counts_totals']
    hist_df['purchase_log1pvisits_ratio'] = hist_df['purchase_multi_month_gap'] / hist_df['card_merchant_counts_totals']

    hist_df['purchase_card_visits_ratio'] = hist_df['purchase_multi_month_gap'] / hist_df['card_visit_counts']
    hist_df['purchase_sqrt_card_visits_ratio'] = hist_df['purchase_multi_month_gap'] / hist_df['card_visit_counts']
    hist_df['purchase_log1p_card_visits_ratio'] = hist_df['purchase_multi_month_gap'] / hist_df['card_visit_counts']

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}
    for col in ['Christmas_Day_2017', 'Mothers_Day_2017', 'fathers_day_2017', 'Children_day_2017', 'Valentine_Day_2017',
                'Mothers_Day_2018']:
        hist_df['purchase_amount_%s' % col] = hist_df['purchase_amount'] * hist_df[col]
        aggs[col] = ['min', 'max', 'sum', 'mean', 'median']

    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var', 'skew']
    aggs['installments'] = ['sum', 'max', 'mean', 'var', 'skew']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['month_diff'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['authorized_flag'] = ['mean']
    aggs['weekend'] = ['mean']  # overwrite
    aggs['weekday'] = ['mean']  # overwrite
    aggs['day'] = ['nunique', 'mean', 'min']  # overwrite
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['card_id'] = ['size', 'count']
    aggs['price'] = ['sum', 'mean', 'max', 'min', 'var']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Mothers_Day_2017'] = ['mean']
    aggs['fathers_day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Valentine_Day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']

    aggs['duration'] = ['mean', 'min', 'max']
    aggs['amount_month_ratio'] = ['mean', 'min', 'max']
    aggs['installments_month_diff'] = ['mean', 'min', 'max']
    aggs['installments_month_diff_ratio'] = ['mean', 'min', 'max']
    aggs['purchase_lag'] = ['mean', 'min', 'max', 'median']
    aggs['purchase/month_lag'] = ['mean', 'min', 'max', 'median']
    aggs['purchase_amount_weekend'] = ['mean', 'min', 'max', 'median']
    aggs['duration/Visitcounts'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['durations_log1p_visits'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['duration/sqrtVisits'] = ['mean', 'sum', 'min', 'max', 'median']

    aggs['purchase/Visitcounts'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['purchase_log1p_visits'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['purchase/sqrtVisits'] = ['mean', 'sum', 'min', 'max', 'median']

    aggs['purchase_amount_installments'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['purchase_amount_authorized_flag'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['purchase_amount_category_1'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['purchase_amount_category_2'] = ['mean', 'sum', 'min', 'max', 'median']

    aggs['duration/CardVisitcounts'] = ['mean', 'max', 'min', 'median']
    aggs['duration/sqrtCardVisits'] = ['mean', 'max', 'min', 'median']
    aggs['durations_log1p_Card_visits'] = ['mean', 'max', 'min', 'median']

    aggs['purchase/CardVisitcounts'] = ['mean', 'max', 'min', 'median']
    aggs['purchase/sqrtCardVisits'] = ['mean', 'max', 'min', 'median']
    aggs['purchase_log1p_Card_visits'] = ['mean', 'max', 'min', 'median']

    aggs['month_diff_installments_card_merchant_counts'] = ['max', 'min', 'mean', 'median']
    aggs['month_diff_installments_card_merchant_counts_ratio'] = ['max', 'min', 'mean', 'median']
    aggs['month_diff_installments_card_counts'] = ['max', 'min', 'mean', 'median']
    aggs['month_diff_installments_card_counts_ratio'] = ['max', 'min', 'mean', 'median']

    aggs['month_gap'] = ['mean', 'min', 'max']
    aggs['purchase_multi_month_gap'] = ['mean', 'max', 'min', 'median']
    aggs['purchase_visitcounts_ratio'] = ['mean', 'max', 'min', 'median']
    aggs['purchase_sqrtvisits_ratio'] = ['mean', 'max', 'min', 'median']
    aggs['purchase_log1pvisits_ratio'] = ['mean', 'max', 'min', 'median']

    aggs['purchase_card_visits_ratio'] = ['mean', 'max', 'min', 'median']
    aggs['purchase_sqrt_card_visits_ratio'] = ['mean', 'max', 'min', 'median']
    aggs['purchase_log1p_card_visits_ratio'] = ['mean', 'max', 'min', 'median']

    cateCols = ['city_C1', 'C2_state', 'C2_state_subsector', 'subsector_city', 'auth_C3']
    hist_df = label_encoding(hist_df, cateCols)
    for col in cateCols:
        aggs[col] = ['mean']

    for col in cateCols + ['category_2', 'category_3']:
        hist_df[col + '_mean'] = hist_df.groupby([col])['purchase_amount'].transform('mean')
        hist_df[col + '_min'] = hist_df.groupby([col])['purchase_amount'].transform('min')
        hist_df[col + '_max'] = hist_df.groupby([col])['purchase_amount'].transform('max')
        hist_df[col + '_sum'] = hist_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col + '_mean'] = ['mean']

    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['hist_' + c for c in hist_df.columns]

    hist_df['hist_purchase_date_diff'] = (hist_df['hist_purchase_date_max'] - hist_df['hist_purchase_date_min']).dt.days
    hist_df['hist_purchase_date_average'] = hist_df['hist_purchase_date_diff'] / hist_df['hist_card_id_size']
    hist_df['hist_purchase_date_uptonow'] = (datetime.datetime.today() - hist_df['hist_purchase_date_max']).dt.days
    hist_df['hist_purchase_date_uptomin'] = (datetime.datetime.today() - hist_df['hist_purchase_date_min']).dt.days

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    return hist_df


# preprocessing new_merchant_transactions
def new_merchant_transactions(num_rows=None):
    print('....new_merchant.......')
    # load csv
    new_merchant_df = pd.read_csv(DATA_PATH + 'new_merchant_transactions.csv', nrows=num_rows)

    # fillna
    new_merchant_df['category_2'].fillna(1.0, inplace=True)
    new_merchant_df['category_3'].fillna('A', inplace=True)
    new_merchant_df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
    new_merchant_df['installments'].replace(-1, np.nan, inplace=True)
    new_merchant_df['installments'].replace(999, np.nan, inplace=True)

    # trim
    new_merchant_df['purchase_amount'] = new_merchant_df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    new_merchant_df['authorized_flag'] = new_merchant_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    new_merchant_df['category_1'] = new_merchant_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    new_merchant_df['category_3'] = new_merchant_df['category_3'].map({'A': 0, 'B': 1, 'C': 2}).astype(int)

    # 交互特征
    new_merchant_df['city_C1'] = ((new_merchant_df['city_id'] < 0) + 0).astype(np.str) + '_' + new_merchant_df[
        'category_1'].astype(np.str)
    new_merchant_df['C2_state'] = new_merchant_df['category_2'].astype(np.str) + '_' + new_merchant_df[
        'state_id'].astype(np.str)
    new_merchant_df['C2_state_subsector'] = new_merchant_df['category_2'].astype(np.str) + '_' + new_merchant_df[
        'state_id'].astype(np.str) + '_' + new_merchant_df['subsector_id'].astype(np.str)
    new_merchant_df['subsector_city'] = new_merchant_df['subsector_id'].astype(np.str) + '_' + (
            (new_merchant_df['city_id'] < 0) + 0).astype(np.str)
    new_merchant_df['auth_C3'] = new_merchant_df['authorized_flag'].astype(np.str) + '_' + new_merchant_df[
        'category_3'].astype(np.str)

    cateCols = ['city_C1', 'C2_state', 'C2_state_subsector', 'subsector_city', 'auth_C3']
    new_merchant_df = label_encoding(new_merchant_df, cateCols)

    # datetime features
    new_merchant_df['purchase_date'] = pd.to_datetime(new_merchant_df['purchase_date'])
    new_merchant_df['month'] = new_merchant_df['purchase_date'].dt.month
    new_merchant_df['day'] = new_merchant_df['purchase_date'].dt.day
    new_merchant_df['hour'] = new_merchant_df['purchase_date'].dt.hour
    new_merchant_df['weekofyear'] = new_merchant_df['purchase_date'].dt.weekofyear
    new_merchant_df['weekday'] = new_merchant_df['purchase_date'].dt.weekday
    new_merchant_df['weekend'] = (new_merchant_df['purchase_date'].dt.weekday >= 5).astype(int)

    # fathers day: August 13 2017
    new_merchant_df['fathers_day_2017'] = (
            pd.to_datetime('2017-08-13') - new_merchant_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Valentine's Day : 12th June, 2017
    new_merchant_df['Valentine_Day_2017'] = (
            pd.to_datetime('2017-06-12') - new_merchant_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Christmas : December 25 2017
    new_merchant_df['Christmas_Day_2017'] = (
            pd.to_datetime('2017-12-25') - new_merchant_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Childrens day: October 12 2017
    new_merchant_df['Children_day_2017'] = (
            pd.to_datetime('2017-10-12') - new_merchant_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Black Friday : 24th November 2017
    new_merchant_df['Black_Friday_2017'] = (
            pd.to_datetime('2017-11-24') - new_merchant_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    new_merchant_df['Mothers_Day_2017'] = (
            pd.to_datetime('2017-06-04') - new_merchant_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)

    # Mothers Day: May 13 2018
    new_merchant_df['Mothers_Day_2018'] = (
            pd.to_datetime('2018-05-13') - new_merchant_df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)

    new_merchant_df['month_diff'] = ((datetime.datetime.today() - new_merchant_df['purchase_date']).dt.days) // 30
    new_merchant_df['month_diff'] += new_merchant_df['month_lag']

    # additional features
    df_temp = new_merchant_df.groupby(['card_id', 'merchant_id'])['purchase_amount'].count().reset_index()
    df_temp.rename(columns={'purchase_amount': 'card_merchant_counts_totals'}, inplace=True)
    new_merchant_df = new_merchant_df.merge(df_temp, on=['card_id', 'merchant_id'], how='left')

    new_merchant_df['duration'] = new_merchant_df['purchase_amount'] * new_merchant_df['month_diff']
    new_merchant_df['amount_month_ratio'] = new_merchant_df['purchase_amount'] / new_merchant_df['month_diff']
    new_merchant_df['installments_month_diff'] = new_merchant_df['installments'] * new_merchant_df['month_diff']
    new_merchant_df['installments_month_diff_ratio'] = new_merchant_df['installments'] / new_merchant_df['month_diff']

    new_merchant_df['duration/Visitcounts'] = new_merchant_df['duration'] / new_merchant_df[
        'card_merchant_counts_totals']
    new_merchant_df['duration/sqrtVisits'] = new_merchant_df['duration'] / np.sqrt(
        new_merchant_df['card_merchant_counts_totals'])
    new_merchant_df['durations_log1p_visits'] = new_merchant_df['duration'] * np.log1p(
        new_merchant_df['card_merchant_counts_totals'])

    new_merchant_df['purchase/Visitcounts'] = new_merchant_df['purchase_amount'] / new_merchant_df[
        'card_merchant_counts_totals']
    new_merchant_df['purchase/sqrtVisits'] = new_merchant_df['purchase_amount'] / np.sqrt(
        new_merchant_df['card_merchant_counts_totals'])
    new_merchant_df['purchase_log1p_visits'] = new_merchant_df['purchase_amount'] * np.log1p(
        new_merchant_df['card_merchant_counts_totals'])

    new_merchant_df['purchase_amount_installments'] = new_merchant_df['purchase_amount'] * new_merchant_df[
        'installments']
    new_merchant_df['price'] = new_merchant_df['purchase_amount'] / new_merchant_df['installments']
    new_merchant_df['purchase_amount_authorized_flag'] = new_merchant_df['purchase_amount'] * new_merchant_df[
        'authorized_flag']
    new_merchant_df['purchase_amount_category_1'] = new_merchant_df['purchase_amount'] * new_merchant_df['category_1']
    new_merchant_df['purchase_amount_category_2'] = new_merchant_df['purchase_amount'] * new_merchant_df['category_2']
    new_merchant_df['price'] = new_merchant_df['purchase_amount'] / new_merchant_df['installments']

    new_merchant_df['amount_month_ratio'] = new_merchant_df['purchase_amount'] / new_merchant_df['month_diff']

    new_merchant_df['purchase_lag'] = new_merchant_df['purchase_amount'] * new_merchant_df['month_lag']
    new_merchant_df['purchase/month_lag'] = new_merchant_df['purchase_amount'] / new_merchant_df['month_lag']

    new_merchant_df['installments_month_diff'] = new_merchant_df['installments'] * new_merchant_df['month_diff']
    new_merchant_df['installments_month_diff_ratio'] = new_merchant_df['installments'] / new_merchant_df['month_diff']

    new_merchant_df['purchase_amount_weekend'] = new_merchant_df['purchase_amount'] * new_merchant_df['weekend']

    df_temp = new_merchant_df.groupby(['card_id'])['purchase_amount'].count().reset_index()
    df_temp.rename(columns={'purchase_amount': 'card_visit_counts'}, inplace=True)
    new_merchant_df = new_merchant_df.merge(df_temp, on='card_id', how='left')

    new_merchant_df['duration/CardVisitcounts'] = new_merchant_df['duration'] / new_merchant_df['card_visit_counts']
    new_merchant_df['duration/sqrtCardVisits'] = new_merchant_df['duration'] / np.sqrt(
        new_merchant_df['card_visit_counts'])
    new_merchant_df['durations_log1p_Card_visits'] = new_merchant_df['duration'] * np.log1p(
        new_merchant_df['card_visit_counts'])

    new_merchant_df['purchase/CardVisitcounts'] = new_merchant_df['purchase_amount'] / new_merchant_df[
        'card_visit_counts']
    new_merchant_df['purchase/sqrtCardVisits'] = new_merchant_df['purchase_amount'] / np.sqrt(
        new_merchant_df['card_visit_counts'])
    new_merchant_df['purchase_log1p_Card_visits'] = new_merchant_df['purchase_amount'] * np.log1p(
        new_merchant_df['card_visit_counts'])

    new_merchant_df['month_diff_installments_card_merchant_counts'] = new_merchant_df['month_diff'] * new_merchant_df[
        'installments'] * new_merchant_df['card_merchant_counts_totals']
    new_merchant_df['month_diff_installments_card_merchant_counts_ratio'] = new_merchant_df['month_diff'] * \
                                                                            new_merchant_df['installments'] / \
                                                                            new_merchant_df[
                                                                                'card_merchant_counts_totals']

    new_merchant_df['month_diff_installments_card_counts'] = new_merchant_df['month_diff'] * new_merchant_df[
        'card_visit_counts'] * new_merchant_df['installments']
    new_merchant_df['month_diff_installments_card_counts_ratio'] = new_merchant_df['month_diff'] * new_merchant_df[
        'installments'] / new_merchant_df['card_visit_counts']

    ##加入month_gap
    new_merchant_df['month_gap'] = ((datetime.datetime.today() - new_merchant_df['purchase_date']).dt.days) // 30
    new_merchant_df['purchase_multi_month_gap'] = new_merchant_df['purchase_amount'] * new_merchant_df['month_gap']
    new_merchant_df['purchase_visitcounts_ratio'] = new_merchant_df['purchase_multi_month_gap'] / new_merchant_df[
        'card_merchant_counts_totals']
    new_merchant_df['purchase_sqrtvisits_ratio'] = new_merchant_df['purchase_multi_month_gap'] / new_merchant_df[
        'card_merchant_counts_totals']
    new_merchant_df['purchase_log1pvisits_ratio'] = new_merchant_df['purchase_multi_month_gap'] / new_merchant_df[
        'card_merchant_counts_totals']

    new_merchant_df['purchase_card_visits_ratio'] = new_merchant_df['purchase_multi_month_gap'] / new_merchant_df[
        'card_visit_counts']
    new_merchant_df['purchase_sqrt_card_visits_ratio'] = new_merchant_df['purchase_multi_month_gap'] / new_merchant_df[
        'card_visit_counts']
    new_merchant_df['purchase_log1p_card_visits_ratio'] = new_merchant_df['purchase_multi_month_gap'] / new_merchant_df[
        'card_visit_counts']

    # reduce memory usage
    new_merchant_df = reduce_mem_usage(new_merchant_df)

    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}
    for col in ['Christmas_Day_2017', 'Mothers_Day_2017', 'fathers_day_2017', 'Children_day_2017', 'Valentine_Day_2017',
                'Mothers_Day_2018']:
        new_merchant_df['purchase_amount_%s' % col] = new_merchant_df['purchase_amount'] * new_merchant_df[col]
        aggs[col] = ['min', 'max', 'sum', 'mean', 'median']

    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var', 'skew']
    aggs['installments'] = ['sum', 'max', 'mean', 'var', 'skew']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['month_diff'] = ['mean', 'var', 'skew']
    aggs['weekend'] = ['mean']
    aggs['month'] = ['mean', 'min', 'max']
    aggs['weekday'] = ['mean', 'min', 'max']
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['card_id'] = ['size', 'count']
    aggs['price'] = ['mean', 'max', 'min', 'var']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']

    aggs['duration'] = ['mean', 'min', 'max']
    aggs['amount_month_ratio'] = ['mean', 'min', 'max']
    aggs['installments_month_diff'] = ['mean', 'min', 'max']
    aggs['installments_month_diff_ratio'] = ['mean', 'min', 'max']
    #     aggs['purchase_exp'] = ['mean','min','max']
    aggs['purchase_lag'] = ['mean', 'min', 'max', 'median']
    aggs['purchase/month_lag'] = ['mean', 'min', 'max', 'median']
    aggs['purchase_amount_weekend'] = ['mean', 'min', 'max', 'median']

    aggs['duration/Visitcounts'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['durations_log1p_visits'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['duration/sqrtVisits'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['purchase/Visitcounts'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['purchase_log1p_visits'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['purchase/sqrtVisits'] = ['mean', 'sum', 'min', 'max', 'median']

    aggs['purchase_amount_installments'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['purchase_amount_authorized_flag'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['purchase_amount_category_1'] = ['mean', 'sum', 'min', 'max', 'median']
    aggs['purchase_amount_category_2'] = ['mean', 'sum', 'min', 'max', 'median']

    aggs['duration/CardVisitcounts'] = ['mean', 'max', 'min', 'median']
    aggs['duration/sqrtCardVisits'] = ['mean', 'max', 'min', 'median']
    aggs['durations_log1p_Card_visits'] = ['mean', 'max', 'min', 'median']

    aggs['purchase/CardVisitcounts'] = ['mean', 'max', 'min', 'median']
    aggs['purchase/sqrtCardVisits'] = ['mean', 'max', 'min', 'median']
    aggs['purchase_log1p_Card_visits'] = ['mean', 'max', 'min', 'median']

    aggs['month_diff_installments_card_merchant_counts'] = ['max', 'min', 'mean', 'median']
    aggs['month_diff_installments_card_merchant_counts_ratio'] = ['max', 'min', 'mean', 'median']
    aggs['month_diff_installments_card_counts'] = ['max', 'min', 'mean', 'median']
    aggs['month_diff_installments_card_counts_ratio'] = ['max', 'min', 'mean', 'median']

    aggs['month_gap'] = ['mean', 'min', 'max']
    aggs['purchase_multi_month_gap'] = ['mean', 'max', 'min', 'median']
    aggs['purchase_visitcounts_ratio'] = ['mean', 'max', 'min', 'median']
    aggs['purchase_sqrtvisits_ratio'] = ['mean', 'max', 'min', 'median']
    aggs['purchase_log1pvisits_ratio'] = ['mean', 'max', 'min', 'median']

    aggs['purchase_card_visits_ratio'] = ['mean', 'max', 'min', 'median']
    aggs['purchase_sqrt_card_visits_ratio'] = ['mean', 'max', 'min', 'median']
    aggs['purchase_log1p_card_visits_ratio'] = ['mean', 'max', 'min', 'median']

    for col in cateCols:
        aggs[col] = ['mean']

    for col in cateCols + ['category_2', 'category_3']:
        new_merchant_df[col + '_mean'] = new_merchant_df.groupby([col])['purchase_amount'].transform('mean')
        new_merchant_df[col + '_min'] = new_merchant_df.groupby([col])['purchase_amount'].transform('min')
        new_merchant_df[col + '_max'] = new_merchant_df.groupby([col])['purchase_amount'].transform('max')
        new_merchant_df[col + '_sum'] = new_merchant_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col + '_mean'] = ['mean']

    new_merchant_df = new_merchant_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    new_merchant_df.columns = pd.Index([e[0] + "_" + e[1] for e in new_merchant_df.columns.tolist()])
    new_merchant_df.columns = ['new_' + c for c in new_merchant_df.columns]

    new_merchant_df['new_purchase_date_diff'] = (
            new_merchant_df['new_purchase_date_max'] - new_merchant_df['new_purchase_date_min']).dt.days
    new_merchant_df['new_purchase_date_average'] = new_merchant_df['new_purchase_date_diff'] / new_merchant_df[
        'new_card_id_size']
    new_merchant_df['new_purchase_date_uptonow'] = (
            datetime.datetime.today() - new_merchant_df['new_purchase_date_max']).dt.days
    new_merchant_df['new_purchase_date_uptomin'] = (
            datetime.datetime.today() - new_merchant_df['new_purchase_date_min']).dt.days

    # reduce memory usage
    new_merchant_df = reduce_mem_usage(new_merchant_df)

    return new_merchant_df


# additional features
def additional_features(df):
    print('......additional........')
    date_features = ['hist_purchase_date_max', 'hist_purchase_date_min',
                     'new_purchase_date_max', 'new_purchase_date_min']

    for f in date_features:
        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_card_id_size'] + df['hist_card_id_size']
    df['card_id_cnt_total'] = df['new_card_id_count'] + df['hist_card_id_count']
    df['card_id_cnt_ratio'] = df['new_card_id_count'] / df['hist_card_id_count']
    df['purchase_amount_total'] = df['new_purchase_amount_sum'] + df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['new_purchase_amount_mean'] + df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['new_purchase_amount_max'] + df['hist_purchase_amount_max']
    df['purchase_amount_min'] = df['new_purchase_amount_min'] + df['hist_purchase_amount_min']
    df['purchase_amount_ratio'] = df['new_purchase_amount_sum'] / df['hist_purchase_amount_sum']
    df['month_diff_mean'] = df['new_month_diff_mean'] + df['hist_month_diff_mean']
    df['month_diff_ratio'] = df['new_month_diff_mean'] / df['hist_month_diff_mean']
    df['month_lag_mean'] = df['new_month_lag_mean'] + df['hist_month_lag_mean']
    df['month_lag_max'] = df['new_month_lag_max'] + df['hist_month_lag_max']
    df['month_lag_min'] = df['new_month_lag_min'] + df['hist_month_lag_min']
    df['category_1_mean'] = df['new_category_1_mean'] + df['hist_category_1_mean']
    df['installments_total'] = df['new_installments_sum'] + df['hist_installments_sum']
    df['installments_mean'] = df['new_installments_mean'] + df['hist_installments_mean']
    df['installments_max'] = df['new_installments_max'] + df['hist_installments_max']
    df['installments_ratio'] = df['new_installments_sum'] / df['hist_installments_sum']
    df['price_total'] = df['purchase_amount_total'] / df['installments_total']
    df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
    df['price_max'] = df['purchase_amount_max'] / df['installments_max']
    df['duration_mean'] = df['new_duration_mean'] + df['hist_duration_mean']
    df['duration_min'] = df['new_duration_min'] + df['hist_duration_min']
    df['duration_max'] = df['new_duration_max'] + df['hist_duration_max']
    df['amount_month_ratio_mean'] = df['new_amount_month_ratio_mean'] + df['hist_amount_month_ratio_mean']
    df['amount_month_ratio_min'] = df['new_amount_month_ratio_min'] + df['hist_amount_month_ratio_min']
    df['amount_month_ratio_max'] = df['new_amount_month_ratio_max'] + df['hist_amount_month_ratio_max']
    df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
    df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']
    df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']

    return df


df_temp = historical_transactions(num_rows=None)
df_temp = pd.merge(df_temp, new_merchant_transactions(num_rows=None), on='card_id', how='left')
df_temp = additional_features(df_temp)
print('...done.....')

df_temp.replace([np.inf, -np.inf], -9999, inplace=True)
df_temp.fillna(0, inplace=True)
df_temp.reset_index(inplace=True)

# cols = [_f for _f in df_temp.columns if 'month' in _f]
cols = ['new_purchase_amount_sum', 'new_purchase_amount_max', 'new_purchase_amount_min', 'new_purchase_amount_mean']
df_temp.drop(columns=cols, inplace=True)
df_temp.head()

df_temp = reduce_mem_usage(df_temp)

print('save to csv：tmp03_df_additional_features.csv')
df_temp.to_csv('tmp03_df_additional_features.csv', index=False)

df_temp = pd.read_csv('tmp03_df_additional_features.csv')

feats = list(set(df_temp.columns) - set(['target', 'card_id', 'is_test']))
cr = df_temp[feats].corr()

interactions = []
for col in cr.columns:
    inter_col = cr[cr[col] == cr[col].min()].index[0]
    interactions.append([col, inter_col])

index = 1
for inter in interactions:
    df_temp['inter_sum_' + str(index)] = df_temp[inter[0]] + df_temp[inter[1]]
    df_temp['inter_sub_' + str(index)] = df_temp[inter[0]] - df_temp[inter[1]]
    df_temp['inter_mult_' + str(index)] = df_temp[inter[0]] * df_temp[inter[1]]
    df_temp['inter_div_' + str(index)] = df_temp[inter[0]] / df_temp[inter[1]]
    index += 1

print(df_temp.shape)
print('all done...')
