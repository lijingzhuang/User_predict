#!/usr/bin/env python
# coding: utf-8


from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from functionUtils import *
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook, tnrange
from nltk.corpus import stopwords
from scipy.stats import ks_2samp
from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import pandas as pd
import numpy as np
import warnings
import datetime
import gc

DATA_PATH = '../../input/'
warnings.filterwarnings("ignore")
# get_ipython().run_line_magic('matplotlib', 'inline')


# #### 构建Card-MerchantCategory矩阵
print('Card-MerchantCategory...')

usecols = ['card_id', 'merchant_category_id', 'purchase_amount']
df_transactions = pd.read_csv('tmp02_df_transactions.csv', usecols=usecols)
df_uid = pd.read_csv('tmp03_df_data.csv', usecols=['card_id'])
df_transactions = reduce_mem_usage(df_transactions)


def getCardMerCateSumFeatures(df_uid, df_feature, group='merchant_category_id', fea='purchase_amount'):
    df_purchase = df_feature.groupby(['card_id', 'merchant_category_id'])['purchase_amount'].sum().reset_index()
    df_purchase.rename(columns={'purchase_amount': 'purchase_amount_sum_merchant'}, inplace=True)
    df_temp = df_purchase.pivot(index='card_id', columns=group, values='purchase_amount_sum_merchant')
    df_temp.columns.name = None
    cols = []
    for col in df_temp.columns:
        cols.append(np.str(col) + '_purcahse_sum')
    df_temp.columns = cols
    df_temp.reset_index(inplace=True)
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')
    df_uid.fillna(0, inplace=True)
    return df_uid


def getCardMerCateCountFeatures(df_uid, df_feature, group='merchant_category_id', fea='purchase_amount',
                                name='card-merCategory'):
    df_purchase = df_feature.groupby(['card_id', 'merchant_category_id'])['purchase_amount'].count().reset_index()
    df_purchase.rename(columns={'purchase_amount': 'purchase_counts_merchant'}, inplace=True)
    df_temp = df_purchase.pivot(index='card_id', columns=group, values='purchase_counts_merchant')
    df_temp.columns.name = None
    cols = []
    for col in df_temp.columns:
        cols.append(np.str(col) + '_purchase_counts')
    df_temp.columns = cols
    df_temp.reset_index(inplace=True)
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')
    df_uid.fillna(0, inplace=True)
    return df_uid


df_cardMerCateSum = getCardMerCateSumFeatures(df_uid, df_transactions)
df_cardMerCateCount = getCardMerCateCountFeatures(df_uid, df_transactions)

df_cardMerCateSum = downCast_dtype(df_cardMerCateSum)
df_cardMerCateCount = downCast_dtype(df_cardMerCateCount)

df_cardMerCateCount.replace([np.inf, -np.inf], 0, inplace=True)
df_cardMerCateSum.replace([np.inf, -np.inf], 0, inplace=True)
df_cardMerCateCount.fillna(0, inplace=True)
df_cardMerCateSum.fillna(0, inplace=True)

##### 矩阵分解SVD或者MF

nmf = NMF(n_components=10, init='random', random_state=42)
W = nmf.fit_transform(df_cardMerCateCount.iloc[0:, 1:].values)
cols = []
for col in range(10):
    cols.append('nmf_%s_count' % col)
df_nmf_features = pd.DataFrame(data=W, columns=cols)
df_nmf_features['card_id'] = df_cardMerCateCount['card_id'].values

svd = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=7, random_state=42)
W = svd.fit_transform(df_cardMerCateSum.iloc[0:, 1:].values)
cols = []
for col in range(10):
    cols.append('svd_%s_sum' % col)
df = pd.DataFrame(data=W, columns=cols)
df['card_id'] = df_cardMerCateSum['card_id'].values

df_nmf_features = df_nmf_features.merge(df, on='card_id', how='left')
print('save to csv: tmp05_df_nmf_card_merCate_features.csv')
df_nmf_features.to_csv('tmp05_df_nmf_card_merCate_features.csv', index=False)

# #### 构建Card-City矩阵
print('Card-City...')

usecols = ['card_id', 'city_id', 'purchase_amount']
df_transactions = pd.read_csv('tmp02_df_transactions.csv', usecols=usecols)
df_uid = pd.read_csv('tmp03_df_data.csv', usecols=['card_id'])
df_transactions = reduce_mem_usage(df_transactions)


def getCardCitySumFeatures(df_uid, df_feature, group='city_id', fea='purchase_amount'):
    df_purchase = df_feature.groupby(['card_id', 'city_id'])['purchase_amount'].sum().reset_index()
    df_purchase.rename(columns={'purchase_amount': 'purchase_amount_sum_city'}, inplace=True)
    df_temp = df_purchase.pivot(index='card_id', columns=group, values='purchase_amount_sum_city')
    df_temp.columns.name = None
    cols = []
    for col in df_temp.columns:
        cols.append(np.str(col) + '_purcahse_sum')
    df_temp.columns = cols
    df_temp.reset_index(inplace=True)
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')
    df_uid.fillna(0, inplace=True)
    return df_uid


def getCardCityCountFeatures(df_uid, df_feature, group='city_id', fea='purchase_amount', name='card-merCategory'):
    df_purchase = df_feature.groupby(['card_id', 'city_id'])['purchase_amount'].count().reset_index()
    df_purchase.rename(columns={'purchase_amount': 'purchase_counts_city'}, inplace=True)
    df_temp = df_purchase.pivot(index='card_id', columns=group, values='purchase_counts_city')
    df_temp.columns.name = None
    cols = []
    for col in df_temp.columns:
        cols.append(np.str(col) + '_purchase_counts')
    df_temp.columns = cols
    df_temp.reset_index(inplace=True)
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')
    df_uid.fillna(0, inplace=True)
    return df_uid


df_cardCitySum = getCardCitySumFeatures(df_uid, df_transactions)
df_cardCityCount = getCardCityCountFeatures(df_uid, df_transactions)
df_cardCitySum = reduce_mem_usage(df_cardCitySum)
df_cardCityCount = reduce_mem_usage(df_cardCityCount)

nmf = NMF(n_components=10, init='random', random_state=42)
W = nmf.fit_transform(df_cardCityCount.iloc[0:, 1:].values)
cols = []
for col in range(10):
    cols.append('nmf_city_%s_count' % col)
df_nmf_features = pd.DataFrame(data=W, columns=cols)
df_nmf_features['card_id'] = df_cardCityCount['card_id'].values

svd = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=7, random_state=42)
W = svd.fit_transform(df_cardCitySum.iloc[0:, 1:].values)
cols = []
for col in range(10):
    cols.append('svd_city_%s_sum' % col)
df = pd.DataFrame(data=W, columns=cols)
df['card_id'] = df_cardCitySum['card_id'].values

df_nmf_features = df_nmf_features.merge(df, on='card_id', how='left')
print('save to csv: tmp05_df_nmf_card_city_features.csv')
df_nmf_features.to_csv('tmp05_df_nmf_card_city_features.csv', index=False)

'''
# #### Card-Merchant 词向量 MemoryError!!!
print('Card-Merchant wordvec...')

usecols = ['card_id', 'city_id', 'merchant_id', 'purchase_amount']
df_transactions = pd.read_csv('tmp02_df_transactions.csv', usecols=usecols)
df_uid = pd.read_csv('tmp03_df_data.csv', usecols=['card_id'])
df_transactions = reduce_mem_usage(df_transactions)





# 获取词向量
def getWord2Vec(df_fea, value, embedding_size=10, name='cardId'):
    def getSeq(x):
        name = ''
        return name.join([x for x in x.split('_')])

    tmp_corpus = df_fea[value].map(lambda x: getSeq(x))
    corpus = []
    for i in range(len(tmp_corpus)):
        words = []
        for word in tmp_corpus[i]:
            words.append(word)
        corpus.append(words)
    model = Word2Vec(corpus, size=embedding_size, window=3, min_count=1, workers=4)
    df_vec = pd.DataFrame(data=df_fea['card_id'].astype(np.str).values, columns=['card_id'])
    nwords = len(tmp_corpus[0])
    seq = nwords * embedding_size
    vec_feas = np.zeros((len(tmp_corpus), seq))
    colsnames = []
    for i in range(seq):
        colsnames.append(name + np.str(i) + '_vec')
    for i in range(len(tmp_corpus)):
        words = []
        for word in tmp_corpus[i]:
            words.append(word)
        vec_feas[i, :] = model.wv[words].reshape(1, -1)
    df = pd.DataFrame(data=vec_feas, columns=colsnames)
    df_vec = pd.concat([df_vec, df], axis=1)
    return df_vec


df_transactions['card_merchant'] = df_transactions['card_id'].astype(np.str) + '_' + df_transactions[
    'merchant_id'].astype(np.str)
df_card_merchant = getWord2Vec(df_transactions, 'card_merchant', name='card_merchant')
df_card_merchant = df_card_merchant.groupby('card_id').mean().reset_index()
df_card_merchant = reduce_mem_usage(df_card_merchant)
print('save to csv: tmp05_df_card_merchant_vec.csv')
df_card_merchant.to_csv('tmp05_df_card_merchant_vec.csv', index=False)

df_cardid = getWord2Vec(df_uid, 'card_id')

# df_cardid = df_vec.groupby('card_id').mean().reset_index()
df_cardid = reduce_mem_usage(df_cardid)

dropCols = []
tr_features = [f for f in df_cardid.columns if df_cardid[f].dtype != 'object']
for col in tr_features:
    if df_cardid[col].std() < 0.01:
        dropCols.append(col)
df_cardid.drop(columns=dropCols, inplace=True)
print('save to csv: tmp05_df_cardid_vec.csv')
df_cardid.to_csv('tmp05_df_cardid_vec.csv', index=False)
'''

# #### Card-Merchant statics（强特)
print('Card-Merchant statics...')

usecols = ['card_id', 'merchant_category_id', 'merchant_id', 'month_lag', 'purchase_amount', 'purchase_date']
df_transactions = pd.read_csv('tmp02_df_transactions.csv', usecols=usecols)
df_uid = pd.read_csv('tmp03_df_data.csv', usecols=['card_id'])
df_transactions = reduce_mem_usage(df_transactions)
df_transactions.sort_values(by=['card_id', 'purchase_date'], ascending=True, inplace=True)

df_hist_transactions = df_transactions[df_transactions.month_lag <= 0]
df_new_transactions = df_transactions[df_transactions.month_lag > 0]

for flag, df_features in zip(['hist', 'new'], [df_hist_transactions, df_new_transactions]):
    print('.........%s.........' % flag)
    df_temp = df_features.groupby(['card_id', 'merchant_id'])['purchase_amount'].mean().reset_index()
    df_uid = getMaxStaticsFeatures(df_uid, df_temp, 'card_id', 'purchase_amount',
                                   name='%s_card_merchant_mean_max' % flag)
    df_uid = getStdStaticsFeatures(df_uid, df_temp, 'card_id', 'purchase_amount',
                                   name='%s_card_merchant_mean_std' % flag)
    df_uid = getMedianStaticsFeatures(df_uid, df_temp, 'card_id', 'purchase_amount',
                                      name='%s_card_merchant_mean_median' % flag)

    df_temp = df_features.groupby(['card_id', 'merchant_category_id'])['purchase_amount'].mean().reset_index()
    df_uid = getMaxStaticsFeatures(df_uid, df_temp, 'card_id', 'purchase_amount',
                                   name='%s_card_merchant_category_mean_max' % flag)
    df_uid = getStdStaticsFeatures(df_uid, df_temp, 'card_id', 'purchase_amount',
                                   name='%s_card_merchant_category_mean_std' % flag)
    df_uid = getMedianStaticsFeatures(df_uid, df_temp, 'card_id', 'purchase_amount',
                                      name='%s_card_merchant_category__mean_median' % flag)

    df_temp = df_features.groupby(['card_id', 'merchant_category_id'])['purchase_amount'].std().reset_index()
    df_uid = getMaxStaticsFeatures(df_uid, df_temp, 'card_id', 'purchase_amount',
                                   name='%s_card_merchant_std_max' % flag)
    df_uid = getMedianStaticsFeatures(df_uid, df_temp, 'card_id', 'purchase_amount',
                                      name='%s_card_merchant_category__std_median' % flag)
    # card-merchant ratio
    df_temp = df_features.groupby(['card_id', 'merchant_category_id'])['purchase_amount'].max().reset_index().rename(
        columns={'purchase_amount': '%s_merchant_cate_purchase_max' % flag})
    df_features = df_features.merge(df_temp, on=['card_id', 'merchant_category_id'], how='left')
    df_features['%s_purchase/purchaseMax' % flag] = df_features['purchase_amount'] / df_features[
        '%s_merchant_cate_purchase_max' % flag]
    df_uid = getMaxStaticsFeatures(df_uid, df_features, 'card_id', '%s_purchase/purchaseMax' % flag,
                                   name='%s_purchase/purchaseMax_max' % flag)
    df_uid = getStdStaticsFeatures(df_uid, df_features, 'card_id', '%s_purchase/purchaseMax' % flag,
                                   name='%s_purchase/purchaseMax_std' % flag)
    df_uid = getMedianStaticsFeatures(df_uid, df_features, 'card_id', '%s_purchase/purchaseMax' % flag,
                                      name='%s_purchase/purchaseMax_median' % flag)

    df_temp = df_features.groupby(['card_id', 'merchant_category_id'])['purchase_amount'].mean().reset_index().rename(
        columns={'purchase_amount': '%s_merchant_cate_purchase_mean' % flag})
    df_features = df_features.merge(df_temp, on=['card_id', 'merchant_category_id'], how='left')
    df_features['%s_purchase/purchaseMean' % flag] = df_features['purchase_amount'] / df_features[
        '%s_merchant_cate_purchase_mean' % flag]
    df_uid = getMaxStaticsFeatures(df_uid, df_features, 'card_id', '%s_purchase/purchaseMean' % flag,
                                   name='%s_purchase/purchaseMean_max' % flag)
    df_uid = getStdStaticsFeatures(df_uid, df_features, 'card_id', '%s_purchase/purchaseMean' % flag,
                                   name='%s_purchase/purchaseMean_std' % flag)
    df_uid = getMedianStaticsFeatures(df_uid, df_features, 'card_id', '%s_purchase/purchaseMean' % flag,
                                      name='%s_purchase/purchaseMean_median' % flag)

    df_features['%s_purchase_merchant_shift' % flag] = df_features.groupby(['card_id', 'merchant_id'])[
        'purchase_amount'].apply(lambda series: series.shift(1)).values
    df_features['%s_purchase_merchant_shift' % flag].fillna(0, inplace=True)
    df_features['%s_purchase_merchant_shift_ratio' % flag] = df_features['%s_purchase_merchant_shift' % flag] / \
                                                             df_features['purchase_amount']
    df_temp = df_features.groupby(['card_id'])['%s_purchase_merchant_shift_ratio' % flag].max().reset_index()
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')

    df_features['%s_purchase_merchantCate_shift' % flag] = df_features.groupby(['card_id', 'merchant_category_id'])[
        'purchase_amount'].apply(lambda series: series.shift(1)).values
    df_features['%s_purchase_merchantCate_shift' % flag].fillna(0, inplace=True)
    df_features['%s_purchase_merchantCate_shift_ratio' % flag] = df_features['%s_purchase_merchantCate_shift' % flag] / \
                                                                 df_features['purchase_amount']
    df_temp = df_features.groupby(['card_id'])['%s_purchase_merchantCate_shift_ratio' % flag].max().reset_index()
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')
    # purchase_amount diff
    df_features['%s_card_merchant_category_purchase_diff' % flag] = \
        df_features.groupby(['card_id', 'merchant_category_id'])['purchase_amount'].apply(
            lambda series: series.diff(1)).values
    df_features['%s_card_merchant_category_purchase_diff' % flag].fillna(0, inplace=True)
    df_temp = df_features.groupby(['card_id'])[
        '%s_card_merchant_category_purchase_diff' % flag].max().reset_index().rename(columns={
        '%s_card_merchant_category_purchase_diff' % flag: '%s_card_merchant_category_purchase_diff_max' % flag})
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')
    df_temp = df_features.groupby(['card_id'])[
        '%s_card_merchant_category_purchase_diff' % flag].median().reset_index().rename(columns={
        '%s_card_merchant_category_purchase_diff' % flag: '%s_card_merchant_category_purchase_diff_median' % flag})
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')

    df_features['%s_card_merchant_purchase_diff' % flag] = df_features.groupby(['card_id', 'merchant_id'])[
        'purchase_amount'].apply(lambda series: series.diff(1)).values
    df_features['%s_card_merchant_purchase_diff' % flag].fillna(0, inplace=True)
    df_temp = df_features.groupby(['card_id'])['%s_card_merchant_purchase_diff' % flag].max().reset_index().rename(
        columns={'%s_card_smerchant_purchase_diff' % flag: '%s_card_merchant_purchase_diff_max' % flag})
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')
    df_temp = df_features.groupby(['card_id'])['%s_card_merchant_purchase_diff' % flag].median().reset_index().rename(
        columns={'%s_card_merchant_purchase_diff' % flag: '%s_card_merchant_purchase_diff_median' % flag})
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')

gc.collect()
df_uid = reduce_mem_usage(df_uid)
df_uid.fillna(0, inplace=True)
dropCols = []
tr_features = [f for f in df_uid.columns if df_uid[f].dtype != 'object']
for col in tr_features:
    if df_uid[col].std() < 0.01:
        dropCols.append(col)
df_uid.drop(columns=dropCols, inplace=True)
print('save to csv: tmp05_df_card_merchant_statics.csv')
df_uid.to_csv('tmp05_df_card_merchant_statics.csv', index=False)

# #### Card-City statics
print('Card-City statics...')

usecols = ['card_id', 'category_2', 'subsector_id', 'state_id', 'month_lag', 'purchase_amount', 'purchase_date']
df_transactions = pd.read_csv('tmp02_df_transactions.csv', usecols=usecols)
df_uid = pd.read_csv('tmp03_df_data.csv', usecols=['card_id'])
df_transactions = reduce_mem_usage(df_transactions)
df_transactions.sort_values(by=['card_id', 'purchase_date'], ascending=True, inplace=True)

df_hist_transactions = df_transactions[df_transactions.month_lag <= 0]
df_new_transactions = df_transactions[df_transactions.month_lag > 0]

for flag, df_features in zip(['hist', 'new'], [df_hist_transactions, df_new_transactions]):
    print('.........%s.........' % flag)
    df_temp = df_hist_transactions.groupby(['card_id', 'category_2'])['purchase_amount'].mean().reset_index().rename(
        columns={'purchase_amount': '%s_card_c2_mean' % flag})
    df_features = df_features.merge(df_temp, on=['card_id', 'category_2'], how='left')
    df_uid = getMaxStaticsFeatures(df_uid, df_features, 'card_id', '%s_card_c2_mean' % flag,
                                   '%s_card_c2_mean_max' % flag)
    df_uid = getMedianStaticsFeatures(df_uid, df_features, 'card_id', '%s_card_c2_mean' % flag,
                                      '%s_card_c2_mean_median' % flag)

    df_temp = df_hist_transactions.groupby(['card_id', 'state_id'])['purchase_amount'].mean().reset_index().rename(
        columns={'purchase_amount': '%s_card_state_mean' % flag})
    df_features = df_features.merge(df_temp, on=['card_id', 'state_id'], how='left')
    df_uid = getMaxStaticsFeatures(df_uid, df_features, 'card_id', '%s_card_state_mean' % flag,
                                   '%s_card_state_mean_max' % flag)
    df_uid = getMedianStaticsFeatures(df_uid, df_features, 'card_id', '%s_card_state_mean' % flag,
                                      '%s_card_state_mean_median' % flag)

    df_temp = df_hist_transactions.groupby(['card_id', 'subsector_id'])['purchase_amount'].mean().reset_index().rename(
        columns={'purchase_amount': '%s_card_subsector_mean' % flag})
    df_features = df_features.merge(df_temp, on=['card_id', 'subsector_id'], how='left')
    df_uid = getMaxStaticsFeatures(df_uid, df_features, 'card_id', '%s_card_subsector_mean' % flag,
                                   '%s_card_subsector_mean_max' % flag)
    df_uid = getMedianStaticsFeatures(df_uid, df_features, 'card_id', '%s_card_subsector_mean' % flag,
                                      '%s_card_subsector_mean_median' % flag)

    # purchase shift比例
    df_features['%s_purchase_c2_shift' % flag] = df_features.groupby(['card_id', 'category_2'])[
        'purchase_amount'].apply(lambda series: series.shift(1)).values
    df_features['%s_purchase_c2_shift' % flag].fillna(0, inplace=True)
    df_features['%s_purchase_c2_shift_ratio' % flag] = df_features['%s_purchase_c2_shift' % flag] / df_features[
        'purchase_amount']
    df_temp = df_features.groupby(['card_id'])['%s_purchase_c2_shift_ratio' % flag].max().reset_index()
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')

    df_features['%s_purchase_subsector_shift' % flag] = df_features.groupby(['card_id', 'subsector_id'])[
        'purchase_amount'].apply(lambda series: series.shift(1)).values
    df_features['%s_purchase_subsector_shift' % flag].fillna(0, inplace=True)
    df_features['%s_purchase_subsector_shift_ratio' % flag] = df_features['%s_purchase_subsector_shift' % flag] / \
                                                              df_features['purchase_amount']
    df_temp = df_features.groupby(['card_id'])['%s_purchase_subsector_shift_ratio' % flag].max().reset_index()
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')

    df_features['%s_card_c2_purchase_diff' % flag] = df_features.groupby(['card_id', 'category_2'])[
        'purchase_amount'].apply(lambda series: series.diff(1)).values
    df_features['%s_card_c2_purchase_diff' % flag].fillna(0, inplace=True)
    df_temp = df_features.groupby(['card_id'])['%s_card_c2_purchase_diff' % flag].max().reset_index().rename(
        columns={'%s_card_c2_purchase_diff' % flag: '%s_card_c2_purchase_diff_max' % flag})
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')
    df_temp = df_features.groupby(['card_id'])['%s_card_c2_purchase_diff' % flag].median().reset_index().rename(
        columns={'%s_card_c2_purchase_diff' % flag: '%s_card_c2_purchase_diff_median' % flag})
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')

    df_features['%s_card_subsector_purchase_diff' % flag] = df_features.groupby(['card_id', 'subsector_id'])[
        'purchase_amount'].apply(lambda series: series.diff(1)).values
    df_features['%s_card_subsector_purchase_diff' % flag].fillna(0, inplace=True)
    df_temp = df_features.groupby(['card_id'])['%s_card_subsector_purchase_diff' % flag].max().reset_index().rename(
        columns={'%s_card_subsector_purchase_diff' % flag: '%s_card_subsector_purchase_diff_max' % flag})
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')
    df_temp = df_features.groupby(['card_id'])['%s_card_subsector_purchase_diff' % flag].median().reset_index().rename(
        columns={'%s_card_subsector_purchase_diff' % flag: '%s_card_subsector_purchase_diff_median' % flag})
    df_uid = df_uid.merge(df_temp, on='card_id', how='left')

gc.collect()
df_uid = reduce_mem_usage(df_uid)
df_uid.fillna(0, inplace=True)
print('save to csv: tmp05_df_card_city_statics.csv')
df_uid.to_csv('tmp05_df_card_city_statics.csv', index=False)

# #### 持卡人对商家的访问序列进行embedding 刻画持卡人的行为向量
print('持卡人对商家的访问序列进行embedding...')

usecols = ['card_id', 'purchase_date', 'merchant_id', 'merchant_category_id', 'city_id', 'state_id', 'category_1',
           'category_2', 'category_3']
df_transactions = pd.read_csv('tmp02_df_transactions.csv', usecols=usecols)
df_uid = pd.read_csv('tmp03_df_data.csv', usecols=['card_id'])
df_transactions.sort_values(by=['card_id', 'purchase_date'], ascending=True, inplace=True)


# 获取词向量
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


def getSequence(series):
    return list(series.values)


cateCols = ['merchant_id', 'merchant_category_id', 'city_id', 'state_id', 'category_1', 'category_2', 'category_3']
for col in cateCols:
    print('........%s........' % col)
    df_transactions[col] = df_transactions.astype(np.str)
    df_temp = df_transactions.groupby('card_id')[col].apply(lambda series: getSequence(series)).reset_index()
    df_temp.rename(columns={col: '%s_sequences' % col}, inplace=True)
    df_vec = getWord2Vec(df_temp, fea='%s_sequences' % col, name='card_%s' % col)
    df_uid = df_uid.merge(df_vec, on='card_id', how='left')
del df_temp, df_vec
gc.collect()
print('save to csv: tmp05_df_card_merchant_vec.csv')
df_uid.to_csv('tmp05_df_card_merchant_vec.csv', index=False)

usecols = ['card_id', 'purchase_date', 'authorized_flag', 'installments', 'city_C1', 'C2_state',
           'C2_state_subsector', 'subsector_city', 'auth_C3', 'day_gap']
df_transactions = pd.read_csv('tmp02_df_transactions.csv', usecols=usecols)
df_uid = pd.read_csv('tmp03_df_data.csv', usecols=['card_id'])
df_transactions.sort_values(by=['card_id', 'purchase_date'], ascending=True, inplace=True)
df_uid = reduce_mem_usage(df_uid)
df_transactions = reduce_mem_usage(df_transactions)


# 获取词向量
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


def getSequence(series):
    return list(series.values)


cateCols = ['authorized_flag', 'installments', 'city_C1', 'C2_state', 'C2_state_subsector',
            'subsector_city', 'auth_C3', 'day_gap']
for col in cateCols:
    print('........%s........' % col)
    df_transactions[col] = df_transactions.astype(np.str)
    df_temp = df_transactions.groupby('card_id')[col].apply(lambda series: getSequence(series)).reset_index()
    df_temp.rename(columns={col: '%s_sequences' % col}, inplace=True)
    df_vec = getWord2Vec(df_temp, fea='%s_sequences' % col, name='_%s' % col)
    df_uid = df_uid.merge(df_vec, on='card_id', how='left')
del df_temp, df_vec
gc.collect()
print('save to csv: tmp05_df_card_merchant_vec1.csv')
df_uid.to_csv('tmp05_df_card_merchant_vec1.csv', index=False)

df_uid.head()

df_train = pd.read_csv(DATA_PATH + 'train.csv')
df_test = pd.read_csv(DATA_PATH + 'test.csv')
df_data = pd.concat([df_train, df_test])


# 获取词向量
def getWord2Vec(df_temp=None, fea=None, embedding_size=3, name=None):
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


df_data['f1%2'] = df_data['feature_1'] % 2
df_data['f1-f3'] = df_data['f1%2'] - df_data['feature_3']

df_data['f1_f2_f3'] = df_data['feature_1'].astype(np.str) + '_' + df_data['feature_2'].astype(np.str) + '_' + df_data[
    'feature_3'].astype(np.str)

df_data['f1_f2_f3'] = df_data['f1_f2_f3'].apply(lambda x: x.split('_'))

df_data['f1_f2'] = df_data['feature_1'].astype(np.str) + '_' + df_data['feature_2'].astype(np.str)
df_data['f1_f2'] = df_data['f1_f2'].apply(lambda x: x.split('_'))

df_data['f1_f3'] = df_data['feature_1'].astype(np.str) + '_' + df_data['feature_3'].astype(np.str)
df_data['f1_f3'] = df_data['f1_f3'].apply(lambda x: x.split('_'))

df_data['f2_f3'] = df_data['feature_2'].astype(np.str) + '_' + df_data['feature_3'].astype(np.str)
df_data['f2_f3'] = df_data['f2_f3'].apply(lambda x: x.split('_'))

df_uid = getWord2Vec(df_data, fea='f1_f2_f3', name='feature123')

df_vec = getWord2Vec(df_data, fea='f1_f2', name='feature12')
df_uid = df_uid.merge(df_vec, on='card_id', how='left')

df_vec = getWord2Vec(df_data, fea='f2_f3', name='feature23')
df_uid = df_uid.merge(df_vec, on='card_id', how='left')

df_vec = getWord2Vec(df_data, fea='f1_f3', name='feature13')
df_uid = df_uid.merge(df_vec, on='card_id', how='left')

df_data['f1_f2_f3'] = df_data['feature_1'].astype(np.str) + '_' + df_data['feature_2'].astype(np.str) + '_' + df_data[
    'feature_3'].astype(np.str)
df_data['f1_f2'] = df_data['feature_1'].astype(np.str) + '_' + df_data['feature_2'].astype(np.str)
df_data['f1_f3'] = df_data['feature_1'].astype(np.str) + '_' + df_data['feature_3'].astype(np.str)
df_data['f2_f3'] = df_data['feature_2'].astype(np.str) + '_' + df_data['feature_3'].astype(np.str)

df_data = label_encoding(df_data, encodCols=['f1_f2_f3', 'f1_f2', 'f1_f3', 'f2_f3', 'f1%2', 'f1-f3'])

df_uid = df_uid.merge(df_data[['card_id', 'f1_f2_f3', 'f1_f2', 'f1_f3', 'f2_f3', 'f1%2', 'f1-f3']], on='card_id',
                      how='left')
df_uid.head()

print('save to csv: tmp05_df_f1_f2_f3_vec.csv')
df_uid.to_csv('tmp05_df_f1_f2_f3_vec.csv', index=False)
