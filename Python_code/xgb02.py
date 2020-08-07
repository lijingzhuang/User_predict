#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 500)

# In[2]:


print('read csv...')
df_train = pd.read_csv('../input/df_feats_train_0512.csv')
df_test = pd.read_csv('../input/df_feats_test_0512.csv')

df_features = pd.read_csv('../input/df_feats_features_0512.csv', names=['feature'])
tr_features = df_features['feature'].values.tolist()

target_out = [1 if i < -30 else 0 for i in df_train['target']]
print(df_train.shape)
print(df_test.shape)
print(df_features.shape)

# In[4]:


target = df_train['target']

# In[ ]:


# 构造XGBRegressor模型
model = xgb.XGBRegressor(
    max_depth=9,
    learning_rate=0.01,
    n_estimators=1000,
    silent=True,
    objective='reg:linear',
    eval_metric='rmse',
    nthread=-1,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.99,
    colsample_bytree=0.7,
    colsample_bylevel=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    seed=1440,
    missing=None,
    verbose=True
)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()
# plt.figure(figsize=(20, 6))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train[tr_features], target_out)):
    print("fold {}".format(fold_ + 1))

    clf = model.fit(df_train.iloc[trn_idx][tr_features], target.iloc[trn_idx],
                    eval_set=[(df_train.iloc[trn_idx][tr_features], target.iloc[trn_idx]),
                              (df_train.iloc[val_idx][tr_features], target.iloc[val_idx])],
                    early_stopping_rounds=100)
    print('save model...')
    joblib.dump(clf, './models/xgb02_' + str(fold_ + 1) + '.model')
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][tr_features])

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = tr_features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(df_test[tr_features]) / folds.n_splits

    # num_realy = len(model.evals_result_['validation_0']['rmse'])
    # plt.subplot(1, folds.n_splits, fold_ + 1)
    # plt.plot(range(num_realy), model.evals_result_['validation_0']['rmse'], '-')
    # plt.plot(range(num_realy), model.evals_result_['validation_1']['rmse'], '--')
    # plt.legend(['training', 'valid'])

local_rmse = np.sqrt(mean_squared_error(oof, target))
print('local rmse:', local_rmse)

# plt.savefig('result/xgb02_loss_' + str(local_rmse) + '.png')
# plt.show()

# In[ ]:


# In[ ]:


sub_df = pd.DataFrame({"card_id": df_test["card_id"].values})
sub_df["target"] = predictions
sub_df.to_csv("result/xgb02_sub_" + str(local_rmse) + "_test.csv", index=False)

sub_df_train = pd.DataFrame({"card_id": df_train["card_id"].values})
sub_df_train["target"] = oof
sub_df_train.to_csv("result/xgb02_sub_" + str(local_rmse) + "_train.csv", index=False)

# In[ ]:


# cols = (feature_importance_df[["Feature", "importance"]]
#         .groupby("Feature")
#         .mean()
#         .sort_values(by="importance", ascending=False)[:1000].index)
#
# best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

# plt.figure(figsize=(14, 25))
# sns.barplot(x="importance",
#             y="Feature",
#             data=best_features.sort_values(by="importance",
#                                            ascending=False))
# plt.title('XGBoost Features (avg over folds)')
# plt.tight_layout()
# plt.savefig('result/xgb02_impt_' + str(local_rmse) + '.png')

# In[ ]:
