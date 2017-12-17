#-*- coding=utf-8 -*-
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
path=r"C:\Users\hanghang\Desktop\regression.train"
print("load data")
df_train=pd.read_csv(path,header=None,sep='\t')
df_test=pd.read_csv(path,header=None,sep='\t')
y_train = df_train[0].values
y_test = df_test[0].values
X_train = df_train.drop(0, axis=1).values
X_test = df_test.drop(0, axis=1).values
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
print('Save model...')
# save model to file
gbm.save_model(r"C:\Users\hanghang\Desktop\model_file\lightgbm.model")
print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print(y_pred)
print('The roc of prediction is:', roc_auc_score(y_test, y_pred) )
print('Dump model to JSON...')
# dump model to json (and save to file)
model_json = gbm.dump_model()
with open('lightgbm/model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)
print('Feature names:', gbm.feature_name())
print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))