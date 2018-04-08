#-*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
def get_score(predict,label):
    print "R2: ",r2_score(label,predict)
    print "MSE: ",mean_absolute_error(label,predict)
def train_test(model,x_train,x_test,y_train,y_test):
    prediction_train = model.predict(x_train)
    print model
    get_score(prediction_train,y_train)
    prediction_test = model.predict(x_test)
    get_score(prediction_test,y_test)

train_df = pd.read_csv(r"C:\Users\hanghang\Desktop\House Prices Advanced Regression Techniques\train.csv")
test_df = pd.read_csv(r"C:\Users\hanghang\Desktop\House Prices Advanced Regression Techniques\test.csv")
# print train_df
# print test_df
# print train_df.head()
# print train_df.tail()
# 空的字段和NA的字段都认为是True
# print train_df.isnull()
# print test_df.isnull()
NUL = pd.concat([train_df.isnull().sum(),test_df.isnull().sum()],axis=1)
# print NUL[NUL.sum(axis=1)>0]

# pop和drop('SalePrice',axis=1,inplace=True) 但是pop只能传一个列标
train_labels = train_df.pop('SalePrice')


features = pd.concat([train_df,test_df],keys=['train','test'])
# print features
features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
              axis=1, inplace=True)
# print type(features)
features['MSSubClass'] = features['MSSubClass'].astype(str)
# mode是众数
features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())

features['Alley'] = features['Alley'].fillna('NOACCESS')
# print features['Alley']
features['OverallCond'] = features['OverallCond'].astype(str)

features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('NoBSMT')

features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)

features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])

features['KitchenAbvGr'] = features['KitchenAbvGr'].astype(str)

features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])

features['FireplaceQu'] = features['FireplaceQu'].fillna('NoFP')

for col in ('GarageType', 'GarageFinish', 'GarageQual'):
    features[col] = features[col].fillna('NoGRG')

features['GarageCars'] = features['GarageCars'].fillna(0.0)

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

# 其他列累加和
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

# 归一化
# features['LotFrontage'] =  (features['LotFrontage'] - features['LotFrontage'].mean())/features['LotFrontage'].std()

# 给列标添加前缀
# print features.add_prefix('ddd')

# 房屋的价格
# print train_df
ax = sns.distplot(train_labels)
# plt.show(ax)

train_labels = np.log(train_labels)

ax = sns.distplot(train_labels)
# plt.show(ax)

numeric_features = features.loc[:,['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]
numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()
ax = sns.pairplot(numeric_features_standardized)
# plt.show(ax)

conditions = set([x for x in features['Condition1']] + [x for x in features['Condition2']])
# print conditions
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(conditions))),
                       index=features.index, columns=conditions)
for i, cond in enumerate(zip(features['Condition1'], features['Condition2'])):
    dummies.ix[i, cond] = 1
# print dummies
features = pd.concat([features, dummies.add_prefix('Condition_')], axis=1)
features.drop(['Condition1', 'Condition2'], axis=1, inplace=True)

exteriors = set([x for x in features['Exterior1st']] + [x for x in features['Exterior2nd']])
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(exteriors))),
                       index=features.index, columns=exteriors)
for i, ext in enumerate(zip(features['Exterior1st'], features['Exterior2nd'])):
    dummies.ix[i, ext] = 1
features = pd.concat([features, dummies.add_prefix('Exterior_')], axis=1)
features.drop(['Exterior1st', 'Exterior2nd', 'Exterior_nan'], axis=1, inplace=True)

# print features.dtypes
for col in features.dtypes[features.dtypes == 'object'].index:
    print col
    for_dummy = features.pop(col)
    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)


features_standardized = features.copy()
# 有列标一样的就替换
features_standardized.update(numeric_features_standardized)

train_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

train_features_st = features_standardized.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features_st = features_standardized.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

train_features_st, train_features, train_labels = shuffle(train_features_st, train_features, train_labels, random_state = 5)

x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_features_st, train_labels, test_size=0.1, random_state=200)

ENSTest = ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(x_train_st, y_train_st)
# train_test(ENSTest, x_train_st, x_test_st, y_train_st, y_test_st)


GBest = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)
# train_test(GBest, x_train, x_test, y_train, y_test)
#
# scores = cross_val_score(GBest, train_features_st, train_labels, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

GB_model = GBest.fit(train_features, train_labels)
ENST_model = ENSTest.fit(train_features_st, train_labels)

Final_labels = (np.exp(GB_model.predict(test_features)) + np.exp(ENST_model.predict(test_features_st))) / 2
pd.DataFrame({'Id': test_df.Id, 'SalePrice': Final_labels}).to_csv(r'C:\Users\hanghang\Desktop\House Prices Advanced Regression Techniques\FIN_SalePrice.csv', index =False)






