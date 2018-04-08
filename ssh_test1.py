#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from xgboost.sklearn import  XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import time
import operator
from sklearn.model_selection import StratifiedKFold
google_df = pd.read_csv(r"C:\Users\hanghang\Desktop\ssh_test1\180111-google1.csv",header=None)
# print google_df[[0,1,2,3,4,5]]
# print google_df[29]
# 使小于等于0的数都为NAN
google_df = google_df[google_df > 0]
# .drop(labels, axis=0)  去掉行或者列
# 去掉包含缺失值的行：
google_df = google_df.dropna(how='any')
print google_df.shape[0]

#180111-youtube1
youtube_df = pd.read_csv(r"C:\Users\hanghang\Desktop\ssh_test1\180111-youtube1.csv",header=None)
youtube_df = youtube_df[youtube_df > 0]
youtube_df = youtube_df.dropna(how='any')
print youtube_df.shape[0]

facebook_df = pd.read_csv(r"C:\Users\hanghang\Desktop\ssh_test1\180112-facebook.csv",header=None)
facebook_df = facebook_df[facebook_df > 0]
# print facebook_df
facebook_df = facebook_df.dropna(how='any')
print facebook_df.shape[0]

twitter_df = pd.read_csv(r"C:\Users\hanghang\Desktop\ssh_test1\180112-twitter.csv",header=None)
twitter_df = twitter_df[twitter_df > 0]
# print twitter_df
twitter_df = twitter_df.dropna(how='any')
print twitter_df.shape[0]

login_df = pd.read_csv(r"C:\Users\hanghang\Desktop\ssh_test1\180113-login.csv",header=None)
login_df = login_df[login_df > 0]
# print login_df.shape[0]
login_df = login_df.dropna(how='any')
# print login_df.shape[0]

http_df = pd.concat([google_df,youtube_df,facebook_df,twitter_df],axis=0).reset_index()
print "................................................."
http_df = http_df.drop('index',axis=1)
# print http_df
# http_df = google_df.append(youtube_df).append(facebook_df).append(twitter_df)
# print http_df
http_df[30] = 1
# print http_df

login_df = login_df.reset_index()
# print login_df
login_df = login_df.drop('index',axis=1)
# print login_df
login_df[30] = 0

login_df.drop([0,1,2,3,4,5],axis=1,inplace=True)
http_df.drop([0,1,2,3,4,5],axis=1,inplace=True)



res_df = http_df.append(login_df).reset_index().drop('index',axis=1)
# print res_df
# 在合并的过程中行索引是可以有相同的，最后合并的结果也有行索引相同的样本
# test = http_df.iloc[0:2]
# test1 = http_df.iloc[0:2]
# test = test.append(test1)
# print test

x_col = [x for x in res_df.columns if x != 30]
X = res_df[x_col]
# print X
Y = res_df[30]
# print Y
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5,random_state=66)
start_time = time.time()
# 输出当前格式化的时间
print time.strftime('%Y-%m-%d',time.localtime(start_time))
# XGB原生API
params={
'booster':'gbtree',
'objective': 'binary:logistic', #二分类的问题
# 'num_class':2, # 类别数，与 multisoftmax 并用
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':5, # 构建树的深度，越大越容易过拟合
'lambda':1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':1,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, # 如同学习率
'seed':1000,
'nthread':4,# cpu 线程数
# 'eval_metric': 'auc'
}
# plst = list(params.items())
num_rounds = 100
# xgb_train = x_train.as_matrix()
# xgb_label = y_train.as_matrix()
train = xgb.DMatrix(x_train,y_train)
# print train
val = xgb.DMatrix(x_test,y_test)
# print x_train
test = xgb.DMatrix(x_test)
skf = StratifiedKFold(n_splits=5)
X = X.as_matrix()
Y = Y.as_matrix()
for train_index, test_index in skf.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    train = xgb.DMatrix(X_train,Y_train)
    test = xgb.DMatrix(X_test)
    model = xgb.train(params, train, num_rounds)
    importance = model.get_fscore()
    # 特征重要性
    print sorted(importance.items(),key=operator.itemgetter(1))
    # model.save_model(r'C:\Users\hanghang\Desktop\xgb.model')
    y_predict = model.predict(test,ntree_limit=model.best_ntree_limit)
    # print y_predict
    y_predict = (y_predict > 0.5)*1
    # print y_predict

    print 'XGB分类结果..............'
    print metrics.classification_report(Y_test,y_predict)
    print metrics.confusion_matrix(Y_test,y_predict)





# XGB的sklearn形式API
# xgb1 = XGBClassifier(
#     max_depth=3,
#     learning_rate=0.1,
#     n_estimators=100,
#     silent=False,
#     objective='binary:logistic',
#     booster='gbtree',
#     n_jobs=1,
#     nthread=4,
#     gamma=0,
#     min_child_weight=1,
#     max_delta_step=0,
#     subsample=1,
#     colsample_bytree=1,
#     colsample_bylevel=1,
#     reg_alpha=0,
#     reg_lambda=1,
#     scale_pos_weight=1,
#     base_score=0.5,
#     random_state=66,
#     seed=None,
#     missing=None，
# )
# xgb1.fit(x_train,y_train)
# y_predict = xgb1.predict(x_test)
# print 'XGB分类结果..............'
# print metrics.classification_report(y_test,y_predict)
# print metrics.confusion_matrix(y_test,y_predict)





# LR
LR = LogisticRegression()
LR.fit(x_train,y_train)
y_predict = LR.predict(x_test)
print 'LR分类结果..............'
print metrics.classification_report(y_test,y_predict)
print metrics.confusion_matrix(y_test,y_predict)

print '交叉验证结果..........'
lr_scores = cross_val_score(LR,X,Y,cv=5)
print lr_scores

#模型持久化
# joblib.dump(LR,r"C:\Users\hanghang\Desktop\ssh_test1\LR.model",compress=3)

# 载入模型
# LR1 = joblib.load(r"C:\Users\hanghang\Desktop\ssh_test1\LR.model")
# y_predict = LR1.predict(x_test)
# print metrics.classification_report(y_test,y_predict)
# print metrics.confusion_matrix(y_test,y_predict)

# CART决策树
DTC = DecisionTreeClassifier()
DTC.fit(x_train,y_train)
y_predict = DTC.predict(x_test)
print 'cart分类结果..............'
print metrics.classification_report(y_test,y_predict)
print metrics.confusion_matrix(y_test,y_predict)

print '交叉验证结果..........'
dtc_scores = cross_val_score(DTC,X,Y,cv=5)
print dtc_scores

#GBDT
GBDT = GradientBoostingClassifier()
GBDT.fit(x_train,y_train)
y_predict = GBDT.predict(x_test)
print 'GBDT分类结果..............'
print metrics.classification_report(y_test,y_predict)
print metrics.confusion_matrix(y_test,y_predict)


print '交叉验证结果..........'
gbdt_scores = cross_val_score(GBDT,X,Y,cv=5)
print gbdt_scores

#GBDT
RF = RandomForestClassifier()
RF.fit(x_train,y_train)
y_predict = RF.predict(x_test)
print '随机森林分类结果..............'
print metrics.classification_report(y_test,y_predict)
print metrics.confusion_matrix(y_test,y_predict)

print '交叉验证结果..........'
rf_scores = cross_val_score(RF,X,Y,cv=5)
print rf_scores

# 高斯NB
GNB = GaussianNB()
GNB.fit(x_train,y_train)
y_predict = GNB.predict(x_test)
print '高斯贝叶斯分类结果..............'
print metrics.classification_report(y_test,y_predict)
print metrics.confusion_matrix(y_test,y_predict)

print '交叉验证结果..........'
gnb_scores = cross_val_score(GNB,X,Y,cv=5)
print gnb_scores

# K近邻
KNN = KNeighborsClassifier()
KNN.fit(x_train,y_train)
y_predict = KNN.predict(x_test)
print 'KNN分类结果..............'
print metrics.classification_report(y_test,y_predict)
print metrics.confusion_matrix(y_test,y_predict)

print '交叉验证结果..........'
knn_scores = cross_val_score(KNN,X,Y,cv=5)
print knn_scores


# SVM
SVC = SVC()
SVC.fit(x_train,y_train)
y_predict = SVC.predict(x_test)
print 'SVC分类结果..............'
print metrics.classification_report(y_test,y_predict)
print metrics.confusion_matrix(y_test,y_predict)

print '交叉验证结果..........'
svc_scores = cross_val_score(SVC,X,Y,cv=5)
print svc_scores

# 今天把前两天提取的一些网站流量作为正例和ssh远程登录作为负例流量做了些的实验，目前来看假如提取的流量和相应的特征具有代表性的话，两类流量目前来看还是很好区分的，即使不调超参数，大部分分类模型分类效果都还是挺好的（除了svm，可能是训练数据有噪音导致的），树类型的模型相对其他分类模型效果最好。





