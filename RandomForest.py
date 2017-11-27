#coding=utf-8
import pandas as pd
import numpy as np
from  sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import  GridSearchCV
from sklearn import cross_validation,metrics
import  matplotlib.pylab as plt
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split


train = pd.read_csv("C:\Users\Administrator\Desktop\hh_practice.csv")
# for row in train:
#     print row
#     print 8888
# print train.head(10)
# print train['result'][:10]
print '类别输出'
target = "result"
print train['result'].value_counts()
print '样本的特征'
print train.columns
# print train[5:6]
#将特征和类型分开
x_col = [x for x in train.columns if x != 'result']
X = train[x_col]
# print X
Y = train['result']

rf_model = RandomForestClassifier();
rf_model.fit(X,Y)
expected = Y
# predicted = rf_model.predict(X)
# #预测结果
# print(metrics.classification_report(expected,predicted))
# print(metrics.confusion_matrix(expected,predicted))
y_predprob = rf_model.predict_proba(X)
# print y_predprob
# #参数调整范围
# param_test1= {'n_estimators':range(10,100,10)}
# gsearch1= GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
#                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),
#                        param_grid =param_test1, scoring='roc_auc',cv=5)
# gsearch1.fit(X,Y)
# print '第一次调整参数'
# print gsearch1.grid_scores_
# print gsearch1.best_params_
# print gsearch1.best_score_
#
# param_test2= {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
# gsearch2= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 70,
#                                  min_samples_leaf=20 ,oob_score=True,random_state=10),
#    param_grid = param_test2,scoring='roc_auc',iid=False, cv=5)
# gsearch2.fit(X,Y)
# print '第二次参数优化'
# print gsearch2.grid_scores_
# print gsearch2.best_params_
# print gsearch2.best_score_
#
# param_test3= {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
# gsearch3= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 70,max_depth=7,min_samples_split=50,
#                                   oob_score=True, random_state=10),
#    param_grid = param_test3,scoring='roc_auc',iid=False, cv=5)
# gsearch3.fit(X,Y)
# print '第三次参数优化'
# print gsearch3.grid_scores_
# print gsearch2.best_params_
# print gsearch2.best_score_
#
# param_test4= {'max_features':range(3,8,2)}
# gsearch4= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 70,max_depth=7, min_samples_split=50,
#                                  min_samples_leaf=20 ,oob_score=True, random_state=10),
#    param_grid = param_test4,scoring='roc_auc',iid=False, cv=5)
# gsearch4.fit(X,Y)
# print '第四次参数优化'
# print gsearch4.grid_scores_
# print gsearch4.best_params_
# print gsearch4.best_score_

#使用默认参数
rf_model = RandomForestClassifier();
rf_model.fit(X,Y)
expected = Y
predicted = rf_model.predict(X)
#预测结果
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

#使用优化后的参数
new_rf_model = RandomForestClassifier(n_estimators=70,min_samples_split=50,max_depth=7,max_features=3);
new_rf_model.fit(X,Y)
expected = Y
predicted = new_rf_model.predict(X)
#预测结果
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

#持久化模型 此处的要注意一定要设置compress=3,不然就会很多npy后缀的文件,是numpy存储文件的格
# 式.这个参数貌似是压缩的
joblib.dump(new_rf_model,r"C:\Users\Administrator\Desktop\temhhhh\rf.model",compress=3)

#载入模型 joblib.load(path)


#将数据划分为训练集和测试集
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5,random_state=66)
print 'X的训练数据'
print X_train
print 'Y的训练数据'
print Y_train
print 'X的测试集'
print X_test
print 'Y的测试集'
print Y_test
hh_rf = RandomForestClassifier()
hh_rf.fit(X_train,Y_train)
train_expected = Y_train
train_predicted=new_rf_model.predict(X_train)
print '训练效果'
print(metrics.classification_report(train_expected,train_predicted))
print(metrics.confusion_matrix(train_expected,train_predicted))

test_expected = Y_test
test_predicted=new_rf_model.predict(X_test)
print '预测效果'
print(metrics.classification_report(test_expected,test_predicted))
print(metrics.confusion_matrix(test_expected,test_predicted))



