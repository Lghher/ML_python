#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import  cross_validation,metrics
from sklearn.grid_search import  GridSearchCV
import matplotlib.pylab as plt
from sklearn.cross_validation import  train_test_split
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv(r"C:\Users\Administrator\Desktop\train_modified.csv")

print train["Disbursed"].value_counts()
X_columns = [x  for x in train.columns if x != "ID" and x != "Disbursed"]
print X_columns
X = train[X_columns]
Y = train["Disbursed"]
gbdt = GradientBoostingClassifier(random_state=10)
gbdt.fit(X,Y)
predicted =  gbdt.predict(X)
    expected = Y
# print(metrics.accuracy_score(expected,predicted))
#和上面那个结果一样
print(metrics.accuracy_score(expected.values,predicted))
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))
# train = pd.read_csv("C:\Users\Administrator\Desktop\hh_practice.csv")
# print train["result"].value_counts()
# X_columns = [x  for x in train.columns if x != "result"]
# print X_columns
# h_X = train[X_columns]
# h_Y = train["result"]

params = {"n_estimators":range(100,2000,100)}
gsearch = GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=params,scoring="roc_auc")
gsearch.fit(X,Y)
print gsearch.grid_scores_
print gsearch.best_params_
print gsearch.best_score_
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5,random_state=66)
gbdt_model =GradientBoostingClassifier(n_estimators=5000)
gbdt_model.fit(X_train,Y_train)
expected = Y_train
#训练结果
predicted = gbdt_model.predict(X_train)
print "训练结果"
print(metrics.accuracy_score(expected.values,predicted))
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))
#测试结果
expected = Y_test
predicted = gbdt_model.predict(X_test)
print "测试结果"
print(metrics.accuracy_score(expected.values,predicted))
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))
