#-*- coding=utf-8 -*-
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
boston = load_boston()
print boston.DESCR
# print boston.data

x = boston.data
y = boston.target
print y
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=66)

print "max target",np,max(boston.target)
print "min target",np.min(boston.target)
print "mean target",np.mean(boston.target)
SX = StandardScaler()
SY = StandardScaler()

x_train = SX.fit_transform(x_train)
x_test = SX.transform(x_test)
# print "测试y数据"
# print y_test
# y_train = SY.fit_transform(y_train)
# y_test = SY.transform(y_test)
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_y_predict = lr.predict(x_test)
# print "预测y数据"
# print lr_y_predict
print "lr总体测试数据成绩"
print lr.score(x_test,y_test)
print "R-squared:",metrics.r2_score(y_test,lr_y_predict)
print "mean squared error:",metrics.mean_squared_error(y_test,lr_y_predict)
print "absolute error:",metrics.median_absolute_error(y_test,lr_y_predict)
sgdr = SGDRegressor()
sgdr.fit(x_train,y_train)
sgdr_y_predict = sgdr.predict(x_test)
print "sgdr总体测试数据成绩"
print sgdr.score(x_test,y_test)
print "R-squared:",metrics.r2_score(y_test,sgdr_y_predict)
print "mean squared error:",metrics.mean_squared_error(y_test,sgdr_y_predict)
print "absolute error:",metrics.median_absolute_error(y_test,sgdr_y_predict )








