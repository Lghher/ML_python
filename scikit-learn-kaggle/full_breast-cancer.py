#-*- coding=utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
from sklearn import metrics
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
#加上列索引
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",names=column_names)
# to_csv保存有行索引
# data.to_csv(r"C:\Users\hanghang\Desktop\afafdasf.csv")
#列中有非数字的就无法计算
#data.loc[0,'Sample code number']='？'
print data.describe()
print data.info()
# 将?替换为标准缺失值表示
data = data.replace(to_replace='?',value=np.nan)
#丢弃带有缺失值的数据（只要有一个维度缺失）
data = data.dropna(how='any')
# print data.shape
# print data.columns
x_columns = [x for x in data.columns if x != 'Class']
print x_columns
x_train,x_test,y_train,y_test = train_test_split(data[x_columns],data['Class'],test_size=0.25,random_state=66)
# print x_train
print y_train.value_counts()
print y_test.value_counts()
# 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导。
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#LR模型
lr = LogisticRegression()
lr.fit(x_train,y_train)
lr_predict = lr.predict(x_test)
print 'LR模型预测结果'
print lr.score(x_test,y_test)
print(metrics.classification_report(y_test,lr_predict))
print(metrics.confusion_matrix(y_test,lr_predict))

#SGD模型
sgdc = SGDClassifier()
sgdc.fit(x_train,y_train)
sgdc_predict = sgdc.predict(x_test)
print 'SGDC模型预测结果'
print sgdc.score(x_test,y_test)
print(metrics.classification_report(y_test,sgdc_predict))
print(metrics.confusion_matrix(y_test,sgdc_predict))







