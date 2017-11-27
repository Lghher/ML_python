#coding=utf-8
import numpy as np
import urllib
from sklearn import  preprocessing
from sklearn.ensemble import  ExtraTreesClassifier
from sklearn import  metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
def savefile(path,content):
    fp = open(path,"wb")
    fp.write(content)
    fp.close()
def readfile(path):
    fp = open(path,"rb")
    content = fp.read()
    fp.close()
    return content
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
data = urllib.urlopen(url)
# savefile("C:\Users\Administrator\Desktop\hh_practice.txt",data.read())
dataset = np.loadtxt(data,delimiter=",")
# print dataset
X = dataset[:,0:8]
Y = dataset[:,8]
# print "X:"
# print X
# print "Y:"
# print Y

#数据归一化
# X=[[6,8,10],[600,800,1000],[600,800,10000]]
# normalized_X = preprocessing.normalize(X)
# print normalized_X

#数据标准化
# standardized_X = preprocessing.scale(X)
# print standardized_X

#决策树
model = ExtraTreesClassifier()
model.fit(X,Y)
# expected = Y
# predicted = model.predict(X)
# print(metrics.classification_report(expected,predicted))
# print(metrics.confusion_matrix(expected,predicted))
# 打印特征的信息增益
"""
为什么每次打印的信息增益不一样
"""
# print(model.feature_importances_)

# LR
model = LogisticRegression()
model.fit(X,Y)
print(model)
expected = Y
predicted = model.predict(X)
print predicted
#预测结果
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))


# 高斯NB
model = GaussianNB()
model.fit(X,Y)
expected = Y
predicted = model.predict(X)
#预测结果
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

# K近邻
model = KNeighborsClassifier()
model.fit(X,Y)
print model
expected = Y
predicted = model.predict(X)
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

# DT
model = DecisionTreeClassifier()
model.fit(X,Y)
print(model)
expected = Y
predicted = model.predict(X)
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

# SVM
model = SVC()
model.fit(X,Y)
print model
expected = Y
predicted = model.predict(X)
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

#调参 参数搜索
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
model = Ridge()
grid = GridSearchCV(estimator=model,param_grid=dict(alpha=alphas))
grid.fit(X,Y)
print grid
print(grid.best_score_)
print(grid.best_estimator_.alpha)











