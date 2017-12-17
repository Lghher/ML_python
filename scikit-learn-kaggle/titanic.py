#-*- coding=utf-8 -*-
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  GradientBoostingClassifier
data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
# print data
# data.to_csv(r"C:\Users\hanghang\Desktop\titanic.csv",index=None)
# new_data = pd.read_csv(r"C:\Users\hanghang\Desktop\titanic.csv")
# print new_data
print "训练数据信息"
print data.info()
print data.describe()
print "输出训练数据前几列"
# data.describe()只有数值变量才有这个信息
print data.head()
print "..............."
print "..............."
x = data[['pclass','age','sex']]
y = data['survived']
print x.info()
print x
x['age'].fillna(x['age'].mean(),inplace=True)
print x.info()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=66)
# print "x的训练数据"
# print x_train
vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
print vec.feature_names_
print x_train
x_test = vec.transform(x_test.to_dict(orient='record'))
DT = DecisionTreeClassifier()
DT.fit(x_train,y_train)
print "单一决策树开始预测....................."
y_predict = DT.predict(x_test)
print "预测准确率"
print DT.score(x_test,y_test)
print "详细的分类性能"
print metrics.classification_report(y_test,y_predict,target_names=['died','survived'])
print "混淆矩阵"
print metrics.confusion_matrix(y_test,y_predict)

DT = RandomForestClassifier()
DT.fit(x_train,y_train)
print "随机森林开始预测....................."
y_predict = DT.predict(x_test)
print "预测准确率"
print DT.score(x_test,y_test)
print "详细的分类性能"
print metrics.classification_report(y_test,y_predict,target_names=['died','survived'])
print "混淆矩阵"
print metrics.confusion_matrix(y_test,y_predict)

DT = GradientBoostingClassifier()
DT.fit(x_train,y_train)
print "梯度提升树开始预测....................."
y_predict = DT.predict(x_test)
print "预测准确率"
print DT.score(x_test,y_test)
print "详细的分类性能"
print metrics.classification_report(y_test,y_predicttarget_names=['died','survived'])
print "混淆矩阵"
print metrics.confusion_matrix(y_test,y_predict)

