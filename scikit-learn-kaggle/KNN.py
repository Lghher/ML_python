#-*- coding=utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
iris = load_iris()

print iris.DESCR
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=666)
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
knnc = KNeighborsClassifier()
knnc.fit(x_train,y_train)
predict = knnc.predict(x_test)
print 'The accuracy of KNN classifier is',knnc.score(x_test,y_test)

print metrics.classification_report(y_test,predict)
print metrics.confusion_matrix(y_test,predict)
