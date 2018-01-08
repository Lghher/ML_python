#-*- coding=utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# 从sklearn.preprocessing导入多项式特征生成器。
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
x_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]
x_test = [[6],[8],[11],[16]]
regressor =  LinearRegression()
regressor.fit(x_train,y_train)

xx = np.linspace(0,26,100)
# print type(xx)
xx = xx.reshape(xx.shape[0],1)
# print type(xx)
yy = regressor.predict(xx)
plt.scatter(x_train,y_train)
plt.plot(xx,yy,label="Degree = 1")
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.show()
print 'The R-squared value of Linear Regressor performing on the training data is', regressor.score(x_train, y_train)
poly2 = PolynomialFeatures(degree=10)
x_train_poly2 = poly2.fit_transform(x_train)
# print  x_train_poly2

regressor_poly2 = LinearRegression()
regressor_poly2.fit(x_train_poly2, y_train)



xx_poly2 = poly2.transform(xx)
yy_poly2 = regressor_poly2.predict(xx_poly2)
plt.scatter(x_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
print  xx
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles = [plt1, plt2])
plt.show()

