#-*- coding=utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Normalizer对每个样本计算其p-范数，再对每个元素除以该范数，
from sklearn.preprocessing import Normalizer
# 定量特征二值化设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

from sklearn.feature_selection import VarianceThreshold
iris = load_iris()
# print(iris.data) #4个特征  float64
# print(iris.target) //3个类别 0 1 2
# standard = StandardScaler()
# iris.data = standard.fit_transform(iris.data)
# print(iris.data)
# minmax = MinMaxScaler()
# iris.data = MinMaxScaler.fit_transform(iris.data)
# normal = Normalizer()
# iris.data = normal.fit_transform(iris.data)
# binarizer = Binarizer(threshold=3)
# print(binarizer.fit_transform(iris.data))
# onhot = OneHotEncoder()
# print(onhot.fit_transform(iris.target.reshape(-1,1)))
# VarianceThreshold(threshold=4).fit_transform(iris.data)
print(PCA(n_components=2).fit_transform(iris.data))