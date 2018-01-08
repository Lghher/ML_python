#-*- coding=utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn import metrics
matrix = np.array([[1,2],[2,4]])
print np.linalg.matrix_rank(matrix)
digit_train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",header=None)
digit_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",header=None)
x_train = digit_train[range(64)]
y_train = digit_train[64]
x_test = digit_test[range(64)]
y_test = digit_test[64]
estimator = PCA(n_components=2)
x_pca = estimator.fit_transform(x_train)
# print x_pca
# print y_train
# print y_train.as_matrix()
def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in xrange(len(colors)):
        x = x_pca[:,0][y_train.as_matrix()==i]
        y = x_pca[:,1][y_train.as_matrix()==i]
        plt.scatter(x,y,c=colors[i])
    # plt.legend(np.arange(10).astype(str))
    plt.xlabel('First')
    plt.ylabel('Second')
    plt.show()
# plot_pca_scatter()
svc = LinearSVC()
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)
estimator = PCA(n_components=20)
pca_x_train = estimator.fit_transform(x_train)
pca_x_test = estimator.transform(x_test)
pca_svc = LinearSVC()
pca_svc.fit(pca_x_train,y_train)
pca_y_predict = pca_svc.predict(pca_x_test)
print svc.score(x_test,y_test)
print metrics.classification_report(y_test,y_predict)
print "PCA压缩重建特征后的分类结果"
print pca_svc.score(pca_x_test,y_test)
print metrics.classification_report(y_test,pca_y_predict)


