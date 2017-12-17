#-*- coding=utf-8 -*-
#C:\Users\hanghang\PycharmProjects\untitled\Datasets\Breast-Cancer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
bc_train = pd.read_csv(r"C:\Users\hanghang\PycharmProjects\untitled\Datasets\Breast-Cancer\breast-cancer-train.csv")
bc_test = pd.read_csv(r"C:\Users\hanghang\PycharmProjects\untitled\Datasets\Breast-Cancer\breast-cancer-test.csv")
bc_test_P = bc_test.loc[bc_test['Type']==0][['Clump Thickness','Cell Size']]
bc_test_N = bc_test.loc[bc_test['Type']==1][['Clump Thickness','Cell Size']]
plt.scatter(bc_test_N['Clump Thickness'],bc_test_N['Cell Size'],c='red',marker='o')
plt.scatter(bc_test_P['Clump Thickness'],bc_test_P['Cell Size'],c='black',marker='x')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
a = np.random.random([1])
b = np.random.random([2])
lx = np.arange(0,12)
ly = (-a-lx*b[0])/b[1]
print ly
plt.plot(lx,ly,c='yellow')
plt.show()
LR = LogisticRegression()
LR.fit(bc_train[['Clump Thickness','Cell Size']][:10],bc_train['Type'][:10])
print 'Testing accurac',LR.score(bc_test[['Clump Thickness','Cell Size']],bc_test['Type'])
intercept = LR.intercept_
coef = LR.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='green')
plt.scatter(bc_test_N['Clump Thickness'],bc_test_N['Cell Size'], marker = 'o', c='red')
plt.scatter(bc_test_P['Clump Thickness'],bc_test_P['Cell Size'], marker = 'x', c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()
new_LR = LogisticRegression()
new_LR.fit(bc_train[['Clump Thickness','Cell Size']],bc_train['Type'])
print 'Testing accurac',new_LR.score(bc_test[['Clump Thickness','Cell Size']],bc_test['Type'])
ntercept = new_LR.intercept_
coef = new_LR.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='blue')
plt.scatter(bc_test_N['Clump Thickness'],bc_test_N['Cell Size'], marker = 'o', c='red')
plt.scatter(bc_test_P['Clump Thickness'],bc_test_P['Cell Size'], marker = 'x', c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()
plt.plot([1,2,3,4],[1,2,1,2])
plt.show()

