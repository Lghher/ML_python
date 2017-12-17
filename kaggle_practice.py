#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import   LogisticRegression

data_train = pd.read_csv("C:\Users\hanghang\Desktop\T_train.csv")
# print data_train
print data_train.info()
print data_train
print data_train.describe()
# fig = plt.figure()
# # 两行三列
# plt.subplot2grid((2,6),(0,0))
# data_train["Survived"].value_counts().plot(kind='bar')
# plt.title(u"获救情况")
# plt.ylabel(u"人数")
#
# plt.subplot2grid((2,6),(0,2))
# data_train["Pclass"].value_counts().plot(kind='bar')
# plt.title(u"乘客等级分布")
# plt.ylabel(u"人数")
# plt.subplot2grid((2,6),(0,4))
# # 一个为x轴，要给为y轴
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel(u"年龄")
# plt.grid(b=True, which='major', axis='y')
# plt.title(u"按年龄看获救分布 (1为获救)")
#
#
# plt.subplot2grid((2,6),(1,0), colspan=4)
# data_train["Age"][data_train.Pclass == 1].plot(kind='kde')
# data_train["Age"][data_train.Pclass == 2].plot(kind='kde')
# data_train["Age"][data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"各等级的乘客年龄分布")# plots an axis lable
# plt.ylabel(u"密度")
# plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best')
#
#
# plt.subplot2grid((2,6),(1,5))
# data_train["Embarked"].value_counts().plot(kind='bar')
# plt.title(u"各登船口岸上船人数")
# plt.ylabel(u"人数")
# plt.show()
#

################
# 性别与获救结果的关系
################
# fig = plt.figure()
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
# df.plot(kind='bar')
# plt.title(u"按性别看获救情况")
# plt.xlabel(u"性别")
# plt.ylabel(u"人数")
# plt.show()
###########
#利用RF进行缺失值的预测
data_train1 = data_train[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
age =  data_train1[data_train1['Age'].notnull()].as_matrix()
null_age = data_train1[data_train1['Age'].isnull()].as_matrix()
print null_age
y = age[:,0]
x = age[:,1:]
rf = RandomForestRegressor()
rf.fit(x,y)
predict_age = rf.predict(null_age[:,1:])
print predict_age
data_train.loc[data_train['Age'].isnull(),'Age'] = predict_age
data_train.to_csv(r"C:\Users\hanghang\Desktop\hhhhhhhhhhhhh_train.csv")

data_train.loc[data_train['Cabin'].notnull(),'Cabin'] = 'YES'
data_train.loc[data_train['Cabin'].isnull(),'Cabin'] = 'NO'
print data_train

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)






