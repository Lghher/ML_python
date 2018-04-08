#-*- coding=utf-8 -*-
# pandas的df[]不能同时行列取值，可以用iloc和loc取值
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor

# data_train[data_train['Survived']==0]
# data_train.columns.dtype
# data_train.shape
# data_train
# data_train.info()
# data_train.describe()

# 不加这个不能bagging，什么鬼
if __name__=='__main__':
    data_train = pd.read_csv("C:/Users/hanghang/Desktop/Titanic/train.csv");
    def graph_analyze():
        fig = plt.figure()
        #设置图片透明度
        fig.set(alpha=0.2)
        #subplot2grid有两个参数 一个是总大小，另一个是当前图形的左上角坐标
        plt.subplot2grid((3,6),(0,0))
        data_train['Survived'].value_counts().plot(kind='bar')
        plt.title("获救情况(1为获救)")
        plt.ylabel("人数")

        plt.subplot2grid((3,6),(0,2))
        data_train['Pclass'].value_counts().plot(kind="bar")
        plt.title("乘客等级")
        plt.ylabel("人数")

        plt.subplot2grid((3,6),(0,4))
        plt.scatter(data_train['Survived'],data_train['Age'])
        plt.title("年龄与获救的关系(1为获救)")
        plt.ylabel("年龄")


        plt.subplot2grid((3,6),(2,0),colspan=4)
        data_train['Age'][data_train['Pclass']==1].plot(kind='kde')
        data_train['Age'][data_train['Pclass']==2].plot(kind='kde')
        data_train['Age'][data_train['Pclass']==3].plot(kind='kde')
        plt.xlabel("年龄")
        plt.ylabel("密度")
        plt.title("各等级的乘客年龄分布")
        plt.legend(("头等舱","二等舱","三等舱"))

        plt.subplot2grid((3,6),(2,5))
        data_train['Embarked'].value_counts().plot(kind='bar')
        plt.ylabel('人数')
        plt.title("各登船口岸上船人数")
        # plt.show()

        fig = plt.figure()
        fig.set(alpha=0.2)
        Survived_0 = data_train['Pclass'][data_train['Survived']==0].value_counts()
        Survived_1 = data_train['Pclass'][data_train['Survived']==1].value_counts()
        df = pd.DataFrame({'获救':Survived_1,'未获救':Survived_0})
        df.plot(kind='bar')
        plt.title("各乘客等级的获救情况")
        plt.xlabel("乘客等级")
        plt.ylabel("人数")
        # plt.show()

        fig = plt.figure()
        fig.set(alpha=0.2)
        Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
        Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
        df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
        df.plot(kind='bar')
        plt.title("按性别看获救情况")
        plt.xlabel("性别")
        plt.ylabel("人数")
        # plt.show()
    def set_miss_ages(df):
        # 把数值型特征取出来
        age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
        # 已经年龄和未知年龄
        age = age_df[age_df['Age'].notnull()]
        age_null = age_df[age_df['Age'].isnull()]
        y = age.iloc[:,0]
        x = age.iloc[:,1:]
        # 用已存在的年龄训练rf，还可以分割一些训练集，看一下模型的回归效果
        rf = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
        rf.fit(x,y)
        predict_age = rf.predict(age_null.iloc[:,1:])
        # 未知年龄
        # print(df.loc[(df['Age'].isnull()), 'Age'])
        df.loc[(df['Age'].isnull()), 'Age'] = predict_age
        # 预测后的年龄
        # print(df.loc[(df['Age'].isnull()), 'Age'])
        return df,rf

    # print(data_train['Cabin'].value_counts())
    # 由于Cabin这个特征是离散的，而且取值比较多，而且比较分散，所以按Cabin有无数据，转换成Yes和No两种取值
    def set_miss_Cabin(df):
        df.loc[df['Cabin'].notnull(),'Cabin'] = 'YES'
        df.loc[df['Cabin'].isnull(),'Cabin'] = 'NO'
        return df
    data_train,rf = set_miss_ages(data_train)
    data_train = set_miss_Cabin(data_train)
    # 由于是二分类用logstic做一个baseline model ，所以要将非数值的特征转化为数值特征，用onehot
    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
    # axis=1表示列连接
    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    # 去掉非数值特征的列，inplace=True表示df已经改变，不用再df = df.drop()
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    # 标准化在这个问题中对最后效果影响不大
    scaler = StandardScaler()
    df['Age'] = scaler.fit_transform(df['Age'].reshape(-1,1))
    scaler1 = StandardScaler()
    df['Fare'] = scaler1.fit_transform(df['Fare'].reshape(-1,1))
    df.drop('PassengerId',axis=1,inplace=True)

    y = df.iloc[:,0]
    x = df.iloc[:,1:]
    lr = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    lr.fit(x,y)
    bagging_lr = BaggingRegressor(lr, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
    # bagging1_lr = BaggingClassifier(lr, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
    bagging_lr.fit(x,y)
    print(cross_val_score(lr, x, y, cv=5))

    #这个效果不太好
    gbdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60)
    gbdt.fit(x,y)


    # 测试数据的处理
    data_test = pd.read_csv("C:/Users/hanghang/Desktop/Titanic/test.csv")
    data_test.loc[ (data_test['Fare'].isnull()), 'Fare' ] = 0
    # 接着我们对test_data做和train_data中一致的特征变换
    # 首先用同样的RandomForestRegressor模型填上丢失的年龄
    tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].as_matrix()
    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rf.predict(X)
    data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

    data_test = set_miss_Cabin(data_test)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    df_test['Age'] = scaler.transform(df_test['Age'].reshape(-1,1))
    df_test['Fare'] = scaler1.transform(df_test['Fare'].reshape(-1,1))
    id = df_test['PassengerId']
    # pandas正则选择列
    # test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    df_test.drop('PassengerId',axis=1,inplace=True)
    # predicted = lr.predict(df_test)
    predicted = (bagging_lr.predict(df_test)+0.5).astype(int)
    res = pd.DataFrame({'PassengerId':id, 'Survived':predicted})
    res.to_csv("C:/Users/hanghang/Desktop/Titanic/bagging_result1.csv", index=False)
    # 系数为正的特征，和预测结果是一个正相关，反之为负相关。
    # print(pd.DataFrame({"columns":list(df.columns)[1:], "coef":list(lr.coef_.T)}))



