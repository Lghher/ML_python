#-*- coding=utf-8 -*-
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

news = fetch_20newsgroups(subset='all')
# print news.data[:3000]
x_train, x_test, y_train, y_test = train_test_split(news.data[:3000], news.target[:3000], test_size=0.25, random_state=33)


clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])

parameters = {'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3,n_jobs=4)

# param_test4= {'max_features':range(3,8,2)}
# gsearch4= GridSearchCV(estimator=RandomForestClassifier(n_estimators= 70,max_depth=7, min_samples_split=50,
#                                  min_samples_leaf=20 ,oob_score=True, random_state=10),param_grid=param_test4,scoring='roc_auc',iid=False, cv=5)
# gsearch4.fit(X,Y)
# print '第四次参数优化'
# print gsearch4.grid_scores_
# print gsearch4.best_params_
# print gsearch4.best_score_

gs.fit(x_train, y_train)
print gs.best_params_, gs.best_score_

# 输出最佳模型在测试集上的准确性。
print gs.score(x_test, y_test)
