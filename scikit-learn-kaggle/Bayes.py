#-*- coding=utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import   MultinomialNB
from sklearn import  metrics
news =  fetch_20newsgroups(subset='all')
print news
x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=66)
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
predict = mnb.predict(x_test)
print 'The accuracy of Naive Bayes Classifier is', mnb.score(x_test, y_test)

print metrics.classification_report(y_test,predict)
