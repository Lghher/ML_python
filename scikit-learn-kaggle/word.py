#-*- coding=utf-8 -*-
from sklearn.feature_extraction import DictVectorizer #dict转换向量
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import  TfidfVectorizer
# 从sklearn.datasets里导入20类新闻文本数据抓取器。
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  MultinomialNB
from sklearn import  metrics
import types
measurements = [{'city': 'Dubai', 'temperature': 33.}, {'city': 'London', 'temperature': 12.}, {'city': 'San Fransisco', 'temperature': 18.}]
vec = DictVectorizer()
data = vec.fit_transform(measurements).toarray()
print data,type(data)
print vec.get_feature_names()

####################################################

news = fetch_20newsgroups(subset='all')
x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=66)
# print news.data

# CountVectorizer可以英文句子向量化，特征为出现的每一个单词，特征值为出现的个数
count_vec = CountVectorizer()
# x_count_train = count_vec.fit_transform(["I like renlingling like","renlingling like hanghang"])
# print x_count_train.toarray(),count_vec.get_feature_names()

x_count_train = count_vec.fit_transform(x_train)
# print x_count_train
x_count_test = count_vec.transform(x_test)

mnb = MultinomialNB()
mnb.fit(x_count_train,y_train)
print 'The accuracy of classifying(CountVectorizer without filtering stopwords):', mnb.score(x_count_test, y_test)
y_count_predict = mnb.predict(x_count_test)
print metrics.classification_report(y_test,y_count_predict,target_names=news.target_names)

tfidf_vec = TfidfVectorizer()
x_tfidf_train = tfidf_vec.fit_transform(x_train)
# print x_tfidf_train
x_tfidf_test = tfidf_vec.transform(x_test)
mnb = MultinomialNB()
mnb.fit(x_tfidf_train,y_train)
print 'The accuracy of classifying(TfidfVectorizer without filtering stopwords):', mnb.score(x_tfidf_test, y_test)
y_tfidi_predict = mnb.predict(x_tfidf_test)
print metrics.classification_report(y_test,y_tfidi_predict,target_names=news.target_names)

print "count过滤停用词结果"
count_filter_vec = CountVectorizer(analyzer='word', stop_words='english')
X_count_filter_train = count_filter_vec.fit_transform(x_train)
X_count_filter_test = count_filter_vec.transform(x_test)
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train, y_train)
print 'The accuracy of classifying (CountVectorizer by filtering stopwords):', mnb_count_filter.score(X_count_filter_test, y_test)
y_count_filter_predict = mnb_count_filter.predict(X_count_filter_test)
print metrics.classification_report(y_test, y_count_filter_predict, target_names = news.target_names)



print "tfidf过滤停用词结果"
tfidf_filter_vec = TfidfVectorizer(analyzer='word', stop_words='english')
X_tfidf_filter_train = tfidf_filter_vec.fit_transform(x_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(x_test)
mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train, y_train)
print 'The accuracy of classifying (TfidfVectorizer by filtering stopwords):', mnb_tfidf_filter.score(X_tfidf_filter_test, y_test)
y_tfidf_filter_predict = mnb_tfidf_filter.predict(X_tfidf_filter_test)

print metrics.classification_report(y_test, y_tfidf_filter_predict, target_names = news.target_names)






