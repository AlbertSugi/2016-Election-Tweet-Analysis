#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 20:14:48 2018

@author: albert
"""

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
#from pandas import Series
import seaborn as sns
#import calendar
#import datetime
import re
#from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import ensemble



from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 


df1 = pd.read_csv('tweets.csv', encoding="utf-8")
df1 = df1[['handle','text','is_retweet']]

df = df1.loc[df1['is_retweet'] == False]
df = df.copy().reset_index(drop=True)

def all_mentions(tw):
    Filter = re.findall('(\@[A-Za-z_]+)', tw)
    if Filter:
        return Filter
    else:
        return ""

df['top_mentions'] = df['text'].apply(lambda x: all_mentions(x))


def get_hashtags(tw):
    test3 = re.findall('(\#[A-Za-z_]+)', tw)
    if test3:
        return test3
    else:
        return ""
    


def candidate_code(x):
    if x == 'HillaryClinton':
        return 1
    elif x == 'realDonaldTrump':
        return 0
    else:
        return ''
    
    

df['top_hashtags'] = df['text'].apply(lambda x: get_hashtags(x))
df['length_no_url'] = df['text']
df['length_no_url'] = df['length_no_url'].apply(lambda x: len(x.lower().split('http')[0]))
df['message'] = df['text'].apply(lambda x: x.lower().split('http')[0])

df['label'] = df['handle'].apply(lambda x: candidate_code(x))


#print(df)

messages = df[['label','message']]

#print(messages[:5])

def split_into_tokens(message):
    message = message  # convert bytes into proper unicode
    return TextBlob(message).words

messages.message.head()
messages.message.head().apply(split_into_tokens)


def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

messages.message.head().apply(split_into_lemmas)


bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
#print(len(bow_transformer.vocabulary_))
#print(bow_transformer.get_feature_names()[:5])

msg_train, msg_test, label_train, label_test = \
    train_test_split(messages['message'], messages['label'], test_size=0.3, random_state=1)


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', DecisionTreeClassifier()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
    
    
pipeline3 = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', ensemble.RandomForestClassifier()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

    
pipeline2 = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])    
    
    
params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}


grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_splits=10),  # what type of cross validation to use
)


grid2 = GridSearchCV(
    pipeline2,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train,  n_splits=10),  # what type of cross validation to use
)



grid3 = GridSearchCV(
    pipeline3,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train,  n_splits=10),  # what type of cross validation to use
)



SV_detector = grid.fit(msg_train, label_train)

#print(SV_detector.grid_scores_)


predictions = SV_detector.predict(msg_test)
#print(classification_report(label_test, predictions))

#print("Accuracy on test set:  %.2f%%" % (100 * (SV_detector.grid_scores_[0][1])))


MNB_detector = grid2.fit(msg_train, label_train)

#print(MNB_detector.grid_scores_)


predictions2 = MNB_detector.predict(msg_test)
#print(classification_report(label_test, predictions2))

#print("Accuracy on test set:  %.2f%%" % (100 * (MNB_detector.grid_scores_[0][1])))

RF_detector = grid3.fit(msg_train, label_train)

#print(RF_detector.grid_scores_)


predictions3 = RF_detector.predict(msg_test)
#print(classification_report(label_test, predictions3))

print("Accuracy on test set:  %.2f%%" % (100 * (RF_detector.grid_scores_[0][1])))



plt.figure(figsize=(8,8))

fpr, tpr, _ = metrics.roc_curve(label_test,  predictions)
auc = metrics.roc_auc_score(label_test, predictions)
plt.plot(fpr,tpr,color= 'yellowgreen',label="data DTC, auc="+str(auc))
plt.legend(loc=4)



fpr, tpr, _ = metrics.roc_curve(label_test,  predictions2)
auc = metrics.roc_auc_score(label_test,  predictions2)
plt.plot(fpr,tpr,color= 'darkred',label="data MNB, auc="+str(auc))
plt.legend(loc=4)


fpr, tpr, _ = metrics.roc_curve(label_test,  predictions3)
auc = metrics.roc_auc_score(label_test,  predictions3)
plt.plot(fpr,tpr,color= 'lightblue',label="data RFC, auc="+str(auc))
plt.legend(loc=4)
lw = 2
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.title('ROC Plot')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(label_test, predictions), annot=True, linewidths=.5, ax=ax, cmap="Blues", fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
plt.title('Test Set Confusion Matrix')

fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(label_test, predictions2), annot=True, linewidths=.5, ax=ax, cmap="Blues", fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
plt.title('Test Set Confusion Matrix')

fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(label_test, predictions2), annot=True, linewidths=.5, ax=ax, cmap="Blues", fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
plt.title('Test Set Confusion Matrix')



your_tweet = input("ENTER YOUR TWEET HERE:  ")


# ---- this part stays the same----
if your_tweet == your_tweet:
    k = (100 * max(MNB_detector.predict_proba([your_tweet])[0]))
    i = MNB_detector.predict([your_tweet])[0]
    if i == 1:
        i = "Hillary"
    else:
        i = "Trump"
    print("Tweet #1:", "'",your_tweet, "'", ' \n \n', "I'm about %.0f%%" % k,  "sure this was tweeted by", i)
