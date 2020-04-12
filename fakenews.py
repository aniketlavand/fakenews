#!/usr/bin/env python
# coding: utf-8

# In[1]:


#important imports
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
#Reading data
an=pd.read_csv('D:\\news.csv')
#shape and head
an.shape
an.head()


# In[2]:


#labels
labels=an.label
labels.head()


# In[4]:


#Spliting the dataset into train and test set 
x_train, x_test, y_train, y_test = train_test_split(an['text'], labels, test_size = 0.2, random_state = 7 )


# In[5]:


#Initializing a TfidfVectorizer

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#fitting and transforming train set, transforming test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[6]:


#Initializing a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#Predicting the test set and calculating accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[7]:


#confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:




