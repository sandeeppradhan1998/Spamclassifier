# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 02:50:02 2019

@author: Dilip
"""

import nltk
import re 
import pandas as pd

#import dataset
messages= pd.read_csv('smsspamcollection', sep='\t', names=['label', 'message'])
              
#cleaning the data
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

pc=PorterStemmer()
lm=WordNetLemmatizer()

corpus=[]
for i in range(0, len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[lm.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

#import TDIDF
from sklearn.feature_extraction.text import TfidfVectorizer
Td=TfidfVectorizer()
x=Td.fit_transform(corpus).toarray()

y=messages.iloc[:,0].values

#creating dummi variables
y=pd.get_dummies(y)
y=y.iloc[:,1].values

#train the set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=0)


#train the model
from sklearn.naive_bayes import MultinomialNB
spame_detect_model=MultinomialNB().fit(x_train,y_train)


#predict the value
y_pred=spame_detect_model.predict(x_test)


from sklearn.metrics import confusion_matrix, accuracy_score
confusion=confusion_matrix(y_test, y_pred)
print('\n')
assuracy=accuracy_score(y_test, y_pred)


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
    

           