# -*- coding: utf-8 -*-
"""analisis_sentimen_svm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1A6JzIQgsVYX-TmLAJmBp_-ONXP0mi-PK
"""

import pandas as pd
import re
import nltk
import Sastrawi

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #import Indonesian Stemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# pip install Sastrawi

"""Dataset"""

from google.colab import drive
drive.mount('/content/drive')

dataset = 'drive/MyDrive/sentimen/dataset_tweet_sentimen.csv'
data = pd.read_csv(dataset)
data

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

data['tweet'][1]

lemma = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('indonesian','english'))                       #
def Cleantweet(txt):
  txt = re.sub(r'http\s+', ' ', txt)
  txt = re.sub('[^a-zA-Z]', ' ', txt)
  txt = str(txt).lower()
  txt = word_tokenize(txt)
  txt = [item for item in txt if item not in stop_words]
  txt = [stemmer.stem(i) for i in txt]
  # txt = [lemma.lemmatize(word=w,pos='v') for w in txt]
  txt = [i for i in txt if len(i) > 2]
  txt = ' '.join(txt)
  return txt
data['Cleantweet'] = data['tweet'].apply(Cleantweet)

data



"""Perform SVM"""

# split x dan y
x = data['Cleantweet']
y = data['Sentimen']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test

# perform count vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(x_train)

# x_train
x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)

x_train.toarray()

from sklearn import svm
#Create a svm Classifier
clf = svm.SVC(kernel='linear') 
# Linear Kernel#Train the model using the training sets
clf.fit(x_train, y_train) 
#Predict the response for test dataset
y_pred = clf.predict(x_test)

y_pred

y_test

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

y_pred = clf.predict(x_test)
print ('Accuracy of SVM classifier on test set: {:.2f}' .format(clf.score(x_test, y_test)))
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))