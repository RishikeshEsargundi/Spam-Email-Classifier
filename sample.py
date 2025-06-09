import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("emails.csv")
data.head()

data.shape
data['text'][0]
data['spam'].value_counts()

import seaborn as sns
sns.countplot(data['spam'])

data.duplicated().sum()
data.drop_duplicates(inplace=True)

data.duplicated().sum()
data.isnull().sum()
data.shape

5728 - 33

sns.countplot(data['spam'])
data['spam'].value_counts()

X = data['text'].values
y = data['spam'].values

y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 , random_state= 0)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
nb = MultinomialNB()

pipe = make_pipeline(cv, nb)
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test)

email = ['Hey i am Elon Musk. Get a brand new car from Tesla']
pipe.predict(email)

import pickle
pickle.dump(pipe, open("Naive_model.pkl",'wb'))
