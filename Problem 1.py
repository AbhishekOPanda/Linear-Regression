#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# In[71]:


feature_names = {i:label for i,label in zip(range(4),
                  ('sepal length in cm',
                  'sepal width in cm',
                  'petal length in cm',
                  'petal width in cm', ))}


# In[72]:


# Reading data to pandas data frame and splitting them as unique class labels in one class
df = pd.read_csv('iris.data', header=None, sep=',')


# In[73]:


df.columns = [l for i,l in sorted(feature_names.items())] + ['class label']
df.dropna(axis=0,how="all",inplace=True)
df.describe()


# In[74]:


df.tail()


# In[75]:


X= df.iloc[:,0:4]
y = df.iloc[:,-1]


# In[76]:


class_label = df['class label'].unique()

setosa = df[df['class label'] == class_label[0]]

versicolor = df[df['class label'] == class_label[1]]

virginica = df[df['class label'] == class_label[2]]


# In[77]:


frame12 =  [setosa,versicolor]
frame13 = [setosa,virginica]
frame14 = [versicolor,virginica]

df_sVer = pd.concat(frame12)
df_sVir = pd.concat(frame13)
df_verVir = pd.concat(frame14)


# In[78]:


#Splitting the train and test data
X_12 = df_sVer.iloc[:, 0:4].values
y_12 = df_sVer.iloc[:, -1].values

X_12_train, X_12_test, y_12_train, y_12_test = train_test_split(X_12, y_12, test_size = 0.2, random_state = 42)


# In[79]:


sc = StandardScaler()
X_12_train = sc.fit_transform(X_12_train)
X_12_test = sc.transform(X_12_test)


# In[80]:


#LDA for classes Setosa and Versicolor
lda = LDA()
X_12_train = lda.fit_transform(X_12_train, y_12_train)
X_12_test = lda.transform(X_12_test)


# In[81]:


#Logistic Regression for Setosa and Versicolor
classifier = LogisticRegression(random_state = 42)
classifier.fit(X_12_train, y_12_train)


# In[82]:


y_12_pred = classifier.predict(X_12_test)
confMat = confusion_matrix(y_12_test, y_12_pred)
print(confMat)
print('Accuracy : ' + str(accuracy_score(y_12_test, y_12_pred)))


# In[83]:


#Splitting the train and test data
X_13 = df_sVir.iloc[:, 0:4].values
y_13 = df_sVir.iloc[:, -1].values

X_13_train, X_13_test, y_13_train, y_13_test = train_test_split(X_13, y_13, test_size = 0.2, random_state = 42)

sc = StandardScaler()
X_13_train = sc.fit_transform(X_13_train)
X_13_test = sc.transform(X_13_test)

#LDA for classes Setosa and Virginica
lda = LDA()
X_13_train = lda.fit_transform(X_13_train, y_13_train)
X_13_test = lda.transform(X_13_test)

#Logistic Regression for Setosa and Virginica
classifier = LogisticRegression(random_state = 42)
classifier.fit(X_13_train, y_13_train)

y_13_pred = classifier.predict(X_13_test)
confMat = confusion_matrix(y_13_test, y_13_pred)
print(confMat)
print('Accuracy : ' + str(accuracy_score(y_13_test, y_13_pred)))


# In[84]:


#Splitting the train and test data
X_23 = df_verVir.iloc[:, 0:4].values
y_23 = df_verVir.iloc[:, -1].values

X_23_train, X_23_test, y_23_train, y_23_test = train_test_split(X_23, y_23, test_size = 0.2, random_state = 42)

sc = StandardScaler()
X_23_train = sc.fit_transform(X_23_train)
X_23_test = sc.transform(X_23_test)

#LDA for classes Versicolor and Virginica
lda = LDA()
X_23_train = lda.fit_transform(X_23_train, y_23_train)
X_23_test = lda.transform(X_23_test)

#Logistic Regression for Versicolor and Virginica
classifier = LogisticRegression(random_state = 42)
classifier.fit(X_23_train, y_23_train)

y_23_pred = classifier.predict(X_23_test)
confMat = confusion_matrix(y_23_test, y_23_pred)
print(confMat)
print('Accuracy : ' + str(accuracy_score(y_23_test, y_23_pred)))


# In[ ]:




