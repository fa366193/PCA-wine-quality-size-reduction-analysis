#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import os


# In[3]:


#Reading CSV Files
df = pd.read_csv("Desktop/winequalityN.csv")
print(df)


# In[4]:


def check_dt(dataframe):
    print("SHAPE".center(70, "-"))
    print(dataframe.shape)
    print("TYPE".center(70, "-"))
    print(dataframe.dtypes)
    print("INFO".center(70, "-"))
    print(dataframe.info())
    print("NA".center(70, "-"))
    print(dataframe.isnull().sum())
    print("DESCRIBE".center(70, "-"))
    print(dataframe.describe().T)
    print("NUNIQUE".center(70, "-"))
    print(dataframe.nunique())
check_dt(df)


# In[6]:


#Preprocessing the data
type = df["type"].values
print(type)
type.shape


# In[7]:


le = preprocessing.LabelEncoder()
type1 = le.fit_transform(df["type"])
type1 = pd.DataFrame(data=type1)
type1.columns=["type1"]
type1


# In[8]:


df1=pd.concat([df,type1],axis=1)
df1.drop("type",axis=1,inplace=True)
df1.head()


# In[9]:


df1.dropna(inplace=True)
df1.isnull().sum()


# In[10]:


#Fragmentation and normalization
x=df1.iloc[:,0:12].values
x.shape


# In[11]:


y=df1.iloc[:,-1:].values
y.shape


# In[12]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)


# In[13]:


#Model creation
pc=PCA(n_components=2)
X_train1=pc.fit_transform(X_train)
X_test1=pc.transform(X_test)


# In[14]:


#LR model without PCA
lr=LogisticRegression()
reg=lr.fit(X_train,y_train)
pred=lr.predict(X_test)


# In[15]:


#LR model after PCA is done
reg1=lr.fit(X_train1,y_train)
pred1=lr.predict(X_test1)


# In[16]:


#Confusion Matrix
print('Actual / Not PCA')
cm=confusion_matrix(y_test,pred)
print(cm)


# In[17]:


print('Actual / PCA')
cm1=confusion_matrix(y_test,pred1)
print(cm1)


# In[18]:


print("Not PCA / PCA")
cm2=confusion_matrix(pred,pred1)
print(cm2)


# In[ ]:




