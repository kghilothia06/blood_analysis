# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:11:47 2020

@author: KUSH
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('blood-train.csv')


df_train.shape
df_train.head()
df_train.tail()
df_train.describe()



#data preprocessing
#renaming unnamed colunm
df_train.rename(columns={"Unnamed: 0":"Donor_id"},inplace=True)
df_train.head()


#checking null values
df_train.isnull()

#no null values are present

#getting info of datasets
df_train.info()
print("\n-------------------------\n")



#finding correlations
train_corr = df_train.corr()
sns.heatmap(train_corr)


#DATA PREPROCESSING
# Training data
X_train = df_train.iloc[:,[1,2,3,4]].values
y_train = df_train.iloc[:,-1].values

X_train
y_train

#Feature Scaling
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)

X_train

#applying random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train,y_train)
score = rf.score(X_train,y_train)
score
#92.01% is the performance of random forest classifier on training set  


