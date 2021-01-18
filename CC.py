#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:47:12 2021

@author: muselin
"""
import pandas as pd
import statistics 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso



file = pd.read_csv("/Users/muselin/Desktop/Python/Credit Card/crx.data", header=None)
cc_apps = pd.DataFrame(file)

cc_apps = cc_apps.drop([11,13], axis=1)
cc_apps = cc_apps.replace(to_replace = '?', value=np.NaN)
# print(cc_apps.head())
# print(cc_apps.isna().sum(axis=0))

flt = [1, 2, 7, 10, 14]
for i in flt:
    cc_apps[i] = cc_apps[i].astype(float)
    # print(cc_apps[i].dtype)

a = cc_apps[1].sum()
b = len(cc_apps[1])
avg = a/b
cc_apps.fillna({1: avg}, inplace=True)

# print(cc_apps.isna().sum())
# print(cc_apps.head())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cc_apps.columns:
    # Compare if the dtype is object
    if cc_apps[col].dtypes =="object":
    # Use LabelEncoder to do the numeric transformation
        cc_apps[col]=le.fit_transform(cc_apps[col].astype(str))
print(cc_apps.head())
print("=====================")
cc_apps = cc_apps.to_numpy()

# Segregate features and labels into separate variables
X,y = cc_apps[:,0:12] , cc_apps[:,13]

X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.2,
                                random_state=42)

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)
# print(rescaledX_train[:,0])


from sklearn.linear_model import LogisticRegression
# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)
# Get the accuracy score of logreg model and print it

from sklearn.metrics import confusion_matrix, classification_report
print("LOG = ",logreg.score(X_test, y_test))
print("Scaled LOG = ",logreg.score(rescaledX_test, y_test))
print(confusion_matrix(y_test, y_pred.round()))

from sklearn.model_selection import GridSearchCV
# Define the grid of values for tol and max_iter
tol = [0.1, 0.01, 0.001, 0.0001]
max_iter = [50, 100, 150, 200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict({'tol':tol, 'max_iter':max_iter})

grid_model = GridSearchCV(logreg, param_grid, cv=5)
grid_model_result = grid_model.fit(rescaledX_train, y_train)
print("BEST SCORE", grid_model_result.best_score_)
print("Best PARAMS", grid_model_result.best_params_)






































