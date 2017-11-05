# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 08:41:02 2017

@author: Thibault
"""
import os 
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from . import PreProcessing

import dill as pickle

from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")

PATH_DATA = "/home/thibault/Documents/Flask/Flask_test/Data/"
PATH_MODEL= "/home/thibault/Documents/Flask/Flask_test/Model/"
FILENAME = 'model_v1.pk'

data = pd.read_csv(PATH_DATA + 'train.csv')

list(data.columns)


for _ in data.columns:
    print("The number of null values in:{} == {}".format(_, data[_].isnull().sum()))
    
pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome',\
            'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']

X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['Loan_Status'], \
                                                    test_size=0.25, random_state=42)
    
y_train = y_train.replace({'Y':1, 'N':0}).as_matrix()
y_test = y_test.replace({'Y':1, 'N':0}).as_matrix()

pipe = make_pipeline(PreProcessing(),
                    RandomForestClassifier())

param_grid = {"randomforestclassifier__n_estimators" : [10, 20, 30],
             "randomforestclassifier__max_depth" : [None, 6, 8, 10],
             "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20], 
             "randomforestclassifier__min_impurity_split": [0.1, 0.2, 0.3]}


grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)

grid.fit(X_train, y_train)

print("Validation set score: {:.2f}".format(grid.score(X_test, y_test)))


test_df = pd.read_csv(PATH_DATA + 'test.csv', encoding="utf-8-sig")
test_df = test_df.head()

grid.predict(test_df)# -*- coding: utf-8 -*-

with open(PATH_MODEL + FILENAME, 'wb') as file:
	pickle.dump(grid, file)
    