#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'ccchen706@126.com'
'''

import csv_reader
import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb

from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn import  metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Perceptron,SGDClassifier,PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import gc


Data_reader = csv_reader.CsvReader('../DataSet/TalkingData')
Data_writer = csv_reader.CsvReader('../output')

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[xgtrain, xgvalid],
                     valid_names=['train','valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=50,
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1

def read_all_data_with_persent(train_data,persent=2.5):


    res_data = pd.DataFrame()

    data_with_label = train_data.loc[train_data['is_attributed'] == 1]
    data_without_label = train_data.loc[train_data['is_attributed'] == 0]

    persent_v = (len(data_with_label) / len(data_without_label)) * persent

    if persent_v <= 0:

        X_train = train_data.iloc[:, 1:5]
        Y_train = train_data.iloc[:, 7:8]

        return X_train, Y_train

    else:
        if persent_v < 1:
            df2 = data_without_label.sample(frac=persent_v)
        else:
            df2 = data_without_label

        res_data = pd.concat([df2, data_with_label], ignore_index=True)

        X_train = res_data.iloc[:, 1:5]
        Y_train = res_data.iloc[:, 7:8]

        return X_train, Y_train


def read_all_data_with_persent_train(train_data,persent=2.5):


    res_data = pd.DataFrame()

    data_with_label = train_data.loc[train_data['is_attributed'] == 1]
    data_without_label = train_data.loc[train_data['is_attributed'] == 0]

    persent_v = (len(data_with_label) / len(data_without_label)) * persent

    if persent_v <= 0:

        X_train = train_data.iloc[:, 1:8]

        return X_train

    else:
        if persent_v < 1:
            df2 = data_without_label.sample(frac=persent_v)
        else:
            df2 = data_without_label

        res_data = pd.concat([df2, data_with_label], ignore_index=True)

        X_train = res_data.iloc[:, 1:8]

        return X_train


def read_data_for_test(test_data):

    X_test = test_data.iloc[:, 1:5]
    Y_test = test_data.iloc[:, 7:8]

    return X_test,Y_test

def get_next_test_data(test_data_chunk):
    for data_chunk in test_data_chunk:
        id = data_chunk.iloc[:, :1]
        X_test = data_chunk.iloc[:, 2:6]
        yield id,X_test

print("Training...")

target = 'is_attributed'
predictors = ['ip','app','device','os', 'channel']
categorical = ['ip','app','device','os', 'channel']

train_data = Data_reader.read_data_with_number('train.csv', 1300000)

test_data = Data_reader.read_data_with_random('train.csv')

(X_test, Y_test) = read_data_for_test(test_data)

persent = 2.3

X_train_text= read_all_data_with_persent_train(train_data,persent)

params = {
    'learning_rate': 0.1,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 1400,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 4,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': .7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99 # because training data is extremely unbalanced
}

bst = lgb_modelfit_nocv(params,
                        X_train_text,
                        test_data,
                        predictors,
                        target,
                        objective='binary',
                        metrics='auc',
                        early_stopping_rounds=30,
                        verbose_eval=True,
                        num_boost_round=500,
                        categorical_features=categorical)

del X_train_text
del test_data


# print("Predicting...")
# sub['is_attributed'] = bst.predict(test_df[predictors])
# print("writing...")
# sub.to_csv('sub_lgb_balanced99.csv',index=False)
# print("done...")
# print(sub.info())