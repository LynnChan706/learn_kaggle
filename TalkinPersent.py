#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'ccchen706@126.com'
'''


import csv_reader
import numpy as np


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

Data_reader = csv_reader.CsvReader('../DataSet/TalkingData')
Data_writer = csv_reader.CsvReader('../output')
#
# clfdir={'MNB':MultinomialNB(),
#         'BNB':BernoulliNB(),
#         'PT':Perceptron(),
#         'SG':SGDClassifier(),
#         'PAC':PassiveAggressiveClassifier(),
#         'DT':DecisionTreeClassifier(),
#         'SGD':SGDClassifier(),
#         'LSVC':LinearSVC(),
#         'RVC':linear_model.RidgeCV(),
#         'MLP': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=5)}


clfdir={
        'DT':DecisionTreeClassifier(),
        'RFC':RandomForestClassifier(),
        'EFC':ExtraTreesClassifier(random_state=5),
        'ADA':AdaBoostClassifier(),
        'GBC':GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
        }




def read_data_with_persent(train_data,number=200000,persent=2.5):
    res_data = pd.DataFrame()
    test_data = train_data.iloc[:number]
    data_with_label = train_data.loc[train_data['is_attributed'] == 1]
    data_without_label = train_data.loc[train_data['is_attributed'] == 0]

    persent_v = (len(data_with_label) / len(data_without_label)) * persent

    if persent_v <= 0:

        X_train = train_data.iloc[number:2*number, 1:5]
        Y_train = train_data.iloc[number:2*number, 7:8]

        X_test = test_data.iloc[:, 1:5]
        Y_test = test_data.iloc[:, 7:8]

        return X_train, Y_train,X_test,Y_test

    else:
        if persent_v < 1:
            df2 = data_without_label.sample(frac=persent_v)
        else:
            df2 = data_without_label

        res_data = pd.concat([df2, data_with_label], ignore_index=True)

        X_train = res_data.iloc[:, 1:5]
        Y_train = res_data.iloc[:, 7:8]

        X_test = test_data.iloc[:, 1:5]
        Y_test = test_data.iloc[:, 7:8]

        return X_train, Y_train,X_test,Y_test


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


def read_data_for_test(test_data):

    X_test = test_data.iloc[:, 1:5]
    Y_test = test_data.iloc[:, 7:8]

    return X_test,Y_test

def get_next_test_data(test_data_chunk):
    for data_chunk in test_data_chunk:
        id = data_chunk.iloc[:, :1]
        X_test = data_chunk.iloc[:, 2:6]
        yield id,X_test


if __name__ == '__main__':

    train_data = Data_reader.read_data_with_number('train.csv',1300000)

    test_data = Data_reader.read_data_with_random('train.csv')

    (X_test, Y_test)=read_data_for_test(test_data)

    # for j in range(1,20):
    #
    #     train_data = Data_reader.read_data_with_number('train.csv',j*100000)
    #     print('train data:',j*100000)

    for i in range(17,30):
        persent = float(i)/10
        print('persent:',persent)
        (X_train_text, y_train) = read_all_data_with_persent(train_data,persent)

        for kv in clfdir.items():
            predict =kv[1]

            key=kv[0]
            print('use',key)
            all_classes = np.array([0, 1])
            socer = []
            if hasattr(predict,'predict_proba'):

                predict.fit(X_train_text, y_train)

                pre_res_data = (np.array(predict.predict_proba(X_test)))[:, 1]

                socer_roc = metrics.roc_auc_score(Y_test.values, pre_res_data)

                print('test data res::', socer_roc)

                if socer_roc >= 0.940:
                    test_data_chunk = Data_reader.read_data_chunk('test.csv', 20000)
                    get_next_test_data_chunk = get_next_test_data(test_data_chunk)

                    for i, (id, X_test_text) in enumerate(get_next_test_data_chunk):
                        res_data = np.array(predict.predict_proba(X_test_text))[:, 1]
                        print(id.values.flatten()[0],end=' | ')
                        if i%20 == 1:
                            print('')
                        Data_writer.write_data_with_index(res_data, id.values.flatten(),
                                                          'TalkinSubmission'+key+str(socer_roc).replace('.','_')[:5]+'.csv',
                                                          columns=('is_attributed',),index_name='click_id')
            else:
                print('has not predict_proba')
