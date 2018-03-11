#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'ccchen706@126.com'
'''


import csv_reader
import numpy as np

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR

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

from sklearn.linear_model import Perceptron,SGDClassifier,PassiveAggressiveClassifier

Data_reader = csv_reader.CsvReader('../DataSet/TalkingData')
Data_writer = csv_reader.CsvReader('../output')

clfdir={'MNB':MultinomialNB(),
        'BNB':BernoulliNB(),
        'PT':Perceptron(),
        'SG':SGDClassifier(),
        'PAC':PassiveAggressiveClassifier()}


# clfdir={'MNB':MultinomialNB(),
#         'BNB':BernoulliNB()}


def get_next_data(train_data_chunk,persent=1,number=20000):
    print('get_next_data ')
    res_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for data_chunk in train_data_chunk:
        if test_data.empty:
            test_data = data_chunk
        else:
            if random.randint(0, 10) > 2:
                data_with_label=data_chunk.loc[data_chunk['is_attributed']==1]
                data_without_label = data_chunk.loc[data_chunk['is_attributed'] == 0]
                persent_v=(len(data_with_label)/len(data_without_label))*persent
                if persent_v<1:
                    df2 = data_without_label.sample(frac=persent_v)
                else:
                    df2 = data_without_label
                if res_data.empty:
                    res_data = pd.concat([df2,data_with_label], ignore_index=True)
                else:
                    res_data = pd.concat([res_data,df2, data_with_label], ignore_index=True)

                if len(res_data) > number:
                    X_train = res_data.iloc[:number, 1:5]
                    Y_train = res_data.iloc[:number, 7:8]

                    X_test=test_data.iloc[:number, 1:5]
                    Y_test=test_data.iloc[:number, 7:8]

                    yield X_train, Y_train,X_test,Y_test
            else:
                pass

def get_next_data_without_persent(train_data_chunk,number=2000):
    for data_chunk in train_data_chunk:
        X_train = data_chunk.iloc[:number, 1:5]
        Y_train = data_chunk.iloc[:number, 7:8]
        yield X_train,Y_train

def get_next_test_data(test_data_chunk):
    for data_chunk in test_data_chunk:
        id = data_chunk.iloc[:, :1]
        X_test = data_chunk.iloc[:, 2:6]
        yield id,X_test


if __name__ == '__main__':

    train_data_chunk = Data_reader.read_data_chunk('train.csv',40000)
    get_next_data_chunk = get_next_data(train_data_chunk)
    for kv in clfdir.items():
        predict =kv[1]
        key=kv[0]
        print('use',key)
        all_classes = np.array([0, 1])

        socer = []

        for i, (X_train_text, y_train,X_test,Y_test) in enumerate(get_next_data_chunk):
            # train_data, vali_data, train_labels, vali_labels = \
            #     train_test_split(X_train_text, y_train, train_size=0.90, random_state=1)
            predict.partial_fit(X_train_text, y_train, classes=all_classes)
            # vali_data = get_next_data_without_persent(train_data_chunk)
            # pre_res_data = (np.array(predict.predict(X_test)))[:,1]
            pre_res_data = (np.array(predict.predict(X_test)))
            # print('shape',pre_res_data.shape)
            idx = 0
            idx_diff = 0
            # for inum,v in enumerate(pre_res_data):
            #     print(v)
            #     if v!=Y_test.values.flatten()[inum]:
            #         idx_diff+=1
            #         if v == 1:
            #             idx+=1
            socer_roc=metrics.roc_auc_score(Y_test.values,pre_res_data)

            print('test data res::',socer_roc)

            socer.append(socer_roc)

            if i > 10:
                if (socer[-1]+socer[-2]+socer[-3]) <= (socer[-4]+socer[-5]+socer[-6]):
                    print('train over :',socer[-1])
                    break

        # print('use:',key,'value:',socer[-1],socer[-2],socer[-3],(socer[-1]+socer[-2]+socer[-3])/3)

        end_p = (socer[-1]+socer[-2]+socer[-3])/3

        print('end_p',end_p)

        if end_p >= 0.972:

            test_data_chunk = Data_reader.read_data_chunk('test.csv', 20000)
            get_next_test_data_chunk = get_next_test_data(test_data_chunk)

            for i, (id, X_test_text) in enumerate(get_next_test_data_chunk):
                res_data = np.array(predict.predict(X_test_text))
                print(id.values.flatten()[0],end=' | ')
                if i%20 == 1:
                    print('')
                Data_writer.write_data_with_index(res_data, id.values.flatten(),
                                                  'TalkinSubmission'+key+str(end_p).replace('.','_')[:5]+'.csv',
                                                  columns=('is_attributed',),index_name='click_id')



