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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer

Data_reader = csv_reader.CsvReader('../DataSet/TalkingData')
Data_writer = csv_reader.CsvReader('../output')


def get_next_data(train_data_chunk):
    for data_chunk in train_data_chunk:
        X_train = data_chunk.iloc[:, :5]
        Y_train = data_chunk.iloc[:, 7:8]
        yield X_train, Y_train

def get_next_test_data(test_data_chunk):
    for data_chunk in test_data_chunk:
        id = data_chunk.iloc[:, :1]
        X_test = data_chunk.iloc[:, 2:7]
        yield id,X_test


if __name__ == '__main__':

    train_data_chunk = Data_reader.read_data_chunk('train_sample.csv',20000)

    test_data_chunk = Data_reader.read_data_chunk('test.csv',20000)

    get_next_test_data_chunk = get_next_test_data(test_data_chunk)

    get_next_data_chunk = get_next_data(train_data_chunk)

    predict = SGDClassifier()
    all_classes = np.array([0, 1])

    for i, (X_train_text, y_train) in enumerate(get_next_data_chunk):
        # pd.DataFrame

        train_data, vali_data, train_labels, vali_labels = \
            train_test_split(X_train_text, y_train, train_size=0.90, random_state=1)

        predict.partial_fit(train_data, train_labels, classes=all_classes)

        print('IN :',i,'acc :{}'.format(predict.score(vali_data, vali_labels)))

    for i, (id,X_test_text) in enumerate(get_next_data_chunk):
        # pd.DataFrame
        print('')




