#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'chenliang@moutum.com'
'''

import csv_reader
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer

from sklearn.neural_network import MLPClassifier



Data_reader = csv_reader.CsvReader('../DataSet')
Data_writer = csv_reader.CsvReader('../output')

if __name__ == '__main__':

    train_data = Data_reader.read_data('train.csv')
    test_data = Data_reader.read_data('test.csv')

    # X_train = train_data.iloc[:, 1:]
    # Y_train = train_data.iloc[:, 0]
    # X_test = test_data

    train_data_bin = Binarizer(threshold=127).fit_transform(train_data)
    X_test_bin = Binarizer(threshold=127).fit_transform(test_data)
    X_train = pd.DataFrame(train_data_bin).iloc[:, 1:]
    Y_train = train_data.iloc[:, 0]
    X_test = pd.DataFrame(X_test_bin)

    train_images, vali_images, train_labels, vali_labels = \
        train_test_split(X_train, Y_train, train_size=0.95,random_state=1)
    print('start predict')

    predict = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(100,), random_state=5)

    predict.fit(train_images,train_labels)
    print('acc:{}'.format(predict.score(train_images, train_labels)))
    print('acc:{}'.format(predict.score(vali_images, vali_labels)))
    res_data = np.array(predict.predict(X_test))
    Data_writer.write_data_without_index(res_data, 'submissionMLP.csv', columns=('Label',),index_name='ImageId')
