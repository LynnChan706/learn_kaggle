#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'ccchen706@126.com'
'''

import pandas as pd
import os
import random
import numpy as np

from numpy import random as nr

class CsvReader():
    def __init__(self,dic=''):
        if dic is not '':
            self.csv_dict=dic+'/'
        else:
            self.csv_dict='./'


    def read_data(self,file_name):
        file_path_name=self.csv_dict+file_name
        read_data=pd.read_csv(file_path_name)
        # print(read_data.head())
        return read_data

    def read_data_chunk(self,file_name,chunkSize=5):
        file_path_name=self.csv_dict+file_name
        read_data_chunk=pd.read_csv(file_path_name,chunksize=chunkSize)
        # print(read_data.head())
        return read_data_chunk

    def read_data_with_number(self,file_name,read_num=500000):

        print('start')
        x = np.arange(1,200000000)

        skiprow = nr.choice(x, 200000000-read_num,replace = False)
        print('end')
        file_path_name=self.csv_dict+file_name
        read_data = pd.read_csv(file_path_name, nrows=read_num, skiprows=skiprow)
        print('read data end')
        return read_data

    def read_data_with_random(self,file_name,read_num=2000000):
        skiprow = random.randint(100000000,200000000)
        file_path_name=self.csv_dict+file_name
        read_data = pd.read_csv(file_path_name,nrows =read_num, skiprows=range(1, skiprow))
        return read_data



    def write_data_without_index(self,res_data,file_name,columns=None,index_name='',index_start=0,):
        file_path_name = self.csv_dict + file_name
        res_data_frame = pd.DataFrame(res_data,columns=(columns))
        res_data_frame.index=index_start
        res_data_frame.index.name=index_name
        if os.path.exists(file_path_name):
            res_data_frame.to_csv(file_path_name)
        else:
            with open(file_path_name, 'a') as f:
                res_data_frame.to_csv(f,header=False)

    def write_data_with_index(self,res_data,index,file_name,columns=None,index_name='',):
        file_path_name = self.csv_dict + file_name
        res_data_frame = pd.DataFrame(res_data,index,columns=(columns))
        res_data_frame.index.name=index_name

        if os.path.exists(file_path_name):
            with open(file_path_name, 'a') as f:
                res_data_frame.to_csv(f, header=False)
        else:
            res_data_frame.to_csv(file_path_name)


    def write_data(self,res_data,file_name,columns=None):
        file_path_name = self.csv_dict + file_name
        res_data_frame = pd.DataFrame(res_data,index=None,columns=(columns))
        # print(res_data_frame.head())
        res_data_frame.to_csv(file_path_name)
