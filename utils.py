# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:29:16 2017

@author: jyang
"""

import pickle
    
def save_variable(var,file_name):
    pkl_file = open(file_name, 'wb')
    pickle.dump(var, pkl_file, -1)
    pkl_file.close()
    
def read_variable(file_name):
    pkl_file = open(file_name, 'rb')
    var = pickle.load(pkl_file)
    pkl_file.close()
    return var
