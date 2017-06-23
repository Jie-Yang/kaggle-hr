# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:40:00 2017

@author: jyang
"""

import pandas as pd


raw_data = pd.read_csv('data/HR_comma_sep.csv')
cate_cols = ['sales','salary']
encoded_data = pd.get_dummies(raw_data,columns=cate_cols)


#%%
'''
Balance the number of emplyees who left and who did not
'''

encoded_data_0 = encoded_data[encoded_data.left==0]
encoded_data_1 = encoded_data[encoded_data.left==1]
print('0s:',encoded_data_0.shape[0],'1s:',encoded_data_1.shape[0])
encoded_data_oversample_1 = encoded_data_1
for i in range(2):
    encoded_data_oversample_1 = encoded_data_oversample_1.append(encoded_data_1)

print('oversampling 0s:',encoded_data_0.shape[0],'1s:',encoded_data_oversample_1.shape[0])
balanced_data = encoded_data_0.append(encoded_data_oversample_1)

'''
[Result]
0s: 11428 1s: 3571
oversampling 0s: 11428 1s: 10713

balanced data have almost the same amount of 1s and 0s.
risk: bias and errors in from sample (left==1) are amplified by the oversampling procedure. 
'''

del encoded_data_0,encoded_data_1,i,encoded_data_oversample_1
#%%
'''

create a regression model targeted at col "left" from all the other cols.

'''


import numpy as np
from sklearn.linear_model import LogisticRegression

'''
Other regression model (ANN, LinearRegress, SVM, etc.) should also be examined as well as other configuration of LogisticRegression
'''

reg_Y_col_names = encoded_data.columns.drop('left')

reg_X = encoded_data[reg_Y_col_names]
reg_Y = encoded_data.left

model = LogisticRegression()
model.fit(reg_X,reg_Y)
pred_Y = model.predict(reg_X)

'''
validate the result: matrix
'''

from sklearn.metrics import accuracy_score
score = accuracy_score(reg_Y, pred_Y)
print("accuracy_score:",score)

'''
accuracy_score: 0.791919461297
'''
#%%
'''
For each col, visulize prob of employee will leave the company at different col value. values of the rest cols will be set to the median value in that col as constant.
risk: when set the other col's value to a constant, different strategies (e.g. median, middle, max , min) could lead to different results. 
'''
import matplotlib.pyplot as plt
for figure_i, col_name in enumerate(reg_Y_col_names):
    col_data = reg_X[col_name]
    col_data_min = min(col_data)
    col_data_max = max(col_data)
    col_samples = np.arange(col_data_min,col_data_max,(col_data_max-col_data_min)/10)
    dummy_samples = pd.DataFrame({col_name:col_samples},index=col_samples)
    for other_col_name in reg_Y_col_names.drop(col_name):
        dummy_col = np.ones(10)*np.median(reg_X[other_col_name])
        dummy_samples[other_col_name]=dummy_col
    probas = model.predict_proba(dummy_samples)
    
    plt.figure(figure_i)
    plt.plot(col_samples,probas[:,0],'g-o',label='left '+str(model.classes_[0]))
    plt.plot(col_samples,probas[:,1],'r-s',label='left '+str(model.classes_[1]))
    plt.legend()
    plt.xlabel(col_name)
    plt.ylabel('Attrition Prob.')
    plt.ylim([-.1,1.1])
    plt.show()

    
