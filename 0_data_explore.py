# -*- coding: utf-8 -*-
"""
Exercise:
    Discover groups of employees that have more similar profiles than the rest and explain their statistical relevance. How many such groups could you find and why do they have similar profiles? Which profiles show more attrition and why?

"""
"""
Conditions:
    You have five days to return your completed exercise. Your work will be assessed on the maturity and comprehensiveness of the analysis, choice of the programming methodology, attention to detail and presentation of results. You must document your assumptions, limitations, thoughts in comments and print-logs, results in visualizations, etc. You must share your source codes which should be minimal, agile, intelligible, and following standard coding practices. Your results should be easily reproducible by executing the codes elsewhere. Good luck!

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_data = pd.read_csv('data/HR_comma_sep.csv')
#raw_data.info()
'''
[Observation on Results:]
    1. sample size: 14999
    2. features: 9
    3. target('left'): 1
    
Since features size (9) is much smaller than sample size (14999), it is good to know that over-fitting (which would be caused by relevantly close amounts of sample size and feature size) would NOT happen
'''
#%%
stat_cols = ['col_name','count','min','max','mean','std','nan']
def explore_nu_col(data,col_name):
    col_data = data[col_name]

    stat_count=len(col_data )
    stat_min=np.min(col_data )
    stat_max=np.max(col_data )
    stat_mean=np.mean(col_data )
    stat_std=np.std(col_data )
    stat_nan=np.sum(np.isnan(col_data ))
    stat = pd.DataFrame([[col_name,stat_count, stat_min,stat_max,stat_mean,stat_std,stat_nan]], columns=stat_cols)

    print('##############',col_name,'###############')
    print(stat)
    return stat
    
def nu_col_hist(data,col_name):
    col_data = data[col_name]
    plt.hist(col_data, bins=100)  # plt.hist passes it's arguments to np.histogram
    plt.title("Distribution of "+col_name)
    plt.show()
    # solution for interactive window Not Response issue if plot in the second window
#    plt.pause(100)
#    plt.close()
#%%
'''
explore numeric cols
'''
num_cols = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company']

num_cols_stat = pd.DataFrame([], columns=stat_cols)
for col_name in num_cols:
    stat = explore_nu_col(raw_data,col_name)
    num_cols_stat = num_cols_stat.append(stat)
    nu_col_hist(raw_data,col_name)
print(num_cols_stat)

'''
[Observation on Results:]
    1. no NaN values in any col
'''
#%%
'''
explore bool (0/1) cols
'''
bool_cols = ['Work_accident', 'left','promotion_last_5years']
stat_cols = ['col_name','count','0','1','nan']
def explore_bool_col(data,col_name):
    col_data = data[col_name]
    stat_count=len(col_data)
    stat_0=np.sum(col_data==0)
    stat_1=np.sum(col_data==1)
    stat_nan=np.sum(np.isnan(col_data))
    stat = pd.DataFrame([[col_name,stat_count,stat_0, stat_1,stat_nan]], columns=stat_cols)

    print('##############',col_name,'###############')
    print(stat)
    return stat
    
bool_cols_stat = pd.DataFrame([], columns=stat_cols)
for col_name in bool_cols:
    stat = explore_bool_col(raw_data, col_name)
    bool_cols_stat = bool_cols_stat.append(stat)
    
print(bool_cols_stat)

'''
[Observation on Results:]
    1. no NaN value in any col
    2. imbalance between 0s and 1s (imbalanced data could lead to classification bias problem)
        a. Work_accident: 1s: 14.46%
        b. left: 1s: 23.81%
        c. promotions_last_5years: 1s: 2.13%
        
[Actions]
    1. to solve the imbalance problem specifically for col "left",  two approachs will be tested:
        a. oversampling: replicate samples whose "left" value are 1 to the same amount of sample with "left" value 0. 
                if this process can not be finished in a reasonable time because of the increased size of data, use undersampling. However, keep in mind that Oversampling could increase the impact of incorrect samples on further data processing if these incorrect samples are replicated multiple times as well as their errors. 
        b. undersampling: undersampling would produce much smaller dataset for further data processing, but since it is more risky than oversampling since some of data are excluded from the further processing. 
'''
len(raw_data.columns)
#%%
'''
explore categorical cols
'''
cate_cols = ['sales','salary']
for col_name in cate_cols:
    print('##############',col_name,'###############')
    col_data = raw_data[col_name]
    uniq_values = col_data.unique()
    print(uniq_values)
    for uniq_value in uniq_values:
        print(uniq_value,end=',')
        print(np.sum(col_data==uniq_value))
        
'''
[Observation]
    1. "sales" col has 10 unique values
    2. "salary" col has 3 unique values

[Actions]
    1. use onehotencoder to transfer categorical cols into bool ones
        it is feasible to use onehotencoder here because there are only 10 features, and two categorical cols only have 13 unique values in total. Hence, eventually there will only be 21 features after onehotencoder which is not too large compare with the sample size (14999)
'''
