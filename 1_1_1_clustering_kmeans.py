# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:34:48 2017

@author: jyang
"""

import pandas as pd


raw_data = pd.read_csv('data/HR_comma_sep.csv')
cate_cols = ['sales','salary']
encoded_data = pd.get_dummies(raw_data,columns=cate_cols)


#%%
'''
Nomarliza data
'''
from sklearn.preprocessing import StandardScaler
norm_data= StandardScaler().fit_transform(encoded_data)
#%%

#%%
'''
KMeans K selection: Silhouette

The best value is 1 and the worst value is -1. 
Values near 0 indicate overlapping clusters. 
Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
'''

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans

    
n_clusters = []
n_sil= []
n_inertia = []
for k in range(2,30,1):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(norm_data)
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    
    sil = silhouette_score(norm_data, labels, metric='euclidean')
    
    print(k, sil,inertia)
    n_sil.append(sil)
    n_clusters.append(k)
    n_inertia.append(inertia)
    
    if sil == 1:
        break
    
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(n_clusters,n_inertia)
plt.xlabel('k')
plt.ylabel('intertia')

plt.figure(2)
plt.plot(n_clusters,n_sil)
plt.xlabel('k')
plt.ylabel('silhouette')
#%%
'''

k-means can not cluster data propriately regardless of different k selection (from 2 to 30)


2 0.0984709348877 287361.472708
3 0.0980368004842 272494.097332
4 0.126023387523 254815.143498
5 0.103990723218 244361.285113
6 0.154398077489 229060.691599
7 0.17735969845 212911.227486
8 0.201745588494 200494.194281
9 0.242320258345 179858.090577
10 0.272709499597 163396.524989
11 0.238895605033 163548.877297
12 0.242267461485 149072.861085
13 0.248561652949 140787.020829
14 0.24648329851 134103.003671
15 0.253504870597 130437.699796
16 0.252781440839 124583.202461
17 0.262854426235 122497.659715
18 0.252958729368 119993.282353
19 0.251829682842 116825.43398
20 0.257315663845 114831.839415
21 0.255510398363 113323.671223
22 0.252370158443 111827.313242
23 0.251998664439 109585.473497
24 0.260203660285 108301.922681
25 0.256952260537 105723.599152
26 0.26670401965 104117.107013
27 0.254808966074 103406.499372
28 0.263230385879 100788.36578
29 0.248157545696 100696.021285


[Result]

k=10 give a suitable silhouette value with the smallest k value.
'''

#%%    
k = 10
kmeans = KMeans(n_clusters=k, random_state=0).fit(norm_data)
labels = kmeans.labels_
inertia = kmeans.inertia_

sil = silhouette_score(norm_data, labels, metric='euclidean')

print(k, sil,inertia)


    