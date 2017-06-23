# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:10:33 2017

@author: jyang
"""




import pandas as pd


raw_data = pd.read_csv('data/HR_comma_sep.csv')
cate_cols = ['sales','salary']
encoded_data = pd.get_dummies(raw_data,columns=cate_cols)


#%%
'''
Normalize the result
'''
from sklearn.preprocessing import StandardScaler
norm_data= StandardScaler().fit_transform(encoded_data)

#%%

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

n_components=norm_data.shape[1]

pca = PCA(n_components=n_components)
pca.fit(norm_data)

pca_variance = pca.explained_variance_ratio_

#%
plt.title('Accumulated PCA Variance')
plt.plot(pca_variance.cumsum())
plt.xlabel('PC nu.')
plt.ylabel('Accumulated Variance')

#%%
pcs = pca.transform(norm_data)

pc_2d = pcs[:,:2]

plt.title('Top 2 PCs')
plt.scatter(pc_2d[:,0],pc_2d[:,1])
plt.xlabel('1st PC')
plt.ylabel('2nd PC')

#%%
'''
[result]

based on the top 2PCs plot, there are two clear groups of employees.
'''

#%%
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

    
n_clusters = []
n_sil= []
n_inertia = []
for k in range(2,10,1):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pc_2d)
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    sil = silhouette_score(pc_2d, labels, metric='euclidean')
    
    print(k,inertia, sil)
    n_sil.append(sil)
    n_clusters.append(k)
    n_inertia.append(inertia)
    
    if sil == 1:
        break
#%%  
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(n_clusters,n_inertia)
plt.xlabel('KMeans k')
plt.ylabel('inertia')


plt.figure(2)
plt.plot(n_clusters,n_sil)
plt.xlabel('KMeans k')
plt.ylabel('silhouette')

#%%
'''
3 17651.8931344 0.517927253705
4 12383.2398798 0.498514988967
5 9338.19064691 0.454534972902
6 6726.66603654 0.467343481454
7 5771.98116461 0.44443887556
8 4842.73828066 0.427560620842
9 4271.28793361 0.426397579452

[result]
there is not a significant elbow in the inertia plot, but silhouette plot research a stable value when k is 10.
Hence, 10 will be used in kmeans for further analysis. 
'''
#%%
import pandas as pd
k=3
kmeans = KMeans(n_clusters=k, random_state=0).fit(pc_2d)
labels = kmeans.labels_
inertia = kmeans.inertia_
sil = silhouette_score(pc_2d, labels, metric='euclidean')

print(k,inertia, sil)

#%
pc2_label_array = {'pc1':pc_2d[:,0],'pc2':pc_2d[:,1],'label':labels}
pc2_label = pd.DataFrame(data=pc2_label_array )
del pc2_label_array

#%%
import numpy as np

uniq_labels = np.unique(labels)

cmap = plt.cm.get_cmap("hsv", len(uniq_labels)+1)
fig = plt.figure(10)

ax = fig.add_subplot(1,1,1)
ax.set_title('KMeans(k=3) on PC1 and PC2')
for label_i, label in enumerate(uniq_labels):
    group = pc2_label.ix[pc2_label.label == label]
    ax.scatter(group.pc1,group.pc2, c=cmap(label_i))
ax.set_xlabel('1st PC')
ax.set_ylabel('2nd PC')
x_scale = ax.get_xlim()
y_scale = ax.get_ylim()


#%%
for label_i, label in enumerate(uniq_labels):
    fig = plt.figure(label_i)
    ax = fig.add_subplot(1,1,1)
    group = pc2_label.ix[pc2_label.label == label]
    ax.scatter(group.pc1,group.pc2, c=cmap(label_i))
    ax.set_xlabel('1st PC')
    ax.set_ylabel('2nd PC')
    ax.set_title('Cluster with label '+str(label_i))
    
    ax.set_xlim(x_scale)
    ax.set_ylim(y_scale)
    plt.show()



#%%
#%%
import pandas as pd
import numpy as np
for k in range(2,4):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pc_2d)
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    sil = silhouette_score(pc_2d, labels, metric='euclidean')
    
    print(k,inertia, sil)
    
    #%
    pc2_label_array = {'pc1':pc_2d[:,0],'pc2':pc_2d[:,1],'label':labels}
    pc2_label = pd.DataFrame(data=pc2_label_array )
    del pc2_label_array
    
    #%
    plt.title('KMeans(k='+str(k)+') on PC1 and PC2')

    uniq_labels = np.unique(labels)
    
    cmap = plt.cm.get_cmap("hsv", len(uniq_labels)+1)
    for label_i, label in enumerate(uniq_labels):
        group = pc2_label.ix[pc2_label.label == label]
        plt.scatter(group.pc1,group.pc2, c=cmap(label_i))
    
    
    plt.xlabel('1st PC')
    plt.ylabel('2nd PC')
    plt.show()

    
#%%
import pandas as pd
k=3
kmeans = KMeans(n_clusters=k, random_state=0).fit(pc_2d)
labels = kmeans.labels_
inertia = kmeans.inertia_
sil = silhouette_score(pc_2d, labels, metric='euclidean')

print(k,inertia, sil)

#%%
raw_data_label = pd.read_csv('data/HR_comma_sep.csv')
raw_data_label['label'] = labels

#%%
'''
interpret PC1 and PC2
'''


#%%
'''
find which cluster has the most left=1
'''

for label in uniq_labels:
    group = raw_data_label[raw_data_label.label == label]
    group_1 = group[group.left==1]
    print(label,',1s:',str(group_1.shape[0])+'/'+str(group.shape[0]),',',group_1.shape[0]/group.shape[0])
    
#%%
'''
0 ,1s: 1096/5622 , 0.19494841693347564
1 ,1s: 663/6786 , 0.09770114942528736
2 ,1s: 1812/2591 , 0.6993438826707835

[Result]
cluster 2 has the largest percentage of employees left the company.
'''