import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd 
cus_df=pd.read_csv('Cust_Segmentation.csv')
# print(cus_df.head())
cus=cus_df.drop('Address',axis=1)
# print(cus.head())
from sklearn.preprocessing import StandardScaler
X = cus.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
# print(Clus_dataSet)
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
# print(labels)
cus["Clus_km"] = labels
# cus.head(5)
cus.groupby('Clus_km').mean()
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float64), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()