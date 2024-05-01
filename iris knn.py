import numpy as np 
import pandas as pd 
from  sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('iris.csv')
print(df.head())
print(df['species'].value_counts())
x=df.drop('species',axis=1).values
print(x)
y=df['species'].values
x=StandardScaler().fit_transform(x)
print(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
k = 3  # choose a value for k
neigh = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
y_pred = neigh.predict(x_test)
print("Accuracy of KNN with k = 3 is: ", metrics.accuracy_score(y_test, y_pred))
# List to hold the accuracy for each value of k
accuracy_list = []

# Range of k values to try
k_range = range(1, 50)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)

# Find the value of k that has the highest accuracy
optimal_k = k_range[accuracy_list.index(max(accuracy_list))]

print("Optimal k is:", optimal_k)
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])

# Standardize the new data point
new_data = StandardScaler().fit_transform(new_data)

# Use the trained model to make a prediction
new_pred = neigh.predict(new_data)

print("Predicted species for the new data point:", new_pred)
