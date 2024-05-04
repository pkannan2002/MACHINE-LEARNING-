# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('diabetes.csv')
# print(df.head())
# print(df.describe)
scaler = StandardScaler()

feature_names = df.drop('Outcome', axis=1).columns
x=df.drop('Outcome',axis=1)
x = scaler.fit_transform(x)
y=df['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
# Train Decision Tree Classifier
drugTree = drugTree.fit(x_train, y_train)

# Predict the response for test dataset
y_pred = drugTree.predict(x_test)

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Import necessary libraries
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the decision tree
plt.figure(figsize=(15,10))
plot_tree(drugTree, filled=True, rounded=True, feature_names=feature_names)
plt.show()

