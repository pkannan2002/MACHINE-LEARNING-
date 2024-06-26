import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import seaborn as sn 

df =pd.read_csv('titanic.csv')
# print(df.head())
# print(df.isnull().sum())
df['Age'].fillna(df['Age'].mean(),inplace=True)
# print(df.isnull().sum())
df=pd.get_dummies(df,columns=['Sex'])
# print(df.head())

x=df[['Age', 'Sex_female','Sex_male','Pclass']]
y=df['Survived']
x=preprocessing.StandardScaler().fit(x).transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
logreg=LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)
y_pred=logreg.predict(x_test)
cnf=metrics.confusion_matrix(y_test,y_pred)
print(cnf)
accuracy=metrics.accuracy_score(y_test,y_pred)

print(accuracy)


