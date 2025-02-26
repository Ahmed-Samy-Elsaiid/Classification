# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading Dataset
data=pd.read_csv(r'titanic_train.csv')
data
data.info()

# Data cleaning
# Checking for null values

data.isnull().sum()
# Handling null values

data.drop(data[data["Embarked"].isnull()].index,axis=0,inplace=True)
data['Age']=data['Age'].fillna(data['Age'].median())
data.drop("Cabin",axis=1,inplace=True)
data.isnull().sum()
# Checking for duplicated values

data.duplicated().sum()
data.head()
data.describe()
# **Classification**

# Feature Engineering
# Ckecking if the Passenger is alone or not

data["IsAlone"] = (data["SibSp"]==0)&(data["Parch"]==0)
# Detecting the title of each Passenger

data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')
# Drop unnecessary columns

data.drop(["PassengerId",'Name',"Ticket","Parch","SibSp","Fare"],axis=1,inplace=True)
data.head()

# Label Encoding
data["Title"].replace({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5},inplace=True)
# Use LabelEncoder to encode categorical values into numbers

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
# Use LabelEncoder to encode the rest of categorical values into numbers

data['Sex']=le.fit_transform(data['Sex'])
data['Embarked']=le.fit_transform(data['Embarked'])
data['IsAlone']=le.fit_transform(data['IsAlone'])

# Feature Selection
data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(),annot=True)
plt.show()
cor=data.corr()['Survived']
cols_to_drop=cor[(cor>-0.15)&(cor<0.15)].index
data=data.drop(cols_to_drop,axis=1)
data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(),annot=True)
plt.show()

# Train - test   split
from sklearn.model_selection import train_test_split

X=data.drop(['Survived'],axis=1)

Y=data['Survived']

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
# Scale all features using StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Decision Tree
# Use DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,Y_train)
y_test_pred_tree = dtree.predict(X_test)
from sklearn.tree import plot_tree
plt.figure(figsize=(100,200))
plot_tree(dtree)
plt.show()
from sklearn.metrics import accuracy_score, precision_score

accuracy = accuracy_score(Y_test, y_test_pred_tree)
print("Accuracy:", accuracy)

precision = precision_score(Y_test, y_test_pred_tree)
print("Precision:", precision)