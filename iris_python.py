# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:53:09 2021

@author: gabriel.arphoenix
"""

import pandas as pd

df = pd.read_csv("iris.data")
df.rename(
    columns={
        '5.1': 'sepal length (cm)', 
        '3.5': 'sepal width (cm)', 
        '1.4': 'petal length (cm)', 
        '0.2': 'petal width (cm)' 
        }, inplace = True)
df.info()
df.describe(include="all")
x = df.drop("Iris-setosa", axis=1)
y = df["Iris-setosa"]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(x)
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)  #O 5 é o padrao
classifier.fit(x_train, y_train) 
y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) 

