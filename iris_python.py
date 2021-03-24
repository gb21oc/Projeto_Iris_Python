# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:53:09 2021

@author: gabriel.arphoenix
"""

import pandas as pd

df = pd.read_csv("iris.data") # Importando o nosso dataser
df.rename(
    columns={
        '5.1': 'sepal length (cm)', 
        '3.5': 'sepal width (cm)', 
        '1.4': 'petal length (cm)', 
        '0.2': 'petal width (cm)' 
        }, inplace = True)  # Renomeando as colunas
df.info()  # Verificando as informaçoes
df.describe(include="all")  # Verificando a descrição completa do dataset
x = df.drop("Iris-setosa", axis=1) # Separando a variavel preditora (x = df.iloc[:, :-1].values)
y = df["Iris-setosa"]  # Separando a variavel targe (y = df.iloc[:, 4].values)

from sklearn.preprocessing import StandardScaler # Importando o modulo para fazer a normalização dos dados
scaler = StandardScaler()
scaler = scaler.fit(x)
x = scaler.transform(x)  # Transformando os dados

from sklearn.model_selection import train_test_split  # Importando o modulo de treino
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)  # Treino do modelo


# Esse modelo foi escolhido pq o dataset é de classificação e na minha opiniao ele é o que se adapta mais
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)  # O 5 é o padrao
classifier.fit(x_train, y_train) 
y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix # Avaliando o Algoritmo 
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) 

