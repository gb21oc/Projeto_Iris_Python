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