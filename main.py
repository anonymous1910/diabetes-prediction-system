# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
p=int(input("enter the pregnancies"))
g=int(input("enter the glucose"))
bp=int(input("enter the blood pressure"))
s=int(input("enter the skin thickness"))
i=int(input("enter the insulin"))
bm=int(input("enter the BMI"))
d=int(input("enter the diabetes pedigree function"))
a=int(input("enter the age"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes = pd.read_csv('diabetes.csv')
print(diabetes.columns)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
t=tree.fit(X_train, y_train)
#v=s.predict([(2,34,45,56,72,3,1,3)])#
v=t.predict([(p,g,bp,s,i,bm,d,a)])
print(v)
