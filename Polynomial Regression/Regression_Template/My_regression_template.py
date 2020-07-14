# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:57:35 2019

@author: Ayush
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
sc_Y=StandardScaler()
Y_train=sc_Y.fit_transform(Y_train)
Y_test=sc_Y.fit_transform(Y_test)

#Regressor

#Prediction
y_pred=regressor.predict(X)

#Visulaisation
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X), color='blue')
plt.show()

X_grid=np.arange(min(X), max(X), 0.1)
X_grid=X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.show()