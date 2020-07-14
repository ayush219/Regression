# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:30:30 2019

@author: Ayush
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:,1:2].values
Y= dataset.iloc[:,2:].values

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)
sc_Y=StandardScaler()
Y=sc_Y.fit_transform(Y)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)

Y_pred=regressor.predict(sc_X.transform(np.array([[6.5]])))
Y_pred=sc_Y.inverse_transform(Y_pred)

plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X), color='blue')
plt.show()

X_grid=np.arange(min(X), max(X), 0.1)
X_grid=X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.show()