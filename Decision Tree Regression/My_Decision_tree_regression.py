# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:58:01 2019

@author: Ayush
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:,1:2].values
Y= dataset.iloc[:,2:].values

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

Y_pred=regressor.predict(6.5)

plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X), color='blue')
plt.show()

X_grid=np.arange(min(X), max(X), 0.1)
X_grid=X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.show()