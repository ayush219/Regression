# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 02:28:55 2019

@author: Ayush
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
Lin_reg=LinearRegression()
Lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
Lin_reg2=LinearRegression()
Lin_reg2.fit(X_poly,y)


y_pred=Lin_reg.predict(6.5)
y_pred2=Lin_reg2.predict(poly_reg.fit_transform(6.5))

plt.scatter(X,y,color='red')
plt.plot(X,Lin_reg.predict(X), color='blue')
plt.show()
plt.scatter(X,y,color='red')
plt.plot(X,Lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

X_grid=np.arange(min(X), max(X), 0.1)
X_grid=X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,Lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()