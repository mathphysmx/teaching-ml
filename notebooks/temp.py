import pandas as pd

x = pd.read_csv("datasets/Guangren_table34.txt")
x.info()

# Linear regression (finding best fit model)
b = 2
x = pd.Series([0, 1, 2])
y = pd.Series([0.1, 0.9, 4.1]) + b
m = -b * (x.sum() / (y**2).sum())

import matplotlib.pyplot as plt
import seaborn as sns
iris = sns.load_dataset('iris')
iris.plot(kind = 'scatter', x = 'petal_length', y = 'petal_width')
# sns.pairplot(iris)
plt.show()

import numpy as np
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(np.c_[iris.petal_length], iris.petal_width.values)
t0, t1 = model.intercept_, model.coef_[0]
t0, t1
iris.plot(kind='scatter', x = 'petal_length', y = 'petal_width')
X=np.linspace(1, 7, num=7)
plt.plot(X, t0 + t1*X, "b")
plt.show()

#############3 linear regression

import numpy as np
import matplotlib.pyplot as plt

# n = 1
# Instances (Sample points)
xi = 12 # data points o o o o o
yi = 4
theta = yi/xi
plt.scatter(xi,yi)
x = np.linspace(10, 14, num = 5); x # model Line _________
y = theta * x
plt.plot(x, y)
plt.axis('equal')
plt.show()

# n = 2
s = 1
# ax = 6; ay = 2 # perpendicular vectors (data points)
# xi = np.array([ax, -ay])
# yi = np.array([ay, ax])
xi = np.array([3, 12]) # data points o o o o o
yi = np.array([9, 4])
theta = sum(xi * yi) / sum(xi**2)
plt.scatter(xi,yi)
# Model
x = np.linspace(min(xi), max(xi), num = 5); x
y = theta * x
plt.plot(x, y)
plt.axis('equal')
plt.show()

# n = n
X = 2 * np.random.rand(100, 1)
y = -3 * X + np.random.rand(100, 1)
plt.plot(X,y, "b.")
m = sum(X * y) / sum(X**2); print(m)
y_predict = m * X
plt.plot(X,y_predict, "r-")
plt.show()