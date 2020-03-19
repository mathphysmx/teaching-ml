import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# n = n
n = 100
X = 2 * np.random.rand(n, 1)
y = -3 * X + np.random.rand(n, 1)
plt.plot(X, y, '.')
plt.show()



plt.plot(X,y, "b.")
m = sum(X * y) / sum(X**2); print(m)
y_predict = m * X
plt.plot(X,y_predict, "r-")
plt.show()


from sklearn.linear_model import LinearRegression
line = LinearRegression()
line.fit(X,y)
line.coef_
line.intercept_

plt.plot(X,y, 'b.')
plt.plot(X, line.predict(X), 'g-')
# plt.plot(X, line.coef_ * X + line.intercept_)
plt.show()

from sklearn.metrics import mean_squared_error
mean_squared_error(y, y_pred = line.predict(X))

mean_squared_error(y, y_pred = y_predict)


from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
mltree = DecisionTreeRegressor()
mltree.fit(X,y)
plt.plot(X, mltree.predict(X), 'g-')
plt.plot(X,y, 'b.')
plt.show()

mean_squared_error(y, y_pred = mltree.predict(X))
