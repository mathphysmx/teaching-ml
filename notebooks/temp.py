import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


co=pd.DataFrame({"x":[-1,0,3],"y":[-7,0,2]})
centers = pd.DataFrame({'x':[1,5], 'y':[2, 6]})
centers.iloc[0, 0]

from sklearn.cluster import KMeans





#################

x = pd.read_csv("datasets/Guangren_table34.csv")
x.info()

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
import pandas as pd


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
n = 100
X = 2 * np.random.rand(n, 1)
y = -3 * X + np.random.rand(n, 1)
plt.plot(X,y, "b.")
m = sum(X * y) / sum(X**2); print(m)
y_predict = m * X
plt.plot(X,y_predict, "r-")
plt.show()

df = pd.DataFrame(np.hstack((X, y)), columns = ['x', 'y'])
df.plot('x', 'y', kind='scatter')
plt.show()

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LinearRegression
mlmodel_reg=LinearRegression()
mlmodel_reg.fit(train.x.T.values.reshape(-1,1),train.y.T.values)
mlmodel_reg.intercept_
mlmodel_reg.coef_

y_predict_sklearn = mlmodel_reg.intercept_ + mlmodel_reg.coef_ * X
plt.plot(X,y, "b.")
plt.plot(X,y_predict_sklearn, "r-")
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df.y, y_predict);mse

train_predictions=mlmodel_reg.predict(X)
mse = mean_squared_error(df.y, train_predictions); mse

