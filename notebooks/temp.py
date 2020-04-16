import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans


############# KMEANS color segmentation
import pandas as pd
import numpy as np
import os
from matplotlib.image import imread # you could also use `imageio.imread()`
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image = imread(os.path.join("datasets","ladybug.png"))
image.shape

    np.array([(1.5,2,3), (4,5,6)])
c = np.array([[(1.5,2,3), (4,5,6)], [(3,2,1), (4,5,6)]], dtype = float)
c.shape
c.reshape(-1, 3)

X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))

plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.subplot(231)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title("{} colors".format(n_clusters))
    plt.axis('off')

plt.show()

########## KMEAS WElL LOGS

import lasio
import welly

las = lasio.read("datasets/47-019-00241-00-00.txt")
las.keys()
df = las.df().iloc[1:,:]
gr = pd.to_numeric(df['GR']).replace(-999.25, np.nan)
gr.describe()

import numpy as np
import matplotlib.pyplot as plt
import welly
welly.__version__
from welly import Well
w = Well.from_las("datasets/P-129_out.LAS")
w.data['GR']
w.plot()
plt.show()
gr = w.data['GR']
gr[1000:1010]
gr.plot(); plt.show()
gr.plot(lw=0.3); plt.show()
gr.plot_2d(cmap='viridis', curve=True, lw=0.3, edgecolor='k')
plt.show()


############
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


###############
co=pd.DataFrame({"x":[-1,0,3],"y":[-7,0,2]})
centers = pd.DataFrame({'x':[1,5], 'y':[2, 6]})
centers.iloc[0, 0]

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

