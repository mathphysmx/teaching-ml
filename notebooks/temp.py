























##################################
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(train_images, train_labels,epochs=5)

a= model.evaluate(test_images, test_labels)

prob = keras.Sequential([model, keras.layers.Softmax()])
predictions = prob.predict(test_images)

predictions[0].max()
predictions[0]
class_names[np.argmax(predictions[0])]
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

prob.save('asdflkaj.h5')

prob = tf.keras.models.load_model('asdflkaj.h5')
tf.keras.models.load_model


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

##############33

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances

#### CLUSTERING
n = 100
np.random.seed(844)
clust1 = np.random.normal(5, 2, (n, 2))
clust2 = np.random.normal(15, 3, (n,2))
dataset = np.concatenate((clust1, clust2))

plt.scatter(x=dataset[:,0], y=dataset[:,1], c=[1]*n + [2]*n)
plt.show()

# KMEANS
kmeans = KMeans(n_clusters=2)
kmeans.fit(dataset)
plt.scatter(x=dataset[:,0], y=dataset[:,1], c=kmeans.labels_)
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=0.01, min_samples=4)
dbscan.fit(dataset)
plt.scatter(x=dataset[:,0], y=dataset[:,1], c=dbscan.labels_)
plt.show()

# Explore distances
datadist=pd.DataFrame(pairwise_distances(dataset,dataset)).apply(distmin)
datadist.describe()
plt.scatter(x = datadist.values, y=range(n))
plt.show()


# creacion de funcion de distancias
x = np.array([[2, 3], [0,0], [10, 15]]).reshape(-1,2)
df = pd.DataFrame(pairwise_distances(x,x))
def distmin(y):
    return((y[y>0]).min())
distmin(y = df.iloc[2])


########## KMEANS WElL LOGS
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

