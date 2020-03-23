# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:51:53 2020

@author: Karla
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import webbrowser
import urllib.request, urllib.error, urllib.parse
import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import seaborn as sns

from itertools import if
[i for i in filter(lambda x:x%5, islice(count(5), 10))]


#Pregunta 4
cadena = "examen Karla"
print (cadena [3]) # aqui puedo ver la letra que corresponde a la posicion 3, recordando que en python 100pre empieza a contar en el '0'
print (cadena)

#pregunta 5
definir = pd. DataFrame ({ "Ml_Classification" : [ " Unsupervised Learning", "Supervised Learning"], "Application percentage in earth science" : ["40","60"]})
print (definir)

#pregunta 6
definir = pd.DataFrame({"x" : [2 ,3, 4, 5, 6, 7],"y" : [1, 7, 5, 12, 11, 4 ],})
definir.plot.scatter(x='x',y='y')
print(definir)
definir.head()

#pregunta 7
url = 'https://raw.githubusercontent.com/GeostatsGuy/GeoDataSets/master/1WellPorPerm.csv'
df= pd.read_csv(url)
df.head()
df.tail()
titulo = ["Depth (m)", "Por (%)", "Perm (mD)"]
df.columns = titulo
df.head()
ruta = "E:/maestria Unam/asignaturas 2do semestre/examen Karla/1WellPorPerm.csv "
df.to_csv(ruta)
df.head()
print(df)
#a)
df = pd.read_csv("E:/maestria Unam/asignaturas 2do semestre/examen Karla/1WellPorPerm.csv ")

plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(df['Por (%)'],kde=False)
plt.xlabel('Por (%)')
plt.axis([1,30,0,30])

#b

plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(df['Perm (mD)'],kde=False)
plt.xlabel('Perm (mD)')
plt.axis([3,600,0,30])
#C
df = pd.read_csv("E:/maestria Unam/asignaturas 2do semestre/examen Karla/1WellPorPerm.csv ")
from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2)
#d

x =df[['Por (%)']] #varieble predictora
y =df[['Perm (mD)']]  #variable predicha
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
model.coef_
model.intercept_
plt.plot(x,model.predict(x),'g-',x,y,"b.")
plt.show()
plt.plot(x,y,"b.")
plt.show()
from sklearn.metrics import mean_squared_error
mean_squared_error(y, y_pred = model.predict(x))

#
from sklearn.ensemble import RandomForestRegressor
modelo = RandomForestRegressor()
modelo.fit(x,y)
plt.plot(x,y,"b.")
plt.plot(x,modelo.predict(x),"r.",x,y,"b.")
plt.show()
mean_squared_error(y, y_pred = modelo.predict(x))
#
from sklearn.tree import DecisionTreeRegressor
mltree = DecisionTreeRegressor()
mltree.fit(x,y)
plt.plot(x,y,"b.")
plt.plot(x,mltree.predict(x),"r.",x,y,"b.")
plt.show()
mean_squared_error(y, y_pred = mltree.predict(x))

# el mejor ajuste fue el del arbol de deciciones ya que el error fue '0'
x =df[['Por (%)']] #varieble predictora
y =df[['Perm (mD)']]  #variable predicha
from sklearn.model_selection import train_test_split
a,b = train_test_split(x,test_size=0.2)
len(a)
len(b)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
ya=model.predict(a)
yb=model.predict(b)
mean_squared_error(ya, y_pred = model.predict(a))
mean_squared_error(yb, y_pred = model.predict(b))

#
from sklearn.ensemble import RandomForestRegressor
modelo = RandomForestRegressor()
modelo.fit(x,y)
x =df[['Por (%)']] #varieble predictora
y =df[['Perm (mD)']]  #variable predicha
from sklearn.model_selection import train_test_split
a,b = train_test_split(x,test_size=0.2)
len(a)
len(b)
ya=model.predict(a)
yb=model.predict(b)
mean_squared_error(ya, y_pred = model.predict(a))
mean_squared_error(yb, y_pred = model.predict(b))