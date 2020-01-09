import pandas as pd

x = pd.read_csv("datasets/Guangren_table34.txt")
x.info()

# Linear regression (finding best fit model)

b = 2
x = pd.Series([0, 1, 2])
y = pd.Series([0.1, 0.9, 4.1]) + b

m = -b * (x.sum() / (y**2).sum())