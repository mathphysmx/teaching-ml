# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:52:13 2020

@author: Karla
"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_score, recall_score, f1_score

A=1
B=2
TrueLabel=[A, A, B, A, B, A, A, B, B, A]
Predicted = [A, B, B, B, B, B, A, A, B, A]
data = {'TrueLabel': [A, A, B, A, B, A, A, B, B, A],'Predicted': [A, B, B, B, B, B, A, A, B, A]}


df = pd.DataFrame (data, columns = ['TrueLabel', 'Predicted'])
confusion_matrix = pd.crosstab (df ['TrueLabel'], df ['Predicted'], rownames = ['yReal'], colnames = ['yPredicted'])
print(df)
sn.heatmap (confusion_matrix, annot = True)
plt.show ()


precision_score=precision_score(TrueLabel, Predicted)
print(precision_score)
recall_score=recall_score(TrueLabel, Predicted)
print(recall_score)
f1_score=f1_score(TrueLabel, Predicted)
print(f1_score)
