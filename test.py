import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from preprocessing_data import get_data
from main import cv_two_stage

df = pd.read_csv('trainingset.csv')

def test_single_feature():
  X, y = get_data()
  features = []
  for col in X.columns:
    f1, _, mae = cv_two_stage(pd.DataFrame(X, columns=[col]), y, 5)
    features.append((col, f1, mae))
    
  features.sort(key=lambda x: (-x[1], x[2]))
  return features

def svr_grid_search():
  svr_params = {
    'C': [100, 500, 1000, 10000],
    'epsilon': [1, 5, 10, 100, 1000],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'] 
  }
  # grid search for svr
  grid_search = GridSearchCV(SVR(), svr_params, cv=5, scoring='neg_root_mean_squared_error')
  
  X, y = get_data('trainingset.csv', normalize=True)
  X, y = X[y > 0], y[y > 0]
  
  grid_search.fit(X, y)
  print(grid_search.best_params_)
  # {'C': 100, 'epsilon': 1, 'gamma': 'auto', 'kernel': 'poly'}
  # {'C': 1000, 'epsilon': 10, 'gamma': 'auto', 'kernel': 'poly'}
  # {'C': 10000, 'epsilon': 1000, 'gamma': 'auto', 'kernel': 'rbf'}
  
  

svr_grid_search()

# for col in df.columns:
#   if df[col].unique().size < 30:
#     print(f'{col}: {df[col].unique().size}, range: {df[col].max()} { df[col].min()}')
#     freq = df[col].value_counts()
    # plt.bar(freq.index, freq.values)
    # plt.xticks(freq.index)
    # plt.title(col)
    # plt.show()
    
    