import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


for col in df.columns:
  if df[col].unique().size < 30:
    print(f'{col}: {df[col].unique().size}, range: {df[col].max()} { df[col].min()}')
    freq = df[col].value_counts()
    # plt.bar(freq.index, freq.values)
    # plt.xticks(freq.index)
    # plt.title(col)
    # plt.show()