import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('trainingset.csv')


for col in df.columns:
  if df[col].unique().size < 30:
    print(f'{col}: {df[col].unique().size}, range: {df[col].max()} { df[col].min()}')
    freq = df[col].value_counts()
    plt.bar(freq.index, freq.values)
    plt.xticks(freq.index)
    plt.title(col)
    plt.show()