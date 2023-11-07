import pandas as pd

import preprocessing_data as processor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# feature 1, 2, 6, 8, 10, 12, 17 for scatter

COL_NUM = 3


def split_data(data):
    X = data.drop(columns=['ClaimAmount'])
    Y = data['ClaimAmount']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


def split_in_out(data):
    X = data.drop(columns=['ClaimAmount'])
    Y = pd.DataFrame(data['ClaimAmount'])
    return X, Y


def scatter(X, Y):
    num_features = X.columns.size
    fig, axs = plt.subplots(int(np.ceil(num_features / COL_NUM)), COL_NUM, figsize=(10, 10))
    for i in range(num_features):
        feature_name = X.columns[i]
        feature = X[feature_name].values

        scatter_plot = axs[int(np.floor(i / COL_NUM)), i % COL_NUM]
        scatter_plot.scatter(feature, Y.values)
        scatter_plot.set_xlabel(feature_name, size=10)
        scatter_plot.set_ylabel(Y.columns[0], size=10)

    fig.tight_layout(pad=2)
    fig.show()


# def hist(X, Y):
#     num_features = X.columns.size
#     fig, axs = plt.subplots(int(np.ceil(num_features / COL_NUM)), COL_NUM, figsize=(10, 10))
#     for i in range(num_features):
#         feature_name = X.columns[i]
#         feature = X[feature_name].vaues
#
#         hist_axs = axs[int(np.floor(i / COL_NUM)), i % COL_NUM]
#         hist_axs.hist(feature, bins=5)
#         hist_axs.set_xlabel(feature_name, size=10)
#
#     fig.tight_layout(pad=2)
#     fig.show()


def main():
    data = processor.process_data('trainingset.csv')
    X, Y = split_in_out(data)
    scatter(X, Y)
    # hist(X, Y)


if __name__ == "__main__":
    main()