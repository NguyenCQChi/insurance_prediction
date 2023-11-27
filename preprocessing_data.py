import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

categorical_fs = ['feature3', 'feature4', 'feature5', 'feature7', 'feature9', 'feature11', 'feature13', 'feature14', 'feature15', 'feature16', 'feature18']

# load and process data
# Inputs:
#  filepath: path to the csv file
# Outputs:
#  data: processed data
def process_data(filepath):
    data_read = pd.read_csv(filepath)
    data_read = data_read.drop(columns=['rowIndex'])
    data_read.dropna(inplace=True)
    return data_read


def get_data(filepath, test=False, normalize=False, one_hot=False, transform=False, pca=False):
    data = process_data(filepath)
    if test:
        X = data
        y = None
    else:
        X = data.drop(columns=['ClaimAmount'])
        y = data['ClaimAmount']
    
    if transform:
        X['m_feature8_feature2'] = (X['feature8'] + X['feature2']) / 2
        # X['p_feature1_2_10'] = X['feature1'] ** 2 + X['feature2'] + X['feature10']
        X.drop(columns=['feature8', 'feature2'], inplace=True)

        
    if one_hot:
        X = get_one_hot_data(X, normalize)
    elif normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
    if pca:
        pca = PCA(n_components=X.shape[1] - 2)
        X = pca.fit_transform(X)
        X = pd.DataFrame(X)
        
    return X, y


def get_one_hot_data(df, normalize=False):
    scaler = StandardScaler()
    for col in df.columns:
        if df[col].unique().size < 20:
            dummies = pd.get_dummies(df[col], prefix=col)
            loc = df.columns.get_loc(col)
            df.drop(columns=[col], inplace=True)
            left = df.iloc[:, :loc]
            right = df.iloc[:, loc:]
            df = pd.concat([left, dummies, right], axis=1)
        elif normalize:
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    return df


def get_test_data(normalize=False):
    X = process_data('testset.csv')
    
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    return X

        
def main():
    # print(process_data('trainingset.csv'))
    get_data(normalize=True, one_hot=True)


if __name__ == '__main__':
    main()
