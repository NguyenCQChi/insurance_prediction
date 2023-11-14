import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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


def get_data(normalize=False):
    data = process_data('trainingset.csv')
    X = data.drop(columns=['ClaimAmount'])
    y = data['ClaimAmount']
    
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    return X, y


def get_test_data(normalize=False):
    X = process_data('testset.csv')
    
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    return X

        
def main():
    print(process_data('trainingset.csv'))


if __name__ == '__main__':
    main()
