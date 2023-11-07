import pandas as pd
import numpy as np


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


def main():
    print(process_data('trainingset.csv'))


if __name__ == '__main__':
    main()
