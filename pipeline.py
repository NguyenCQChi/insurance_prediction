import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import HuberRegressor
from xgboost import XGBRFRegressor
import joblib
import sys


def main():
    # Check if the number of arguments is correct and set filepath
    if len(sys.argv) != 2:
        print("Usage: python pipeline.py <filepath>")
        sys.exit(1)
    filepath = sys.argv[1]

    # Load and process data
    scaler = StandardScaler()
    df = pd.read_csv(filepath)
    df = df.drop(columns=['rowIndex'])
    df.dropna(inplace=True)
    data = scaler.fit_transform(df)
    
    # Load models and predict
    rf_model = joblib.load('./rf_model.sav')
    y_pred = (rf_model.predict_proba(data)[:,1] > 0.25).astype(int)
    zero_indices = np.where(y_pred == 0)[0]
    hr_model = joblib.load('./hr_model.sav')
    y_pred = hr_model.predict(data)

    # Set negative predictions to 0
    y_pred[zero_indices] = 0
    
    # Save predictions to csv
    df = pd.DataFrame({'ClaimAmount': y_pred})
    df.to_csv('predictedclaimamount.csv', index_label='rowIndex')

if __name__ == "__main__":
    main()