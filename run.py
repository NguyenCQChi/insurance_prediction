import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import sys


def main():
    # Check if the number of arguments is correct and set filepath
    if len(sys.argv) != 2:
        print("Usage: python run.py <filepath>")
        sys.exit(1)
    filepath = sys.argv[1]

    # Load and process data
    scaler = StandardScaler()
    X = pd.read_csv(filepath)
    X = X.drop(columns=['rowIndex'])
    X.dropna(inplace=True)
    
    # Load models and predict
    clf_model = joblib.load('./clf.sav')
    reg_model = joblib.load('./reg.sav')
    
    y_pred = np.zeros(X.shape[0])

    # Predict with classifier
    y1_pred = clf_model.predict(X)
    
    # Predict with regressor
    X2 = X[y1_pred > 0]
    X2 = scaler.fit_transform(X2)
    y2_pred = reg_model.predict(X2)
    
    y_pred[y1_pred > 0] = y2_pred
    
    # Save predictions to csv
    df = pd.DataFrame({'ClaimAmount': y_pred})
    df.to_csv('predictedclaimamount.csv', index_label='rowIndex')

if __name__ == "__main__":
    main()