import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB

from preprocessing_data import get_data
from model import get_nn_model


def split_data(data):
    X = data.drop(columns=['ClaimAmount'])
    Y = data['ClaimAmount']

    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    # Feature Scaling (Standardization)
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    return X, Y

def undersample(filepath, undersample_rate):
	df = pd.read_csv(filepath)
	data = df.drop(columns=['rowIndex'])
	data.dropna(inplace=True)
	majority = data[data['ClaimAmount'] == 0]
	minority = data[data['ClaimAmount'] != 0]
	print(majority['ClaimAmount'].value_counts())
	print(minority['ClaimAmount'].value_counts())

	majority_count = int((len(minority)/undersample_rate)-len(minority))
	majority = majority.sample(majority_count)
	undersampled_data = pd.concat([majority, minority])
	return undersampled_data

def cv(model, X, y, K):
	kf = KFold(K, shuffle=True, random_state=42)
	mae = []
	
	for train_idx, test_idx in kf.split(X):
		model.compile(optimizer='adam', loss='mse', metrics=['mae'])
		model.fit(X[train_idx], y[train_idx], epochs=10, batch_size=32, verbose=1)
		loss, metrics = model.evaluate(X[test_idx], y[test_idx], verbose=1)
		mae.append(metrics)
		
	return np.mean(mae) 

def evaluate(model_type='lr'):
	K = 5
  
	
	if model_type == 'ridge':
		X, y = get_data(normalize=True)
		X_test = get_data(train=False)
		model = Ridge(alpha=1.0)
		model.fit(X, y)
		y_pred = model.predict(X_test)
		df= pd.DataFrame({'ClaimAmounts': y_pred})
		df.to_csv('ridge.csv', index_label='rowIndex')
		result = cross_validate(model, X, y, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV error (Ridge): {-np.mean(result["test_score"])}')
	elif model_type == 'ridge_undersampled':
		data = undersample("trainingset.csv", 0.2)
		X, y = split_data(data)
		model = Ridge(alpha=1.0)
		model.fit(X, y)
		X_test = get_data(train=False)
		y_pred = model.predict(X_test)
		df= pd.DataFrame({'ClaimAmounts': y_pred})
		df.to_csv('ridge_undersampled.csv', index_label='rowIndex')
		result = cross_validate(model, X, y, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV error (Ridge_Undersampled (0.2)): {-np.mean(result["test_score"])}')
	elif model_type == 'lasso':
		X, y = get_data(normalize=True)
		X_test = get_data(train=False)
		model = Lasso(alpha=1.0)
		model.fit(X, y)
		y_pred = model.predict(X_test)
		df= pd.DataFrame({'ClaimAmounts': y_pred})
		df.to_csv('lasso.csv', index_label='rowIndex')
		result = cross_validate(model, X, y, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV error (Lasso): {-np.mean(result["test_score"])}')
	elif model_type == 'lasso_undersampled':
		data = undersample("trainingset.csv", 0.2)
		X, y = split_data(data)
		model = Lasso(alpha=1.0)
		model.fit(X, y)
		X_test = get_data(train=False)
		y_pred = model.predict(X_test)
		df= pd.DataFrame({'ClaimAmounts': y_pred})
		df.to_csv('lasso_undersampled.csv', index_label='rowIndex')
		result = cross_validate(model, X, y, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV error (Lasso_Undersampled (0.2)): {-np.mean(result["test_score"])}')
	else:
		X, y = get_data(normalize=True)
		model = LinearRegression()
		result = cross_validate(model, X, y, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV error (LR): {-np.mean(result["test_score"])}')
   

def main():
	# evaluate()
	# evaluate('nn')
	# evaluate('rf')
	rates = 0.2
	undersample("trainingset.csv", rates)
	evaluate('ridge')
	evaluate('ridge_undersampled')
	evaluate('lasso')
	evaluate('lasso_undersampled')
	evaluate('nb')


if __name__ == "__main__":
    main()