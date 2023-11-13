import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from preprocessing_data import get_data
from model import get_nn_model


def split_data(data):
    X = data.drop(columns=['ClaimAmount'])
    Y = data['ClaimAmount']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


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
  
	if model_type == 'nn':
		X, y = get_data(normalize=True)
		model = get_nn_model(X.shape[1])
		err = cv(model, X, y, K)
		print(f'CV error (NN): {err}')
	elif model_type == 'rf':
		X, y = get_data(normalize=False)
		model = RandomForestRegressor(n_estimators=100, random_state=42)
		result = cross_validate(model, X, y, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV error (RF): {-np.mean(result["test_score"])}')
	else:
		X, y = get_data(normalize=True)
		model = LinearRegression()
		result = cross_validate(model, X, y, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV error (LR): {-np.mean(result["test_score"])}')
   

def main():
	# evaluate()
	# evaluate('nn')
	evaluate('rf')


if __name__ == "__main__":
    main()