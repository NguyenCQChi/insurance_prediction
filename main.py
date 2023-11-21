import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from preprocessing_data import get_data


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

def evaluate(model_type, X_train, y_train, X_test, y_train_binary):
	K = 5
	
	if model_type == 'rf':
		rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
		rf_classifier.fit(X_train, y_train_binary)

		binary_predictions = rf_classifier.predict(X_test)

		no_claim_indices = np.where(binary_predictions == 0)[0]
		model = HuberRegressor(max_iter=1000)
		param_grid = {'epsilon': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]}

		grid_search = GridSearchCV(model, param_grid, cv=K)
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		y_pred = best_model.predict(X_test)
		y_pred[no_claim_indices] = 0

		result = cross_validate(best_model, X_train, y_train, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV MAE (RF): {-np.mean(result["test_score"])}')
		df= pd.DataFrame({'ClaimAmounts': y_pred})
		df.to_csv('rf_hr.csv', index_label='rowIndex')
	elif model_type == 'svm':
		svm_classifier = SVC(kernel='rbf', C=1.0)
		svm_classifier.fit(X_train, y_train_binary)

		binary_predictions = svm_classifier.predict(X_test)

		no_claim_indices = np.where(binary_predictions == 0)[0]
		model = HuberRegressor(max_iter=1000)
		param_grid = {'epsilon': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]}

		grid_search = GridSearchCV(model, param_grid, cv=K)
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		y_pred = best_model.predict(X_test)
		y_pred[no_claim_indices] = 0
		result = cross_validate(best_model, X_train, y_train, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV MAE (SVM): {-np.mean(result["test_score"])}')
		df= pd.DataFrame({'ClaimAmounts': y_pred})
		df.to_csv('svm_hr.csv', index_label='rowIndex')
	elif model_type == 'sgd':
		sgd_classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
		sgd_classifier.fit(X_train, y_train_binary)

		binary_predictions = sgd_classifier.predict(X_test)

		no_claim_indices = np.where(binary_predictions == 0)[0]
		model = HuberRegressor(max_iter=1000)
		param_grid = {'epsilon': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]}

		grid_search = GridSearchCV(model, param_grid, cv=K, scoring='neg_mean_absolute_error')
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		y_pred = best_model.predict(X_test)
		y_pred[no_claim_indices] = 0

		result = cross_validate(best_model, X_train, y_train, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV MAE (SGD): {-np.mean(result["test_score"])}')
		df= pd.DataFrame({'ClaimAmounts': y_pred})
		df.to_csv('sgd_hr.csv', index_label='rowIndex')
	elif model_type == 'ridge':
		ridge_classifier = RidgeClassifier(alpha=1.0)
		ridge_classifier.fit(X_train, y_train_binary)

		binary_predictions = ridge_classifier.predict(X_test)

		no_claim_indices = np.where(binary_predictions == 0)[0]
		model = HuberRegressor(max_iter=1000)
		param_grid = {'epsilon': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]}

		grid_search = GridSearchCV(model, param_grid, cv=K)
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		y_pred = best_model.predict(X_test)
		y_pred[no_claim_indices] = 0
		result = cross_validate(best_model, X_train, y_train, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV MAE (Ridge): {-np.mean(result["test_score"])}')
		df= pd.DataFrame({'ClaimAmounts': y_pred})
		df.to_csv('ridge_hr.csv', index_label='rowIndex')
   

def main():
	data = undersample("trainingset.csv", 0.2)
	X_train, y_train = split_data(data)
	# X_train, y_train = get_data(normalize=True)
	X_test = get_data(train=False, normalize=True)
	y_train_binary = (y_train > 0).astype(int)

	evaluate('rf', X_train, y_train, X_test, y_train_binary)
	evaluate('svm', X_train, y_train, X_test, y_train_binary)
	evaluate('sgd', X_train, y_train, X_test, y_train_binary)
	evaluate('ridge', X_train, y_train, X_test, y_train_binary)

if __name__ == "__main__":
    main()