import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

def rf_hr():
	X_train, y_train = get_data(normalize=True)
	X_test = get_data(train=False, normalize=True)
	y_train_binary = (y_train > 0).astype(int)
	rf = RandomForestClassifier(n_estimators=200, random_state=0)

	rf.fit(X_train, y_train_binary)
	y_pred = rf.predict(X_test)

	zero_indices_test = np.where(y_pred == 0)[0]
	zero_indices_train = np.where(y_train_binary == 0)[0]

	df = pd.read_csv('trainingset.csv')
	data = df.drop(columns=['rowIndex'])
	data.dropna(inplace=True)
	X = data.drop(columns=['ClaimAmount'])
	y = data['ClaimAmount']
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	x_non_zero = X[np.where(y > 0)[0]]
	y_non_zero = y[np.where(y > 0)[0]]
	print("Huber Regression")
	model = HuberRegressor(max_iter=1000)
	model.fit(x_non_zero, y_non_zero)

	# training predictions
	y_pred_train = model.predict(X_train)
	y_pred_train[zero_indices_train] = 0
	df = pd.DataFrame({'ClaimAmounts': y_pred_train})
	df.to_csv('rf_hr_train.csv', index_label='rowIndex')

	print("training mean", np.sum(y_train)/len(y_train))
	print("training mae: ", np.sum(np.abs(y_pred_train - y_train))/len(y_train))
	print("training f1 score: ", f1_score(y_train_binary, rf.predict(X_train)))
	# test predictions
	y_pred = model.predict(X_test)
	y_pred[zero_indices_test] = 0
	df = pd.DataFrame({'ClaimAmounts': y_pred})
	df.to_csv('rf_hr.csv', index_label='rowIndex')

def rf_rf():
	X_train, y_train = get_data(normalize=True)
	X_test = get_data(train=False, normalize=True)
	y_train_binary = (y_train > 0).astype(int)
	rf = RandomForestClassifier(n_estimators=200, random_state=0)

	rf.fit(X_train, y_train_binary)
	y_pred = rf.predict(X_test)

	zero_indices_test = np.where(y_pred == 0)[0]
	zero_indices_train = np.where(y_train_binary == 0)[0]

	df = pd.read_csv('trainingset.csv')
	data = df.drop(columns=['rowIndex'])
	data.dropna(inplace=True)
	X = data.drop(columns=['ClaimAmount'])
	y = data['ClaimAmount']
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	x_non_zero = X[np.where(y > 0)[0]]
	y_non_zero = y[np.where(y > 0)[0]]
	print("Random Forest")
	model = RandomForestRegressor(n_estimators=200, random_state=42)
	model.fit(x_non_zero, y_non_zero)

	# training predictions
	y_pred_train = model.predict(X_train)
	y_pred_train[zero_indices_train] = 0
	df = pd.DataFrame({'ClaimAmounts': y_pred_train})
	df.to_csv('rf_rf_train.csv', index_label='rowIndex')

	print("training mean", np.sum(y_train)/len(y_train))
	print("training mae: ", np.sum(np.abs(y_pred_train - y_train))/len(y_train))
	print("training f1 score: ", f1_score(y_train_binary, rf.predict(X_train)))
	# test predictions
	y_pred = model.predict(X_test)
	y_pred[zero_indices_test] = 0
	df = pd.DataFrame({'ClaimAmounts': y_pred})
	df.to_csv('rf_rf.csv', index_label='rowIndex')

def rf_svm():
	X_train, y_train = get_data(normalize=True)
	X_test = get_data(train=False, normalize=True)
	y_train_binary = (y_train > 0).astype(int)
	rf = RandomForestClassifier(n_estimators=200, random_state=0)

	rf.fit(X_train, y_train_binary)
	y_pred = rf.predict(X_test)

	zero_indices_test = np.where(y_pred == 0)[0]
	zero_indices_train = np.where(y_train_binary == 0)[0]

	df = pd.read_csv('trainingset.csv')
	data = df.drop(columns=['rowIndex'])
	data.dropna(inplace=True)
	X = data.drop(columns=['ClaimAmount'])
	y = data['ClaimAmount']
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	x_non_zero = X[np.where(y > 0)[0]]
	y_non_zero = y[np.where(y > 0)[0]]
	print("SVM")
	model = svm.SVR()
	model.fit(x_non_zero, y_non_zero)

	# training predictions
	y_pred_train = model.predict(X_train)
	y_pred_train[zero_indices_train] = 0
	df = pd.DataFrame({'ClaimAmounts': y_pred_train})
	df.to_csv('rf_svm_train.csv', index_label='rowIndex')

	print("training mean", np.sum(y_train)/len(y_train))
	print("training mae: ", np.sum(np.abs(y_pred_train - y_train))/len(y_train))
	print("training f1 score: ", f1_score(y_train_binary, rf.predict(X_train)))
	# test predictions
	y_pred = model.predict(X_test)
	y_pred[zero_indices_test] = 0
	df = pd.DataFrame({'ClaimAmounts': y_pred})
	df.to_csv('rf_svm.csv', index_label='rowIndex')

def cv(model, X, y, K):
	kf = KFold(K, shuffle=True, random_state=42)
	mae = []
	
	for train_idx, test_idx in kf.split(X):
		model.compile(optimizer='adam', loss='mse', metrics=['mae'])
		model.fit(X[train_idx], y[train_idx], epochs=10, batch_size=32, verbose=1)
		loss, metrics = model.evaluate(X[test_idx], y[test_idx], verbose=1)
		mae.append(metrics)
		
	return np.mean(mae) 

def oversample(filepath, oversample_rate):
	df = pd.read_csv(filepath)
	data = df.drop(columns=['rowIndex'])
	data.dropna(inplace=True)
	majority = data[data['ClaimAmount'] == 0]
	minority = data[data['ClaimAmount'] != 0]

	minority_count = int((len(majority)*oversample_rate)-len(majority))
	minority = minority.sample(minority_count, replace=True)
	oversampled_data = pd.concat([majority, minority])
	return oversampled_data
   
def f1_score(y_true, y_pred):
	tp = np.sum(y_true * y_pred)
	fp = np.sum((1 - y_true) * y_pred)
	fn = np.sum(y_true * (1 - y_pred))
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	return 2 * precision * recall / (precision + recall)

def main():
	rf_hr()
	rf_rf()
	rf_svm()


if __name__ == "__main__":
    main()