import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from keras.callbacks import EarlyStopping

from preprocessing_data import get_data
from model import NeuralNetwork, get_rf_model


def split_data(data):
    X = data.drop(columns=['ClaimAmount'])
    Y = data['ClaimAmount']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


def cv_nn(model, X, y, K):
	kf = KFold(K, shuffle=True, random_state=42)
	mae = []
 
	for train_idx, test_idx in kf.split(X):
		model.train(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
		loss, metrics = model.evaluate(X[test_idx], y[test_idx])
		mae.append(metrics)
		
	return np.mean(mae) 

def cv_rf_two_stage(K):
	kf = KFold(K, shuffle=True, random_state=42)
	clf_metrics = []
	reg_metrics = []
	metrics = []
	
	X, y = get_data(normalize=False)
 
	clf = get_rf_model(type='classifier')
	reg = get_rf_model(type='regressor')
 
	for train_idx, test_idx in kf.split(X):
		X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
		X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
  
		X1_train, X1_test = X_train, X_test
		y1_train, y1_test = (y_train > 0).astype(int), (y_test > 0).astype(int)
  
		X2_train, X2_test = X_train[y_train > 0], X_test[y_test > 0]
		y2_train, y2_test = y_train[y_train > 0], y_test[y_test > 0]
  
		clf.fit(X1_train, y1_train)
		y1_pred = clf.predict(X1_test)
		clf_metrics.append(accuracy_score(y1_test, y1_pred))

		reg.fit(X2_train, y2_train)
		y2_pred = reg.predict(X2_test)
		reg_metrics.append(mean_absolute_error(y2_test, y2_pred))
  
		X_test_reg = X_test[y1_pred > 0]
		y_pred = np.zeros(y_test.shape)

		if len(X_test_reg) > 0:
			y_pred_reg = reg.predict(X_test_reg)
			y_pred[y1_pred > 0] = y_pred_reg
 
		metrics.append(mean_absolute_error(y_test, y_pred))
  
	print(f'Classifier accuracy: {np.mean(clf_metrics)}')
	print(f'Regressor MAE: {np.mean(reg_metrics)}')
	print(f'Model MAE: {np.mean(metrics)}')
  
  
def evaluate(model_type='lr'):
	K = 10
  
	if model_type == 'nn':
		X, y = get_data(normalize=True)
		model = NeuralNetwork(X.shape[1])
		err = cv_nn(model, X, y, K)
		print(f'CV error (NN): {err}')
	elif model_type == 'rf':
		X, y = get_data(normalize=False)
		model = get_rf_model()
		result = cross_validate(model, X, y, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV error (RF): {-np.mean(result["test_score"])}')
	else:
		X, y = get_data(normalize=True)
		model = LinearRegression()
		result = cross_validate(model, X, y, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
		print(f'CV error (LR): {-np.mean(result["test_score"])}')
   

def main():
	# evaluate()
	evaluate('nn')
	# evaluate('rf')
	# cv_rf_two_stage(10)


if __name__ == "__main__":
    main()