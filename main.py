import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler

from preprocessing_data import get_data, process_data, get_test_data
from model import NeuralNetwork, get_rf_model, get_xgb_model


def split_data(data):
    X = data.drop(columns=['ClaimAmount'])
    Y = data['ClaimAmount']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Feature Scaling (Standardization)
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


def cv_nn(X, y, K):
  kf = KFold(K, shuffle=True, random_state=42)
  mae = []
 
  for train_idx, test_idx in kf.split(X):
    model = NeuralNetwork(X.shape[1])
    model.train(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
    loss, metrics = model.evaluate(X[test_idx], y[test_idx])
    mae.append(metrics)
    
  return np.mean(mae) 

def cv_two_stage_nn(K):
  kf = KFold(K, shuffle=True, random_state=42)
  clf_metrics = []
  reg_metrics = []
  model_metrics = []
  
  X, y = get_data(normalize=False)
 
  clf = get_rf_model(type='classifier')
  
  # nn
  # clf = NeuralNetwork(X.shape[1], type='classifier')
  reg = NeuralNetwork(X.shape[1], type='regressor')
 
  for train_idx, test_idx in kf.split(X):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
  
    X1_train, X1_test = X_train, X_test
    y1_train, y1_test = (y_train > 0).astype(int), (y_test > 0).astype(int)

    # sampling
    # smote_enn = SMOTEENN(random_state=42)
    # X1_train, y1_train = smote_enn.fit_resample(X1_train, y1_train)
  
    X2_train, X2_test = X_train[y_train > 0], X_test[y_test > 0]
    y2_train, y2_test = y_train[y_train > 0], y_test[y_test > 0]
  
    # normalise
    # scaler = StandardScaler()
    # X2_train, X2_test = scaler.fit_transform(X2_train), scaler.transform(X2_test)
  
    clf.fit(X1_train, y1_train)
    y1_pred = (clf.predict_proba(X1_test)[:, 1] > 0.25).astype(int)
    # y1_pred = clf.predict(X1_test)
    acc_score = f1_score(y1_test, y1_pred)
    print('clf', acc_score)
    clf_metrics.append(acc_score)
  
    # nn
    # clf.train(X1_train, y1_train, X1_test, y1_test)
    # loss, metrics = clf.evaluate(X1_test, y1_test)
    # y1_pred = clf.predict(X1_test).flatten()
    # clf_metrics.append(metrics)

    reg.train(X2_train, y2_train, X2_test, y2_test)
    loss, metrics = reg.evaluate(X2_test, y2_test)
    print('reg', metrics)
    reg_metrics.append(metrics)
  
    X_test_reg = X_test[y1_pred > 0]
    # normalise
    # X_test_reg = scaler.transform(X_test_reg)
    y_pred = np.zeros(y_test.shape)
    print(len(X_test_reg))
    print(np.sum(y_test > 0))
  
    if len(X_test_reg) > 0:
      y_pred_reg = reg.predict(X_test_reg).flatten()
      y_pred[y1_pred > 0] = y_pred_reg
 
    model_metrics.append(mean_absolute_error(y_test, y_pred))
  
  print(f'Classifier accuracy: {np.mean(clf_metrics)}')
  print(f'Regressor MAE: {np.mean(reg_metrics)}')
  print(f'Model MAE: {np.mean(model_metrics)}')
 
   
def cv_two_stage(K):
  kf = KFold(K, shuffle=True, random_state=42)
  clf_metrics = []
  reg_metrics = []
  model_metrics = []
  
  X, y = get_data(normalize=False)
 
  clf = get_rf_model(type='classifier')
  reg = get_rf_model(type='regressor')
  # reg = get_xgb_model()
 
  for train_idx, test_idx in kf.split(X):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
  
    X1_train, X1_test = X_train, X_test
    y1_train, y1_test = (y_train > 0).astype(int), (y_test > 0).astype(int)

    # sampling
    # smote_enn = SMOTEENN(random_state=42)
    # X1_train, y1_train = smote_enn.fit_resample(X1_train, y1_train)
  
    X2_train, X2_test = X_train[y_train > 0], X_test[y_test > 0]
    y2_train, y2_test = y_train[y_train > 0], y_test[y_test > 0]
  
    # normalise
    # scaler = StandardScaler()
    # X2_train, X2_test = scaler.fit_transform(X2_train), scaler.transform(X2_test)
  
    clf.fit(X1_train, y1_train)
    y1_pred = (clf.predict_proba(X1_test)[:, 1] > 0.25).astype(int)
    # y1_pred = clf.predict(X1_test)
    acc_score = f1_score(y1_test, y1_pred)
    print('clf', acc_score)
    clf_metrics.append(acc_score)

    reg.fit(X2_train, y2_train)
    y2_pred = reg.predict(X2_test)
    print('reg ', mean_absolute_error(y2_test, y2_pred))
    reg_metrics.append(mean_absolute_error(y2_test, y2_pred))
  
  
    X_test_reg = X_test[y1_pred > 0]
    # normalise
    # X_test_reg = scaler.transform(X_test_reg)
    y_pred = np.zeros(y_test.shape)
    print(len(X_test_reg))
    print(np.sum(y_test > 0))
  
    if len(X_test_reg) > 0:
      y_pred_reg = reg.predict(X_test_reg)
      y_pred[y1_pred > 0] = y_pred_reg
 
    model_metrics.append(mean_absolute_error(y_test, y_pred))
  
  print(f'Classifier accuracy: {np.mean(clf_metrics)}')
  print(f'Regressor MAE: {np.mean(reg_metrics)}')
  print(f'Model MAE: {np.mean(model_metrics)}')
  
  
def evaluate(model_type='lr'):
  K = 10
  
  if model_type == 'nn':
    X, y = get_data(normalize=True)
    err = cv_nn(X, y, K)
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
   
def train_nn():
  model_type = 'nn_two_stage3'	
  X, y = get_data(normalize=False)
  clf = get_rf_model(type='classifier')

  # nn
  # clf = NeuralNetwork(X.shape[1], type='classifier')
  reg = NeuralNetwork(X.shape[1], type='regressor')

  X_train, X_test, y_train, y_test = split_data(process_data('trainingset.csv'))
  X1_train, X1_test = X_train, X_test
  y1_train, y1_test = (y_train > 0).astype(int), (y_test > 0).astype(int)
  X2_train, X2_test = X_train[y_train > 0], X_test[y_test > 0]
  y2_train, y2_test = y_train[y_train > 0], y_test[y_test > 0]

  clf.fit(X1_train, y1_train)

  # nn
  # clf.train(X1_train, y1_train, X1_test, y1_test)

  reg.train(X2_train, y2_train, X2_test, y2_test)

  X_test_set = get_test_data(normalize=False)
  y1_test_pred = clf.predict(X_test_set)

  X_test_reg = X_test_set[y1_test_pred > 0]
  y_pred = np.zeros(X_test_set.shape[0])

  if len(X_test_reg) > 0:
    y_pred_reg = reg.predict(X_test_reg).flatten()
    y_pred[y1_test_pred > 0] = y_pred_reg

  pred = y_pred

  df = pd.DataFrame(pred, columns=['ClaimAmount'])
  df.insert(0, 'rowIndex', range(len(df)))
  df.to_csv(f'submission_{model_type}.csv', index=False)


def train(model_type='lr'):
  if model_type == 'nn':
    X_train, X_val, y_train, y_val = split_data(process_data('trainingset.csv'))
    model = NeuralNetwork(X_train.shape[1])
    model.train(X_train, y_train, X_val, y_val)
    X_test = get_test_data(normalize=True)
    pred = model.predict(X_test)
    
  elif model_type == 'rf':
    X, y = get_data(normalize=False)
    model = get_rf_model()
    model.fit(X, y)
    X_test = get_test_data(normalize=False)
    pred = model.predict(X_test)
    
  else:
    X, y = get_data(normalize=False)
 
    clf = get_rf_model(type='classifier')
    reg = get_rf_model(type='regressor')
  
    # reg = get_xgb_model()

    X_train, y_train = get_data(normalize=False)

  
    X1_train = X_train
    y1_train = (y_train > 0).astype(int)
    X2_train = X_train[y_train > 0]
    y2_train = y_train[y_train > 0]
  
    clf.fit(X1_train, y1_train)
    reg.fit(X2_train, y2_train)
  
    X_test_set = get_test_data(normalize=False)
    y1_test_pred = (clf.predict_proba(X_test_set)[:, 1] > 0.25).astype(int)

    X_test_reg = X_test_set[y1_test_pred > 0]
    y_pred = np.zeros(X_test_set.shape[0])
  
    if len(X_test_reg) > 0:
      y_pred_reg = reg.predict(X_test_reg)
      y_pred[y1_test_pred > 0] = y_pred_reg

    pred = y_pred
  
  df = pd.DataFrame(pred, columns=['ClaimAmount'])
  df.insert(0, 'rowIndex', range(len(df)))
  df.to_csv(f'submission_{model_type}.csv', index=False)
  

def main():
  # evaluate()
  # evaluate('nn') 
  # evaluate('rf')
  # cv_two_stage(10)
  # train('nn')
  # train('rf')
  # train('nn_two_stage2')
  # cv_two_stage_nn(10)
  train_nn()

if __name__ == "__main__":
    main()