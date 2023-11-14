import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNet
from sklearn.neighbors import KNeighborsRegressor

import preprocessing_data
from model import get_nn_model
from preprocessing_data import get_data, process_data


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


def evaluate(model_type):
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
    elif model_type == 'lr':
        X, y = get_data(normalize=True)
        model = LinearRegression()
        result = cross_validate(model, X, y, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
        print(f'CV error (LR): {-np.mean(result["test_score"])}')
    elif model_type == 'knn':
        X_train, y_train = get_data(normalize=True)
        X_test = get_data(train=False)
        model = KNeighborsRegressor(n_neighbors=265)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Predictions on the test set:", y_pred)
        df = pd.DataFrame({'ClaimAmounts': y_pred})
        df.to_csv('KNN.csv', index_label='rowIndex')

        # result = cross_validate(model, X_train, y_train, cv=K, scoring='neg_mean_absolute_error', return_train_score=True)
        # print(f'CV error (KNN) {-np.mean(result["test_score"])}')
    elif model_type == 'hr':
        X_train, y_train = get_data(normalize=True)
        X_test = get_data(train=False)
        huber_reg = HuberRegressor()
        param_grid = {'epsilon': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]}

        grid_search = GridSearchCV(huber_reg, param_grid, cv=K)
        grid_search.fit(X_train, y_train)
        # best_epsilon = grid_search.best_params_['epsilon']
        best_model = grid_search.best_estimator_

        result = cross_validate(best_model, X_train, y_train, cv=K, scoring='neg_mean_absolute_error',
                                return_train_score=True)
        print(f'Average Cross-Validation Mean Squared Error: {-np.mean(result["test_score"])}')

        y_pred = best_model.predict(X_test)
        print("Predictions on the test set:", y_pred)

        df = pd.DataFrame({'ClaimAmounts': y_pred})
        df.to_csv('Huber.csv', index_label='rowIndex')
    elif model_type == 'en':
        X_train, y_train = get_data(normalize=True)
        X_test = get_data(train=False)

        elastic_net = ElasticNet()
        param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
        grid_search = GridSearchCV(elastic_net, param_grid, cv=K, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)

        best_alpha = grid_search.best_params_['alpha']
        print(f"Best Alpha: ${best_alpha}")
        best_model = grid_search.best_estimator_
        result = cross_validate(best_model, X_train, y_train, cv=K, scoring='neg_mean_absolute_error')
        print("Average Mean Absolute Error:", -np.mean(result['test_score']))

        y_pred = best_model.predict(X_test)
        print("Predictions on the test set:", y_pred)

        df = pd.DataFrame({'ClaimAmounts': y_pred})
        df.to_csv('Elastic_net.csv', index_label='rowIndex')


def main():
    # evaluate('lr')
    # evaluate('nn')
    # evaluate('rf')
    evaluate('knn')
    # 194.77

    # evaluate('hr')
    # 103.05

    evaluate('en')
    # 194.74


if __name__ == "__main__":
    main()
