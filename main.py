import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNet
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC

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


def evaluate(model_type, X_train, y_train, X_test, y_train_binary):
    K = 5
    if model_type == 'knn':
        clf = KNeighborsClassifier(n_neighbors=265)
        clf.fit(X_train, y_train_binary)

        binary_predictions = clf.predict(X_test)

        no_claim_indeces = np.where(binary_predictions == 0)[0]

        model = KNeighborsRegressor(n_neighbors=265)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_pred[no_claim_indeces] = 0

        df = pd.DataFrame({'ClaimAmounts': y_pred})
        df.to_csv('KNN.csv', index_label='rowIndex')
    elif model_type == 'hr':
        svm_model = SVC(kernel='linear', C=1.0)  # You can choose different kernels and hyperparameters
        svm_model.fit(X_train, y_train_binary)

        # Make predictions on the test set
        binary_predictions = svm_model.predict(X_test)

        no_claim_indeces = np.where(binary_predictions == 0)[0]

        huber_reg = HuberRegressor()
        param_grid = {'epsilon': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]}

        grid_search = GridSearchCV(huber_reg, param_grid, cv=K)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # result = cross_validate(best_model, X_train, y_train, cv=K, scoring='neg_mean_absolute_error',
        #                         return_train_score=True)
        # print(f'Average Cross-Validation Mean Squared Error: {-np.mean(result["test_score"])}')

        y_pred = best_model.predict(X_test)
        y_pred[no_claim_indeces] = 0

        df = pd.DataFrame({'ClaimAmounts': y_pred})
        df.to_csv('Huber.csv', index_label='rowIndex')
    elif model_type == 'en':
        gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gb_classifier.fit(X_train, y_train_binary)

        # Make predictions on the test set
        binary_predictions = gb_classifier.predict(X_test)
        no_claim_indeces = np.where(binary_predictions == 0)[0]

        elastic_net = ElasticNet()
        param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
        grid_search = GridSearchCV(elastic_net, param_grid, cv=K, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_pred[no_claim_indeces] = 0

        df = pd.DataFrame({'ClaimAmounts': y_pred})
        df.to_csv('Elastic_net.csv', index_label='rowIndex')


def main():
    X_train, y_train = get_data(normalize=True)
    X_test = get_data(train=False, normalize=True)
    y_train_binary = (y_train > 0).astype(int)

    # Classify with KNN and train with KNN
    # evaluate('knn', X_train, y_train, X_test, y_train_binary)
    # 194.77

    # Classify with SVC and train with Huber
    # evaluate('hr', X_train, y_train, X_test, y_train_binary)
    # 103.05

    # Classify with Gradient Boosting and train with Elastic Net
    evaluate('en', X_train, y_train, X_test, y_train_binary)
    # 194.74


if __name__ == "__main__":
    main()
