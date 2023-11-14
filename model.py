from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
  
def get_nn_model(input_shape, type='regressor'):
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    
    if type == 'classifier':
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(1))
        
    
    return model
  
def get_rf_model(type='regressor'):
    if type == 'classifier':
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    else:
        return RandomForestRegressor(n_estimators=100, random_state=42)