from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Concatenate
from keras.callbacks import EarlyStopping
from keras.optimizers import AdamW
from keras.metrics import F1Score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVR


class NeuralNetwork:
    def __init__(self, input_shape, type="regressor"):
        self.type = type
        model = Sequential()
        model.add(Dense(128, activation="relu", input_shape=(input_shape,)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))

        if type == "classifier":
            model.add(Dense(1, activation="sigmoid"))
            optimizer = AdamW(learning_rate=1e-3, weight_decay=4e-3)
            model.compile(
                optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
            )
        else:
            model.add(Dense(1))
            optimizer = AdamW(learning_rate=1e-4, weight_decay=4e-4)
            model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        self.model = model

    def train(self, X_train, y_train, X_val, y_val):
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=5)
        self.model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[es],
        )
      
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)
      
    def predict(self, X):
        pred = self.model.predict(X)
        
        if self.type == "classifier":
            return (self.model.predict(X) > 0.25).astype(int)

        return pred
    

class CombinedNeuralNetwork:
    def __init__(self, input_shape):
        base_model = Sequential([
            Dense(128, activation="relu", input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2)
        ])
        
        clf_layer = Dense(64, activation="relu")(base_model.output)
        clf_layer = Dropout(0.2)(clf_layer)
        clf_output = Dense(1, activation="sigmoid", name='clf_output')(clf_layer)
        
        reg_input = Concatenate()([base_model.output, clf_output])
        reg_layer = Dense(64, activation="relu")(reg_input)
        reg_layer = Dropout(0.2)(reg_layer)
        reg_output = Dense(1, name='reg_output')(reg_layer)
        
        self.model = Model(inputs=base_model.input, outputs=[clf_output, reg_output])
        
    def compile(self):
        optimizer = AdamW(learning_rate=1e-3, weight_decay=4e-3)
        self.model.compile(
            optimizer=optimizer, 
            loss={'clf_output': 'binary_crossentropy', 'reg_output': 'mse'},
            metrics={'clf_output': F1Score(average='macro'), 'reg_output': 'mae'}
        )
        
    def fit(self, X_train, y_train, X_val, y_val):
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=5, restore_best_weights=True)
        
        y_train_clf = (y_train > 0).astype('float32')
        y_val_clf = (y_val > 0).astype('float32')
        
        self.model.fit(
            X_train, {'clf_output': y_train_clf, 'reg_output': y_train},
            epochs=10,
            batch_size=32,
            verbose=0,
            validation_data=(X_val, {'clf_output': y_val_clf, 'reg_output': y_val}),
            callbacks=[es],
        )
        
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)
        
    def predict(self, X):
        return self.model.predict(X)


def get_rf_model(type="regressor"):
    if type == "classifier":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        return RandomForestRegressor(n_estimators=200, random_state=42)


def get_xgb_model(type="regressor", alpha=0.1):
    if type == 'classifier':
        return XGBClassifier(n_estimators = 200, random_state=42)
    
    return XGBRegressor(objective ='reg:squaredlogerror', alpha=alpha, n_estimators = 200, random_state=42)

def get_svr_model(k='poly', C=100, e=1, g='auto'):
    return SVR(kernel=k, C=C, epsilon=e, gamma=g)