from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


class NeuralNetwork:
    def __init__(self, input_shape, type="regressor"):
        self.type = type
        model = Sequential()
        model.add(Dense(20, activation="relu", input_shape=(input_shape,)))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation="relu"))
        model.add(Dropout(0.2))

        if type == "classifier":
            model.add(Dense(1, activation="sigmoid"))
            model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
        else:
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])

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
            return (self.model.predict(X) > 0.1).astype(int)

        return pred


def get_rf_model(type="regressor"):
    if type == "classifier":
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    else:
        return RandomForestRegressor(n_estimators=100, random_state=42)
