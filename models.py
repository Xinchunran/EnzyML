from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

class XGBoostRegressor:
    def __init__(self, hyperparams):
        if hyperparams is None:
            hyperparams = {}
        self.model = XGBRegressor(**hyperparams)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class KNNRegressor:
    def __init__(self, hyperparams):
        if hyperparams is None:
            hyperparams = {}
        self.model = KNeighborsRegressor(**hyperparams)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class SVMRegressor:
    def __init__(self, hyperparams):
        if hyperparams is None:
            hyperparams = {}
        self.model = SVR(**hyperparams)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
