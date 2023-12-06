import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from preprocessing import preprocess_data

class XGBoostRegressor:
    def __init__(self):
        self.model = XGBRegressor()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')

class KNNRegressor:
    def __init__(self):
        self.model = KNeighborsRegressor()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')

class SVMRegressor:
    def __init__(self):
        self.model = SVR()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('kcat_transferase.csv')

    # Preprocess data
    X, y = preprocess_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost Model
    xgb_model = XGBoostRegressor()
    xgb_model.train(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print("XGBoost Model Evaluation:")
    xgb_model.evaluate(y_test, y_pred_xgb)

    # KNN Model
    knn_model = KNNRegressor()
    knn_model.train(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    print("\nKNN Model Evaluation:")
    knn_model.evaluate(y_test, y_pred_knn)

    # SVM Model
    svm_model = SVMRegressor()
    svm_model.train(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print("\nSVM Model Evaluation:")
    svm_model.evaluate(y_test, y_pred_svm)