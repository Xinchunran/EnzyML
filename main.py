import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import DataPreprocessor
from models import XGBoostRegressor, KNNRegressor, SVMRegressor
from metrics import evaluate


if __name__ == '__main__':

    # Input file and load data
    file = input('Enter data file name: ')
    df = pd.read_csv(file)

    # Preprocess data
    X, y = DataPreprocessor.preprocess_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost Model
    xgb_model = XGBoostRegressor(hyperparams=None)
    xgb_model.train(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print("XGBoost Model Evaluation:")
    evaluate(y_test, y_pred_xgb)

    # KNN Model
    knn_model = KNNRegressor(hyperparams=None)
    knn_model.train(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    print("\nKNN Model Evaluation:")
    evaluate(y_test, y_pred_knn)

    # SVM Model
    svm_model = SVMRegressor(hyperparams=None)
    svm_model.train(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print("\nSVM Model Evaluation:")
    evaluate(y_test, y_pred_svm)