import pandas as pd
from preprocessing import DataPreprocessor
from models import XGBoostRegressor, KNNRegressor, SVMRegressor
from metrics import evaluate

# Example for adding specific hyperparameters 
xgb_regressor = XGBoostRegressor(hyperparams={'n_estimators': 100, 'max_depth': 3})
knn_regressor = KNNRegressor(hyperparams={'n_neighbors': 5})
svm_regressor = SVMRegressor(hyperparams={'C': 1.0, 'kernel': 'linear'})

if __name__ == '__main__':
    file = input('Enter file name: ')

    # Load data
    df = pd.read_csv(file)

    # Preprocess data
    X_train, X_test, y_train, y_test = DataPreprocessor.splits(df)

    # XGBoost Model
    #xgb_model = XGBoostRegressor(hyperparams=None)
    xgb_model = xgb_regressor
    xgb_model.train(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print("XGBoost Model Evaluation:")
    evaluate(y_test, y_pred_xgb)

    # KNN Model
    #knn_model = KNNRegressor(hyperparams=None)
    knn_model = knn_regressor
    knn_model.train(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    print("\nKNN Model Evaluation:")
    evaluate(y_test, y_pred_knn)

    # SVM Model
    #svm_model = SVMRegressor(hyperparams=None)
    svm_model = svm_regressor
    svm_model.train(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print("\nSVM Model Evaluation:")
    evaluate(y_test, y_pred_svm)


    
 

