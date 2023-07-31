import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

#The part read in the dataset
data = pd.read_csv("kcat_transferase.csv")
onehot_encoded_smiles = pd.get_dummies(data['smiles'].apply(list).apply(pd.Series).stack()).groupby(level=0).sum()


X = onehot_encoded_smiles
y = data["kcat"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVR()
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
mse_svm = mean_squared_error(y_test, y_pred_svm)
print("SVM Mean Squared Error:", mse_svm)


