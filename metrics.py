from sklearn.metrics import mean_squared_error, r2_score

def evaluate(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')