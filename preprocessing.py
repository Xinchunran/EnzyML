import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# from sklearn.model_selection import train_test_split

def preprocess_data(df):
    smiles_encoder = OneHotEncoder(sparse_output=False)
    smiles_encoded = smiles_encoder.fit_transform(df['smiles'].values.reshape(-1, 1))
    sequence_encoder = OneHotEncoder(sparse_output=False)
    sequence_encoded = sequence_encoder.fit_transform(df['enzyme_seq'].values.reshape(-1, 1))

    
    X = pd.concat([pd.DataFrame(smiles_encoded), pd.DataFrame(sequence_encoded)], axis=1)
    y = df['kcat'].values

    return X, y

if __name__ == '__main__':
    df = pd.read_csv('kcat_transferase.csv')
    X, y = preprocess_data(df)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
