import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class DataPreprocessor:
    def __init__(self):
        pass

    def preprocess_data(df):
        # SMILES strings
        smiles_encoder = OneHotEncoder(sparse_output=False)
        smiles_encoded = smiles_encoder.fit_transform(df['smiles'].values.reshape(-1, 1))
        smiles_columns = [f"smiles_{i}" for i in range(smiles_encoded.shape[1])]

        # Enzyme sequence
        sequence_encoder = OneHotEncoder(sparse_output=False)
        sequence_encoded = sequence_encoder.fit_transform(df['enzyme_seq'].values.reshape(-1, 1))
        sequence_columns = [f"sequence_{i}" for i in range(sequence_encoded.shape[1])]

        # Concatenate one-hot encoded features
        X = pd.concat([pd.DataFrame(smiles_encoded, columns=smiles_columns), 
                    pd.DataFrame(sequence_encoded, columns=sequence_columns)], axis=1)
        
        # Get target (Kcat)
        y = df['kcat'].values

        return X, y

