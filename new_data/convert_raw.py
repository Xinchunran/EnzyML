import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

def convert_to_isomeric_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        isomeric_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return isomeric_smiles
    else:
        raise ValueError("Invalid SMILES string: '{}'".format(smiles))

def eyring_kcat(kcat: float, temp: float):
    R = 8.314 #(J/mol*K)
    dg_dag = -R * temp * math.log(kcat)
    dg_dd = dg_dag /4184 # to kcal/mol
    return dg_dd

def csv_reader(path):
    data = pd.read_csv(path)
    return data

def main():
   data = csv_reader("./data.csv")
   scaler = StandardScaler()
   ec_numbers = data["ECNumber"]
   values_ = data["Value"]
   names = data["Substrate"]
   smiles = data["Smiles"]
   seqs = data["Sequence"]
   iso_smiles = smiles.apply(convert_to_isomeric_smiles)
   data['iso_smiles'] = iso_smiles
   finalized_iso = pd.concat([iso_smiles, names, seqs, ec_numbers, seqs,
                              values], axis=1)
   finalized_iso.to_csv("updated_iso.csv")

if __name__ == "__main__":
    main()

