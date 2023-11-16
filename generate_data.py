from rdkit import Chem
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('zinc_250k.csv')  # Replace with your file path
smiles_column = 'smiles'  # Replace with the name of the column containing SMILES strings

def get_atom_features(mol):
    """ Returns a list of atom features for each atom in the molecule """
    atom_features = []
    for atom in mol.GetAtoms():
        # Example features: atomic number, degree, hybridization
        features = [atom.GetAtomicNum(), atom.GetDegree(), atom.GetHybridization()]
        atom_features.append(features)
    return atom_features

n = 38 #maximum number of nodes
b = 9 #node types
c = 3 #edge types

X = np.zeors((n,b+1))
A = np.zeros((n,n,c+1))

# Kekulize each molecule
for index, row in df.iterrows():
    smiles = row[smiles_column]
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        Chem.Kekulize(mol)

        #get feature matrix
        atom_features = np.array(get_atom_features(mol))

        print(atom_features)
        
        #create adjacency matrix

        num_atoms = mol.GetNumAtoms()
        adj_matrix = np.zeros((num_atoms, num_atoms), dtype=int)

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj_matrix[start, end] = adj_matrix[end, start] = 1
            #print(adj_matrix)