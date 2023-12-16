from rdkit import Chem
import pandas as pd
import numpy as np
import pickle as pkl
import torch

# Load the dataset
df = pd.read_csv('zinc_250k.csv')  # Replace with your file path
smiles_column = 'smiles'  # Replace with the name of the column containing SMILES strings



n = 38 #maximum number of nodes
b = 9 #node types
c = 3 #edge types

atom_set = set()
edge_set = set()
for i, row in df.iterrows():
    if len(atom_set) == 9 and len(edge_set) == 3:
        break
    smiles = row[smiles_column]
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.Kekulize(mol)
        for atom in mol.GetAtoms():
            atom_set.add(atom.GetAtomicNum())
        for bond in mol.GetBonds():
            edge_set.add(bond.GetBondTypeAsDouble())

atom_type_dict = {} #maps atomic number to index
for i, atom_num in enumerate(sorted(list(atom_set))):
    atom_type_dict[atom_num] = i 

bond_type_dict = {} #maps bond type to index
for i, bond_type in enumerate(sorted(list(edge_set))):
    bond_type_dict[bond_type] = i        

# Kekulize each molecule
X_list = []
edge_attrs = []
edge_indices = []
count = 1
for index, row in df.iterrows():
    if index>0 and (index+1) % 2500 == 0:
        with open(f'data/features/X_{count}.pkl', 'wb') as f:
            pkl.dump(X_list, f)
        with open(f'data/edge_indices/edge_indices_{count}.pkl', 'wb') as f:
            pkl.dump(edge_indices, f)
        with open(f'data/edge_attrs/edge_attrs_{count}.pkl', 'wb') as f:
            pkl.dump(edge_attrs, f)
        X_list = []
        edge_attrs = []
        edge_indices = []

        count += 1
        print(f'Saved {index+1} molecules')

    smiles = row[smiles_column]
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        Chem.Kekulize(mol)

        X = np.zeros((n,b))
        A = np.zeros((n,n,c))

        #populate feature matrix
        for i, atom in enumerate(mol.GetAtoms()):
            X[i, atom_type_dict[atom.GetAtomicNum()]] = 1
        #X += np.random.uniform(0,1,X.shape)

        #populate adjacency matrix
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            A[start, end, bond_type_dict[bond.GetBondTypeAsDouble()]] = 1
            A[end, start, bond_type_dict[bond.GetBondTypeAsDouble()]] = 1

        #Convert adjacency matrix to edge list and edge attributes
        edge_list = []
        edge_attr_list = []

        for i in range(n):
            for j in range(n):
                for k in range(c):
                    if A[i, j, k] != 0:  # Check if there is an edge of type k from i to j
                        edge_list.append((i, j))
                        edge_attr = [0] * c
                        edge_attr[k] = 1  # One-hot encoding for edge type
                        edge_attr_list.append(edge_attr)
        edge_array = np.array(edge_list)
        edge_attr_array = np.array(edge_attr_list)

        X_list.append(X)
        edge_attrs.append(edge_attr_array)
        edge_indices.append(edge_array)
        #edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        #edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)



        

if len(X_list) > 0:
    with open(f'data/features/X_{count}.pkl', 'wb') as f:
        pkl.dump(X_list, f)
    with open(f'data/edge_indices/edge_indices_{count}.pkl', 'wb') as f:
        pkl.dump(edge_indices, f)
    with open(f'data/edge_attrs/edge_attrs_{count}.pkl', 'wb') as f:
        pkl.dump(edge_attrs, f)
    print(f'Saved {index+1} molecules')

# Save the atom and bond type dictionaries
with open('data/atom_types.pkl', 'wb') as f:
    pkl.dump(atom_type_dict, f)

with open('data/bond_types.pkl', 'wb') as f:
    pkl.dump(bond_type_dict, f)
        

print("FINISHED GENERATING DATA TENSORS AND DICTIONARIES")