import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image


# Example SMILES
smiles_list = ['c1ccccc1', 'C1CCCCC1', 'C(C(=O)O)N']

# Convert SMILES to RDKit molecule objects
molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

# Draw molecules
mols_per_row = 3
img = Draw.MolsToGridImage(molecules, molsPerRow=mols_per_row, subImgSize=(200,200),
                            legends=[f'Molecule {i+1}' for i in range(len(molecules))], returnPNG=False)

plt.imshow(img)
plt.axis('off')  # Remove axis since it's not relevant for an image
plt.show()
plt.savefig('molecules.png')