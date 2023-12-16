from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

def evaluate_molecule(smiles, perform_conformer_analysis=False):
    # Initialize reward
    reward = 0

    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -2  # Invalid molecule

    # 1. Chemical Validity Check
    reward += 0

    # 2. Stereochemistry Check
    if Chem.FindMolChiralCenters(mol, includeUnassigned=True):
        reward += 1

    # 3. Ring Analysis (penalize small rings)
    if any(len(ring) < 5 for ring in mol.GetRingInfo().AtomRings()):
        reward -= 0.5

    # 4. Molecular Weight and Descriptors
    mw = Descriptors.MolWt(mol)
    if 160 <= mw <= 480:  # Common drug-like range
        reward += 1

    # Adding logP check
    logP = Descriptors.MolLogP(mol)
    if -0.4 <= logP <= 5.6:  # Commonly accepted range for drug-likeness
        reward += 1
    else:
        reward -= 1

    # 5. Substructure Search (penalize toxic functional groups)
    nitro = Chem.MolFromSmarts('[N+](=O)[O-]')
    if mol.HasSubstructMatch(nitro):
        reward -= 1

    # 6. Conformer Generation and Analysis
    if perform_conformer_analysis:
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol) != -1:
            reward += 1
        else:
            reward -= 1

    return reward

if __name__ == '__main__':
    # Test the function
    test_molecules = ['[H]B([Be][Li])[Be][Li][BeH]', '[H]N1O[Li]12([H])(B)[Li]1[Li]C12', '[H]B(BN)C(N)[Li]([BeH])(B)N', '[H][Be][Li]=[Li]12([BeH])([BeH])(B)(BC([H])([H])B)[Be]B1C2', '[H][Li]12=[Li]([Be]1)O2', '[H][Li]=[Li]([H])([H])N[Li]([H])#[Li](B)O', '[H]B1[Be]N=[Li]23([H])(=[Li])#[Li]4([H])(C[Li](BO)(=B2)C4(O[BeH])C3([BeH])[BeH])N1', '[H]B([H])[BeH]', '[H]B[Be]B[Li]([H])B1CN[Li]2([H])([Li](O)O)([Be]B2)[Be][Li]1(B)(C)(O)O']
    
    for smiles in test_molecules:
        print(evaluate_molecule(smiles))
    #reward = evaluate_molecule('[H][Li]=[Li]([H])([H])N[Li]([H])#[Li](B)O')
    #print("Reward:", reward)