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
        return -1  # Invalid molecule

    # 1. Chemical Validity Check
    reward += 1

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
    reward = evaluate_molecule('CCO')
    print("Reward:", reward)