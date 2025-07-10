import pandas as pd
from rdkit import Chem
from MacFrag import mol_with_atom_index, searchBonds
from rdkit.Chem.BRICS import BreakBRICSBonds
from tqdm import tqdm
from collections import Counter

def count_real_fragments(smiles, maxSR=8, minFragAtoms=1):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = mol_with_atom_index(mol)

        bonds = list(searchBonds(mol, maxSR=maxSR))
        broken = BreakBRICSBonds(mol, bonds=bonds)
        frags = Chem.GetMolFrags(broken, asMols=True)

        count = 0
        for frag in frags:
            atom_ids = {a.GetAtomMapNum() for a in frag.GetAtoms() if a.GetAtomMapNum() >= 0}
            if len(atom_ids) >= minFragAtoms:
                count += 1
        return count
    except:
        return None

def analyze_fragment_counts(csv_file, smiles_column="mol", maxSR=20, minFragAtoms=1):
    df = pd.read_csv(csv_file)
    frag_hist = Counter()
    total_valid = 0

    for smi in tqdm(df[smiles_column]):
        frag_count = count_real_fragments(smi, maxSR=maxSR, minFragAtoms=minFragAtoms)
        if frag_count is not None:
            frag_hist[frag_count] += 1
            total_valid += 1

    print(f"\n=== Fragment Count Distribution ===")
    for k in sorted(frag_hist.keys()):
        count = frag_hist[k]
        percent = 100 * count / total_valid
        print(f"{k} fragments: {count} molecules ({percent:.2f}%)")
    print(f"\nTotal valid molecules: {total_valid}")

# Example usage
if __name__ == "__main__":
    analyze_fragment_counts("raw/bbbp.csv", smiles_column="mol", maxSR=8, minFragAtoms=1)