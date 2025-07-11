import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from rdkit import Chem
from collections import defaultdict
import torch
from torch_geometric.data import Data
from rdkit.Chem import AllChem
import numpy as np
import rdkit.DataStructs as DataStructs

def get_ecfp(smiles, radius=2, n_bits=100):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)  # fallback
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# feature dim
ATOM_DIM = 101
BOND_DIM = 11
FG_DIM = 73
FG_EDGE_DIM = ATOM_DIM

ALLOWABLE_BOND_FEATURES = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
    'conjugated': ['T/F'],
    'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY']
}

PATT = {
    'HETEROATOM': '[!#6]',
    'DOUBLE_TRIPLE_BOND': '*=,#*',
    'ACETAL': '[CX4]([O,N,S])[O,N,S]'
}
PATT = {k: Chem.MolFromSmarts(v) for k, v in PATT.items()}


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = 'SINGLE'
        #raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_feature(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
            'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
            'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetFormalCharge(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()] +
        [atom.IsInRing()]
    )



def get_bond_feature(bond):
    #print(str(bond.GetBondType()))
    return np.array(
        one_of_k_encoding(str(bond.GetBondType()), ALLOWABLE_BOND_FEATURES['bond_type']) +
        [bond.GetIsConjugated()] +
        one_of_k_encoding(str(bond.GetStereo()), ALLOWABLE_BOND_FEATURES['stereo'])
    )


def get_fg_feature(fg_prop, fg=None):
    return np.array(
        one_of_k_encoding_unk(fg_prop['#C'], range(11)) +  # 0-10, 10+
        one_of_k_encoding_unk(fg_prop['#O'], range(6)) +  # 0-5, 5+
        one_of_k_encoding_unk(fg_prop['#N'], range(6)) +
        one_of_k_encoding_unk(fg_prop['#P'], range(6)) +
        one_of_k_encoding_unk(fg_prop['#S'], range(6)) +
        [fg_prop['#X'] > 0] +
        [fg_prop['#UNK'] > 0] +
        one_of_k_encoding_unk(fg_prop['#SINGLE'], range(11)) +  # 0-10, 10+
        one_of_k_encoding_unk(fg_prop['#DOUBLE'], range(8)) +  # 0-6, 6+
        one_of_k_encoding_unk(fg_prop['#TRIPLE'], range(8)) +
        one_of_k_encoding_unk(fg_prop['#AROMATIC'], range(8)) +
        [fg_prop['IsRing']]
    )


def mol_to_graphs(mol):
    fgs = []  # Function Groups

    # <editor-fold desc="identify and merge rings">
    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]  # get simple rings
    flag = True  # flag == False: no rings can be merged
    while flag:
        flag = False
        for i in range(len(rings)):
            if len(rings[i]) == 0: continue
            for j in range(i+1, len(rings)):
                shared_atoms = rings[i] & rings[j]
                if len(shared_atoms) > 2:
                    rings[i].update(rings[j])
                    rings[j] = set()
                    flag = True
    rings = [r for r in rings if len(r) > 0]
    # </editor-fold>

    # <editor-fold desc="identify functional atoms and merge connected ones">
    marks = set()
    for patt in PATT.values():  # mark functional atoms
        for sub in mol.GetSubstructMatches(patt):
            marks.update(sub)
    atom2fg = [[] for _ in range(mol.GetNumAtoms())]  # atom2fg[i]: list of i-th atom's FG idx
    for atom in marks:  # init: each marked atom is a FG
        fgs.append({atom})
        atom2fg[atom] = [len(fgs)-1]
    for bond in mol.GetBonds():  # merge FGs
        if bond.IsInRing(): continue
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in marks and a2 in marks:  # a marked atom should only belong to a FG, if atoms are both marked, merge their FGs into a FG
            assert a1 != a2
            assert len(atom2fg[a1]) == 1 and len(atom2fg[a2]) == 1
            # merge a2' FG to a1's FG
            fgs[atom2fg[a1][0]].update(fgs[atom2fg[a2][0]])
            fgs[atom2fg[a2][0]] = set()
            atom2fg[a2] = atom2fg[a1]
        elif a1 in marks:  # only one atom is marked, add neighbour atom to its FG as its environment
            assert len(atom2fg[a1]) == 1
            # add a2 to a1's FG
            fgs[atom2fg[a1][0]].add(a2)
            atom2fg[a2].extend(atom2fg[a1])
        elif a2 in marks:
            # add a1 to a2's FG
            assert len(atom2fg[a2]) == 1
            fgs[atom2fg[a2][0]].add(a1)
            atom2fg[a1].extend(atom2fg[a2])
        else:  # both atoms are unmarked, i.e. a trivial C-C single bond
            # add single bond to fgs
            fgs.append({a1, a2})
            atom2fg[a1].append(len(fgs)-1)
            atom2fg[a2].append(len(fgs)-1)
    tmp = []
    for fg in fgs:
        if len(fg) == 0: continue
        if len(fg) == 1 and mol.GetAtomWithIdx(list(fg)[0]).IsInRing(): continue  # single atom FGs: 1. marked atom only in ring: remove; 2. ion or simple substance: retain
        tmp.append(fg)
    fgs = tmp
    # </editor-fold>

    fgs.extend(rings)  # final FGs: rings + FGs (not in rings)
    atom2fg = [[] for _ in range(mol.GetNumAtoms())]
    for i in range(len(fgs)): # update atom2fg
        for atom in fgs[i]:
            atom2fg[atom].append(i)

    # <editor-fold desc="generate atom-level graph and get FG's properties">
    atom_features, bond_list, bond_features = [], [], []
    fg_prop = [defaultdict(int) for _ in range(len(fgs))]  # prop: atom: #C, #O, #N, #P, #S, #X, #UNK; bond: #SINGLE, #DOUBLE, #TRIPLE, #AROMATIC, IsRing
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_feature(atom).tolist())
        elem = atom.GetSymbol()
        if elem in ['C', 'O', 'N', 'P', 'S']:
            key = '#'+elem
        elif elem in ['F', 'Cl', 'Br', 'I']:
            key = '#X'
        else:
            key = '#UNK'
        for fg_idx in atom2fg[atom.GetIdx()]:
            fg_prop[fg_idx][key] += 1
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_list.extend([[a1, a2], [a2, a1]])
        bond_features.extend([get_bond_feature(bond).tolist()] * 2)
        key = '#'+str(bond.GetBondType())
        for fg_idx in (set(atom2fg[a1]) & set(atom2fg[a2])):
            fg_prop[fg_idx][key] += 1
            if bond.IsInRing():
                fg_prop[fg_idx]['IsRing'] = 1
    # </editor-fold>

    # <editor-fold desc="generate FG-level graph">
    #print(fgs)
    #print(fg_prop)
    fg_features, fg_edge_list, fg_edge_features = [], [], []
    for i in range(len(fgs)):
        fg_features.append(get_fg_feature(fg_prop[i]).tolist())
        for j in range(i+1, len(fgs)):
            shared_atoms = list(fgs[i] & fgs[j])
            if len(shared_atoms) > 0:
                fg_edge_list.extend([[i, j], [j, i]])
                if len(shared_atoms) == 1:
                    fg_edge_features.extend([atom_features[shared_atoms[0]]] * 2)
                else:  # two rings shared 2 atoms, i.e. 1 edge
                    assert len(shared_atoms) == 2
                    ef = [(i+j)/2 for i, j in zip(atom_features[shared_atoms[0]], atom_features[shared_atoms[1]])]
                    fg_edge_features.extend([ef] * 2)
    # </editor-fold>

    atom2fg_list = []
    for fg_idx in range(len(fgs)):
        for atom_idx in fgs[fg_idx]:
            atom2fg_list.append([atom_idx, fg_idx])

    # Now extract atom subgraphs for each FG as separate PyG Data objects
    fragment_subgraphs = []
    for fg in fgs:
        fg_atom_indices = sorted(list(fg))


        if len(fg_atom_indices) == 0:
            continue
            
        # Build subgraph edge index and edge attr restricted to these atoms
        # Map old atom idx to new indices for subgraph
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(fg_atom_indices)}

        edge_index_list = []
        edge_attr_list = []
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in fg_atom_indices and a2 in fg_atom_indices:
                edge_index_list.extend([[idx_map[a1], idx_map[a2]], [idx_map[a2], idx_map[a1]]])
                bf = get_bond_feature(bond)
                edge_attr_list.extend([bf.tolist(), bf.tolist()])

        fg_atom_feats = np.array([get_atom_feature(mol.GetAtomWithIdx(idx)) for idx in fg_atom_indices])
        x = torch.tensor(fg_atom_feats, dtype=torch.float)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous() if edge_index_list else torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float) if edge_attr_list else torch.empty((0, BOND_DIM), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        fragment_subgraphs.append(data)
    
    return atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list, fragment_subgraphs


if __name__ == '__main__':
    smiles = 'C[C@@H](O[C@H]1OCCN(CC2=NNC(=O)N2)[C@H]1C1=CC=C(F)C=C1)C1=CC(=CC(=C1)C(F)(F)F)C(F)(F)F'
    s = 'C/C=C/1\C[C@H]2[C@H](NC3=CC=CC=C3C(=O)N2C1)O'
    mol = Chem.MolFromSmiles(s)
    atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list = mol_to_graphs(mol)
    print(len(fg_features),fg_edge_list)
    pass
