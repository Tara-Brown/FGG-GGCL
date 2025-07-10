import os
from random import Random
import numpy as np
import re
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from MacFrag import MacFrag
#from MacFrag import parse_args,
from utils import get_task_names
from torch_geometric.data import DataLoader
from config import _C
from torch.serialization import add_safe_globals
from rdkit import RDLogger
import matplotlib.pyplot as plt
from MacFrag import mol_with_atom_index, searchBonds
from rdkit import Chem
from MacFrag import mol_with_atom_index, searchBonds
from rdkit.Chem.BRICS import BreakBRICSBonds
RDLogger.DisableLog('rdApp.*')
  
fun_smarts = {
    'Hbond_donor': '[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]',
    'Hbond_acceptor': '[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&X2&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]',
    'Basic': '[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),$([n;X2;+0;-0])]',
    'Acid': '[C,S](=[O,S,P])-[O;H1,-1]',
    'Halogen': '[F,Cl,Br,I]'
}
FunQuery = dict([(pharmaco, Chem.MolFromSmarts(s)) for (pharmaco, s) in fun_smarts.items()])


def remove_sym(smiles):
    pattern = r'\[[^\[\]]+\]'
    smiles = re.sub(pattern, '', str(smiles))
    return smiles


def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def tag_pharmacophore(mol):
    for fungrp, qmol in FunQuery.items():
        matches = mol.GetSubstructMatches(qmol)
        match_idxes = []
        for mat in matches:
            match_idxes.extend(mat)
        for i, atom in enumerate(mol.GetAtoms()):
            tag = '1' if i in match_idxes else '0'
            atom.SetProp(fungrp, tag)
    return mol


def tag_scaffold(mol):
    core = MurckoScaffold.GetScaffoldForMol(mol)
    match_idxes = mol.GetSubstructMatch(core)
    for i, atom in enumerate(mol.GetAtoms()):
        tag = '1' if i in match_idxes else '0'
        atom.SetProp('Scaffold', tag)
    return mol


def atom_attr(mol, explicit_H=False, use_chirality=True, pharmaco=True, scaffold=True):
    if pharmaco:
        mol = tag_pharmacophore(mol)
    if scaffold:
        mol = tag_scaffold(mol)

    feat = []
    for i, atom in enumerate(mol.GetAtoms()):
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other'
             ]) + onehot_encoding_unk(atom.GetDegree(),
                                      [0, 1, 2, 3, 4, 5, 'other']) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  onehot_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'
                  ]) + [atom.GetIsAromatic()]
        if not explicit_H:
            results = results + onehot_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + onehot_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [0, 0] + [atom.HasProp('_ChiralityPossible')]
        if pharmaco:
            results = results + [int(atom.GetProp('Hbond_donor'))] + [int(atom.GetProp('Hbond_acceptor'))] + \
                      [int(atom.GetProp('Basic'))] + [int(atom.GetProp('Acid'))] + \
                      [int(atom.GetProp('Halogen'))]
        if scaffold:
            results = results + [int(atom.GetProp('Scaffold'))]
        feat.append(results)

    return np.array(feat)


def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)


def frag_info(frag):
    cluster_idx = []
    Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False, frags=cluster_idx)
    fra_edge_index, fra_edge_attr = bond_attr(frag)
    cluster_idx = torch.LongTensor(cluster_idx)
    return fra_edge_index, fra_edge_attr, cluster_idx


class MolData(Data):
    def __init__(self, fra_edge_index=None, fra_edge_attr=None, cluster_index=None, **kwargs):
        super(MolData, self).__init__(**kwargs)
        self.cluster_index = cluster_index
        self.fra_edge_index = fra_edge_index
        self.fra_edge_attr = fra_edge_attr

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cluster_index':
            return int(self.cluster_index.max()) + 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


def read_data(target):
    df = pd.read_csv('../.csv')
    label = df[target].tolist()
    smiles = df['smiles'].tolist()
    return smiles, label


class MolDataset(InMemoryDataset):

    def __init__(self, root, dataset, task_type, tasks, logger=None,
                 transform=None, pre_transform=None, pre_filter=None):

        self.tasks = tasks
        self.dataset = dataset
        self.task_type = task_type
        self.logger = logger
        # self.target = target
        super(MolDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # Allow MolData class to be unpickled
        add_safe_globals([MolData])
        # Load the processed dataset fully (not weights-only)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    @property
    def raw_file_names(self):
        return ['{}.csv'.format(self.dataset)]

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    def pro_smi(self):
        # Read from the hard‑coded MacFrag opt.input_file (e.g. 'raw/BBBP.csv')
        df = pd.read_csv(opt.input_file)
        # Adjust column names to match your CSV: here 'mol' and 'Class'
        smiles = df['mol'].tolist()
        labels = df['Class'].tolist()
        return df, smiles, labels

    def pro_nan(self, labels, default_value=1):
        pro_labels = []
        for label in labels:
            if isinstance(label, float) and np.isnan(label):
                label = default_value
                pro_labels.append(label)
            else:
                pro_labels.append(label)

        return pro_labels

    def get_y(self, i):
        df, smiles, labels = self.pro_smi()
        pro_labels = self.pro_nan(labels)
        y = torch.LongTensor([pro_labels[i]]) if self.task_type == 'classification' else torch.FloatTensor([labels[i]])
        return y

    def process(self):

        #opt = parse_args()
        maxBlocks = int(opt.maxBlocks)
        maxSR = int(opt.maxSR)
        minFragAtoms = int(opt.minFragAtoms)
        d, smiles, l = self.pro_smi()
        data_list = []
        mols = []

        for i, smi in enumerate(tqdm(smiles)):
            mol = Chem.MolFromSmiles(smi)
            mols.append(mol)

        for i, smi in enumerate(tqdm(smiles)):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                macfrag_list = [Chem.MolFromSmiles(macfrag_smiles)
                                for macfrag_smiles in
                                (MacFrag(mol, maxBlocks=maxBlocks,
                                         maxSR=maxSR, asMols=False, minFragAtoms=minFragAtoms))]
                for frag_macfrag in macfrag_list:
                    data = self.mol2graph(smi, mol, frag_macfrag, i)
                    data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def mol2graph(self, smi, mol, frag, i):
        if mol is None: return None
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        frag_node_attr = atom_attr(frag)
        fra_edge_index, fra_edge_attr, cluster_index = frag_info(frag)

        data = MolData(
            x=torch.FloatTensor(node_attr),
            edge_index=torch.LongTensor(edge_index).t(),
            edge_attr=torch.FloatTensor(edge_attr),
            frag_x=torch.FloatTensor(frag_node_attr),
            frag_edge_index=torch.LongTensor(fra_edge_index).t(),
            frag_edge_attr=torch.FloatTensor(fra_edge_attr),
            cluster_index=torch.LongTensor(cluster_index),
            y=self.get_y(i),
            smiles=smi,
        )

        return data


def get_target_list():
    if _C.DATA.DATASET == 'bbbp':
        target_list = ["p_np"]

    elif _C.DATA.DATASET == 'tox21':
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif _C.DATA.DATASET == 'clintox':
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif _C.DATA.DATASET == 'hiv':
        target_list = ["HIV_active"]

    elif _C.DATA.DATASET == 'bace':
        target_list = ["Class"]

    elif _C.DATA.DATASET == 'toxcast':
        target_list = ['ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive',
                       'APR_HepG2_CellCycleArrest_24h_dn', 'APR_HepG2_CellCycleArrest_72h_dn',
                       'APR_HepG2_CellLoss_24h_dn', 'APR_HepG2_CellLoss_72h_dn']

    elif _C.DATA.DATASET == 'sider':
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", "Eye disorders",
            "Investigations",
            "Musculoskeletal and connective tissue disorders", "Gastrointestinal disorders", "Social circumstances",
            "Immune system disorders", "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions",
            "Endocrine disorders", "Surgical and medical procedures", "Vascular disorders",
            "Blood and lymphatic system disorders",
            "Skin and subcutaneous tissue disorders", "Congenital, familial and genetic disorders",
            "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", "Renal and urinary disorders",
            "Pregnancy, puerperium and perinatal conditions", "Ear and labyrinth disorders", "Cardiac disorders",
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]

    elif _C.DATA.DATASET == 'MUV':
        target_list = [
            "MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-692", "MUV-712", "MUV-713",
            "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"
        ]

    elif _C.DATA.DATASET == 'FreeSolv':
        target_list = ["freesolv"]

    elif _C.DATA.DATASET == 'ESOL':
        target_list = ["ESOL predicted log solubility in mols per litre"]

    elif _C.DATA.DATASET == 'Lipo':
        target_list = ["lipo"]

    elif _C.DATA.DATASET == 'qm7':
        target_list = ["u0_atom"]

    elif _C.DATA.DATASET == 'qm8':
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0",
            "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"
        ]

    else:
        raise ValueError('Unspecified dataset!')

    return target_list


def build_dataset(cfg, logger):
    cfg.defrost()
    task_name = get_task_names(os.path.join(cfg.DATA.DATA_PATH, 'raw/{}.csv'.format(cfg.DATA.DATASET)))
    if cfg.DATA.TASK_TYPE == 'classification':
        out_dim = 2 * len(task_name)
    elif cfg.DATA.TASK_TYPE == 'regression':
        out_dim = len(task_name)
    else:
        raise Exception('Unknown task type')
    opts = ['DATA.TASK_NAME', task_name, 'MODEL.OUT_DIM', out_dim]
    cfg.defrost()
    cfg.merge_from_list(opts)
    cfg.freeze()
    target_list = get_target_list()

    for target in target_list:
        print('It is the', target, 'target')

        train_dataset, valid_dataset, test_dataset, weights = load_dataset_random(cfg.DATA.DATA_PATH,
                                                                                  cfg.DATA.DATASET,
                                                                                  cfg.SEED,
                                                                                  cfg.DATA.TASK_TYPE,
                                                                                  cfg.DATA.TASK_NAME,
                                                                                  logger)

        return train_dataset, valid_dataset, test_dataset, weights


def load_dataset_random(path, dataset, seed, task_type, tasks=None, logger=None):
    save_path = path + 'processed/train_valid_test_{}_seed_{}.ckpt'.format(dataset, seed)

    pro_dataset = MolDataset(root=path, dataset=dataset, task_type=task_type, tasks=tasks, logger=logger)
    pro_dataset.process()

    random = Random(seed)
    indices = list(range(len(pro_dataset)))
    random.seed(seed)
    random.shuffle(indices)

    train_size = int(0.8 * len(pro_dataset))
    val_size = int(0.1 * len(pro_dataset))

    trn_id, val_id, test_id = indices[:train_size], \
                              indices[train_size:(train_size + val_size)], \
                              indices[(train_size + val_size):]

    trn, val, test = pro_dataset[torch.LongTensor(trn_id)], \
                     pro_dataset[torch.LongTensor(val_id)], \
                     pro_dataset[torch.LongTensor(test_id)]

    assert task_type == 'classification' or 'regression'
    if task_type == 'classification':
        weights = []
        pos_count, neg_count = 0, 0

        for i in range(1):
            for data in pro_dataset:

                if data.y == 0 or data.y == 1:
                    if data.y == 1:
                        pos_count += 1
                    else:
                        neg_count += 1

            weight_pos = (neg_count + pos_count) / pos_count
            weight_neg = (neg_count + pos_count) / neg_count
        weights = torch.tensor([weight_neg, weight_pos])
    else:
        weights = None

    torch.save([trn, val, test], save_path)
    return trn, val, test, weights


'''def build_loader(cfg, logger):
    train_dataset, valid_dataset, test_dataset, weights = build_dataset(cfg, logger)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.DATA.BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.DATA.BATCH_SIZE)

    return train_dataloader, valid_dataloader, test_dataloader, weights'''


import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
import matplotlib.pyplot as plt


def compute_info_score(frag):
    # e.g., just atom count or more advanced like Morgan fingerprint entropy
    return frag.GetNumAtoms()

def assign_levels(fragments, n_levels=3):
    scores = [compute_info_score(frag) for frag in fragments]
    sorted_idxs = sorted(range(len(fragments)), key=lambda i: -scores[i])
    levels = [0] * len(fragments)
    for i, idx in enumerate(sorted_idxs):
        levels[idx] = min(i * n_levels // len(fragments), n_levels - 1)
    return levels

def build_hierarchical_macfrag(smiles, maxBlocks=4, maxSR=8, minFragAtoms=1, n_levels=3):
    mol = Chem.MolFromSmiles(smiles)
    mol = mol_with_atom_index(mol)
    bonds = list(searchBonds(mol, maxSR=maxSR))
    broken = BreakBRICSBonds(mol, bonds=bonds)
    frags = [f for f in Chem.GetMolFrags(broken, asMols=True)
             if f.GetNumAtoms() >= minFragAtoms]

    # Assign levels
    levels = assign_levels(frags, n_levels=n_levels)

    # Build graph
    G = nx.Graph()
    for i, (frag, level) in enumerate(zip(frags, levels)):
        G.add_node(i, smiles=Chem.MolToSmiles(frag), level=level, size=frag.GetNumAtoms())

    # Connect fragments based on original bonds
    atom_map_to_frag = {}
    for i, frag in enumerate(frags):
        for atom in frag.GetAtoms():
            atom_map_to_frag[atom.GetAtomMapNum()] = i

    for (i, j), _ in bonds:
        if i in atom_map_to_frag and j in atom_map_to_frag:
            a, b = atom_map_to_frag[i], atom_map_to_frag[j]
            if a != b:
                G.add_edge(a, b)

    # Optionally add a supernode for full molecule
    G.add_node("super", level=0)
    for i in range(len(frags)):
        G.add_edge("super", i)

    return G

def draw_hierarchy(G, out_file="macfrag_hierarchy.jpg"):
    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]["level"] if n != "super" else -1 for n in G.nodes()]
    cmap = plt.cm.viridis

    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True,
            node_color=node_colors,
            node_size=800,
            cmap=cmap)
    nx.draw_networkx_labels(G, pos, labels={n: f"{n}" for n in G.nodes()})
    plt.title("MacFrag Fragment Hierarchy")
    plt.savefig(out_file, dpi=200)
    plt.close()
    
from rdkit.Chem import Draw
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import networkx as nx

def draw_hierarchy_with_mols(G, out_file="macfrag_hierarchy_with_mols.jpg"):
    """
    Visualize hierarchical fragment graph with actual fragment images at each node.
    """
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw edges only
    nx.draw_networkx_edges(G, pos, ax=ax, width=2)

    # Draw each fragment as an image
    for node, attr in G.nodes(data=True):
        if node == "super":
            ax.scatter(*pos[node], s=800, c="black", label="Supernode", zorder=3)
            ax.text(pos[node][0], pos[node][1] + 0.05, "super", color="white", ha='center', va='center', fontsize=10)
        else:
            smiles = attr.get("smiles")
            mol = Chem.MolFromSmiles(smiles)
            img = Draw.MolToImage(mol, size=(100, 100))
            imagebox = OffsetImage(img, zoom=0.5)
            ab = AnnotationBbox(imagebox, pos[node], frameon=False)
            ax.add_artist(ab)

    ax.set_title("MacFrag Fragment Hierarchy with Molecule Nodes", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved fragment graph with molecules to {out_file}")

if __name__ == "__main__":
    smiles = "[Cl].CC(C)NCC(O)COc1cccc2ccccc12"
    G = build_hierarchical_macfrag(smiles, maxSR=8, minFragAtoms=1, n_levels=3)
    draw_hierarchy_with_mols(G, out_file="macfrag_hierarchy_with_mols.jpg")