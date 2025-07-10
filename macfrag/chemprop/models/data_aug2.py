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
from chemprop.models.MacFrag import MacFrag
#from MacFrag import parse_args,
from chemprop.models.utils import get_task_names
from torch_geometric.data import DataLoader
from chemprop.models.config import _C
#from torch.serialization import add_safe_globals
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
class Opt:
    input_file   = 'raw/bbbp.csv'
    output_path  = 'processed/'
    maxBlocks    = 4
    maxSR        = 8
    asMols       = False
    minFragAtoms = 1

opt = Opt()
    
    
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
        # Original features list with bools, ints, floats
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other']
        ) + onehot_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 'other']) + \
            [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
            onehot_encoding_unk(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                    'other'
                ]
            ) + [atom.GetIsAromatic()]

        if not explicit_H:
            results += onehot_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        if use_chirality:
            try:
                results += onehot_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results += [0, 0] + [atom.HasProp('_ChiralityPossible')]

        if pharmaco:
            # These are properties set as strings '1' or '0', convert to int then float
            results += [
                float(int(atom.GetProp('Hbond_donor'))),
                float(int(atom.GetProp('Hbond_acceptor'))),
                float(int(atom.GetProp('Basic'))),
                float(int(atom.GetProp('Acid'))),
                float(int(atom.GetProp('Halogen')))
            ]

        if scaffold:
            results += [float(int(atom.GetProp('Scaffold')))]

        # Explicitly cast all elements to float to avoid mixed types
        results = [float(x) for x in results]

        feat.append(results)

    arr = np.array(feat, dtype=np.float32)
    #print(f"atom_attr output shape: {arr.shape}, dtype: {arr.dtype}")
    return arr

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

    return np.array(index), np.array(feat, dtype=np.float32)


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

    def __init__(self, root, dataset, task_type, tasks, logger=None, transform=None, pre_transform=None, pre_filter=None):

        #print("MolDataset: __init__ start")
        self.tasks = tasks
        self.dataset = dataset
        self.task_type = task_type
        self.logger = logger
        super(MolDataset, self).__init__(root, transform, pre_transform, pre_filter)
        #add_safe_globals([MolData])
        #print("MolDataset: loading processed data from:", self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        #print("MolDataset: loaded data with", len(self.data.x), "nodes")  # may not reflect full data length
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
        smiles = df['smiles'].tolist()
        labels = df['label_1'].tolist()
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
        #print("MolDataset: process start")
        maxBlocks = int(opt.maxBlocks)
        maxSR = int(opt.maxSR)
        minFragAtoms = int(opt.minFragAtoms)
        d, smiles, l = self.pro_smi()
        #print(f"MolDataset: loaded {len(smiles)} smiles")
        data_list = []
        mols = []

        for i, smi in enumerate(tqdm(smiles)):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"MolDataset: invalid mol at index {i}, smiles={smi}")
            mols.append(mol)

        for i, smi in enumerate(tqdm(smiles)):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                macfrag_list = [Chem.MolFromSmiles(macfrag_smiles)
                                for macfrag_smiles in
                                (MacFrag(mol, maxBlocks=maxBlocks,
                                         maxSR=maxSR, asMols=False, minFragAtoms=minFragAtoms))]
                #print(f"MolDataset: mol {i} has {len(macfrag_list)} fragments")
                for frag_macfrag in macfrag_list:
                    if frag_macfrag is None:
                        print(f"MolDataset: warning - fragment is None for mol index {i}")
                    data = self.mol2graph(smi, mol, frag_macfrag, i)
                    #print(f"MolDataset: created data for mol {i}, data keys: {data.keys}")
                    data_list.append(data)

        if self.pre_filter is not None:
            #print(f"MolDataset: applying pre_filter on {len(data_list)} data samples")
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            #print(f"MolDataset: applying pre_transform on {len(data_list)} data samples")
            data_list = [self.pre_transform(data) for data in data_list]

        #print(f"MolDataset: collating {len(data_list)} data samples")
        data, slices = self.collate(data_list)
        #print("MolDataset: saving processed data")
        torch.save((data, slices), self.processed_paths[0])
        #print("MolDataset: process done")


    def mol2graph(self, smi, mol, frag, i):
        if mol is None:
            return None

        # Get node (atom) features for the whole molecule
        node_attr = atom_attr(mol)

        # Bond info for the molecule
        edge_index, edge_attr = bond_attr(mol)
        edge_attr = edge_attr.astype(np.float32)  # ensure float32

        # Node features for fragment
        frag_node_attr = atom_attr(frag)

        # Fragment edges and cluster indices
        fra_edge_index, fra_edge_attr, cluster_index = frag_info(frag)

        # Convert everything to PyTorch tensors explicitly with right dtypes
        data = MolData(
            x=torch.FloatTensor(node_attr),                 # [num_atoms, num_features], float32
            edge_index=torch.LongTensor(edge_index).t(),   # [2, num_edges]
            edge_attr=torch.FloatTensor(edge_attr),         # [num_edges, num_edge_features], float32
            frag_x=torch.FloatTensor(frag_node_attr),
            frag_edge_index=torch.LongTensor(fra_edge_index).t(),
            frag_edge_attr=torch.FloatTensor(fra_edge_attr),
            cluster_index=torch.LongTensor(cluster_index),
            y=self.get_y(i),
            smiles=smi,
        )

        #print(f"mol2graph: created data with x shape {data.x.shape}, edge_index shape {data.edge_index.shape}")
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
        #print('It is the', target, 'target')

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
import networkx as nx
import torch
def visualize_mol_data(data, out_path='molecule.png'):
    """
    Draw the graph and save it to a JPEG file.

    Args:
        data (MolData): your PyG graph
        out_path (str): path to write the .jpg file
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # Build NetworkX graph
    #print("visualize_mol_data: start")
    #print(f"visualize_mol_data: data keys: {data.keys}")
    #print(f"visualize_mol_data: data.x shape: {data.x.shape}, dtype: {data.x.dtype}")
    #print(f"visualize_mol_data: data.edge_index shape: {data.edge_index.shape}")

    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    #print(f"visualize_mol_data: edge_index numpy shape {edge_index.shape}")
    for i, j in edge_index.T:
        G.add_edge(i, j)

    atom_types = ['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','other']
    #print("Type of data.x:", type(data.x))
    #print("Data.x tensor dtype:", data.x.dtype)
    #print("Data.x tensor shape:", data.x.shape)
    #print("Sample elements:", data.x[:5])
    #print("Attempting to convert to numpy array:")
    
    node_feats = data.x.cpu().numpy()
    
    #print("Raw node_feats.dtype:", node_feats.dtype)
    #print("node_feats.dtype.type:", node_feats.dtype.type)
    #print("repr of dtype:", repr(node_feats.dtype))
    #print("str of dtype:", str(node_feats.dtype))
    #print("Node_feats type:", type(node_feats))
    #print("Node_feats dtype (before #print):", node_feats.dtype)
    #print(f"visualize_mol_data: node_feats dtype {node_feats.dtype}, shape {node_feats.shape}")
    labels = []
    for idx, feat in enumerate(node_feats):
        try:
            arr = np.array(feat[:16])
            label_idx = arr.argmax()
            labels.append(atom_types[label_idx])
        except Exception as e:
            #print(f"visualize_mol_data: error at node {idx} feat: {feat} - {e}")
            labels.append('unknown')
    #print("Type of node_feats[0,0]:", type(node_feats[0,0]))
    #print("Value of node_feats[0,0]:", node_feats[0,0])
    #print(f"Node_feats shape: {node_feats.shape}")
    #print(f"Node_feats element type via python type(): {type(node_feats.flat[0])}")
    #print(f"visualize_mol_data: labels: {labels}")

    plt.figure(figsize=(8,8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=300)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels={i:labels[i] for i in G.nodes()}, font_size=12)
    plt.title('Molecular Graph')
    plt.axis('off')

    plt.tight_layout()
    dir_path = os.path.dirname(out_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    plt.savefig(out_path, format='png', dpi=300)
    plt.close()
    #print(f"visualize_mol_data: saved graph to {out_path}")


from torch_geometric.data import DataLoader

# Assuming dataset root and dataset name are set correctly
#print("Main: Loading dataset")
dataset_root = '.'  
dataset_name = 'bbbp'  

dataset = MolDataset(root=dataset_root, dataset=dataset_name, task_type='classification', tasks=None)
#print(f"Main: Dataset loaded with {len(dataset)} samples")

data_sample = dataset[0]
#print(f"Main: Sample 0 loaded with keys: {data_sample.keys}")
visualize_mol_data(data_sample, out_path='molecule_graph.png')


# a helper to turn an RDKit Mol into a NetworkX graph & labels
def mol_to_nx(mol):
    G = nx.Graph()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(i, j)
    # label nodes by element
    labels = {atm.GetIdx(): atm.GetSymbol() for atm in mol.GetAtoms()}
    return G, labels

def visualize_macfrag(
    smiles,
    maxBlocks=4,
    maxSR=8,
    minFragAtoms=1,
    asMols=False,
    out_file="macfrag_splits.pmg"
):
    """
    Split the input SMILES via MacFrag, then draw each fragment in a grid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    # get fragment SMILES strings
    frags = MacFrag(
        mol,
        maxBlocks=maxBlocks,
        maxSR=maxSR,
        asMols=False,
        minFragAtoms=minFragAtoms
    )

    n = len(frags)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 4 * rows))
    for idx, frag_smi in enumerate(frags):
        frag_mol = Chem.MolFromSmiles(frag_smi)
        G, labels = mol_to_nx(frag_mol)

        ax = plt.subplot(rows, cols, idx + 1)
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgreen', node_size=300)
        nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=12)
        ax.set_title(f"Frag {idx+1}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved MacFrag splits to {out_file}")
    
 

import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from chemprop.models.MacFrag import mol_with_atom_index, searchBonds
from rdkit.Chem.BRICS import BreakBRICSBonds

def build_frag_graph(smiles, maxBlocks=4, maxSR=8, minFragAtoms=1):
    # index the molecule so atomMapNum is set
    mol = Chem.MolFromSmiles(smiles)
    mol = mol_with_atom_index(mol)

    # get the full MacFrag bond tuples

    bonds = list(searchBonds(mol, maxSR=maxSR))
    #print("bonds variable:", bonds)
    #print("type(bonds):", type(bonds))

    # break exactly those bonds
    broken = BreakBRICSBonds(mol, bonds=bonds)

    # get the first‐pass fragments
    column = Chem.GetMolFrags(broken, asMols=True)

    # filter out any too‐small fragments and collect their atomMapNum sets
    frag_sets = []
    for frag in column:
        atom_ids = {a.GetAtomMapNum() for a in frag.GetAtoms() if a.GetAtomMapNum() >= 0}
        if len(atom_ids) >= minFragAtoms:
            frag_sets.append((frag, atom_ids))

    # build a NetworkX graph with one node per fragment
    G = nx.Graph()
    G.add_nodes_from(range(len(frag_sets)))

    # for each cut bond, connect the two fragments it joined
    for (i,j),_ in bonds:
        A = next(idx for idx,(_,s) in enumerate(frag_sets) if i in s)
        B = next(idx for idx,(_,s) in enumerate(frag_sets) if j in s)
        if A != B:
            G.add_edge(A, B)

    return G, [f for f,_ in frag_sets]


def draw_frag_graph(G, frags, out_file="macfrag_graph_fixed.png"):
    # label each node with its index and fragment‐SMILES
    labels = {i: f"#{i}\n{Chem.MolToSmiles(frag, canonical=True)}"
              for i,frag in enumerate(frags)}

    plt.figure(figsize=(6,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color="lightcoral", node_size=800)
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title("MacFrag Fragment‐Connectivity")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved fragment graph to {out_file}")


if __name__=="__main__":
    # pick one of your molecules (or load from CSV)
    example = "CC[C@]1(O)C[C@H]2CN(CCc3c([nH]c4ccccc34)[C@@](C2)(C(=O)OC)c5cc6c(cc5OC)N(C=O)[C@H]7[C@](O)([C@H](OC(C)=O)[C@]8(CC)C=CCN9CC[C@]67[C@H]89)C(=O)OC)C1"
    G, frags = build_frag_graph(example, maxBlocks=30, maxSR=8, minFragAtoms=1)
    draw_frag_graph(G, frags, out_file="macfrag_graph_fixed.png")
