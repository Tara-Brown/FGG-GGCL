import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool, GlobalAttention
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
import pandas as pd
from argparse import Namespace
from rdkit import Chem
from chemprop.models.loader import MoleculeNetDataset
from chemprop.new_features.chem import *  # includes mol_to_graphs

class FragmentGNN(nn.Module):
    def __init__(self, input_dim=101, hidden_per_head=32, num_layers=3, heads=4):
        super(FragmentGNN, self).__init__()
        self.hidden_dim = hidden_per_head * heads
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(input_dim, hidden_per_head, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(self.hidden_dim, hidden_per_head, heads=heads))

        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.pool(x, batch)
        return x

class FGAttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, fg_x, fg_embeds):
        q = self.query(fg_x).unsqueeze(1)
        k = self.key(torch.stack([fg_x, fg_embeds], dim=1))
        v = self.value(torch.stack([fg_x, fg_embeds], dim=1))
        scores = torch.matmul(q, k.transpose(-1, -2)) / (fg_x.size(-1) ** 0.5)
        attn_weights = self.softmax(scores)
        fused = torch.matmul(attn_weights, v).squeeze(1)
        return fused

class SerGINE(nn.Module):
    def __init__(self, num_atom_layers=3, num_fg_layers=2, latent_dim=128,
                 atom_dim=101, fg_dim=73, bond_dim=11, fg_edge_dim=101, fragment_gnn=None,
                 atom2fg_reduce='mean', pool='mean', dropout=0, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_atom_layers = num_atom_layers
        self.num_fg_layers = num_fg_layers
        self.fragment_gnn = fragment_gnn

        self.atom_embedding = nn.Linear(atom_dim, latent_dim)
        self.fg_embedding = nn.Linear(fg_dim, latent_dim)
        self.bond_embedding = nn.ModuleList([nn.Linear(bond_dim, latent_dim) for _ in range(num_atom_layers)])
        self.fg_edge_embedding = nn.ModuleList([nn.Linear(fg_edge_dim, latent_dim) for _ in range(num_fg_layers)])

        self.atom_gin = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim * 2),
                    nn.BatchNorm1d(latent_dim * 2),
                    nn.ReLU(),
                    nn.Linear(latent_dim * 2, latent_dim)
                ),
                edge_dim=latent_dim
            ) for _ in range(num_atom_layers)
        ])

        self.atom_bn = nn.ModuleList([nn.BatchNorm1d(latent_dim) for _ in range(num_atom_layers)])

        self.fg_gin = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim * 2),
                    nn.ReLU(),
                    nn.Linear(latent_dim * 2, latent_dim)
                )
            ) for _ in range(num_fg_layers)
        ])

        self.fg_bn = nn.ModuleList([nn.BatchNorm1d(latent_dim) for _ in range(num_fg_layers)])
        self.atom2fg_lin = nn.Linear(latent_dim, latent_dim)

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'sum':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling method")

        self.att_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, 1)
            )
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fg_fusion = FGAttentionFusion(embed_dim=latent_dim)

    def forward(self, data):
        atom_x, atom_edge_index, atom_edge_attr, atom_batch = data.x, data.edge_index, data.edge_attr, data.batch
        fg_x, fg_edge_index, fg_edge_attr, fg_batch = data.fg_x, data.fg_edge_index, data.fg_edge_attr, data.fg_x_batch
        subgraphs = data.subgraphs

        flat_subgraphs = [fg for mol_subgraphs in subgraphs for fg in mol_subgraphs]
        batched_fgs = Batch.from_data_list(flat_subgraphs)

        fg_embeds = self.fragment_gnn(batched_fgs)

        fg_x = self.fg_embedding(fg_x)
        fg_x = fg_x + fg_embeds

        for i in range(self.num_fg_layers):
            fg_edge_attr_embed = self.fg_edge_embedding[i](fg_edge_attr)
            fg_x = self.fg_gin[i](fg_x, fg_edge_index, fg_edge_attr_embed)
            if i != self.num_fg_layers - 1:
                fg_x = self.relu(fg_x)
            fg_x = self.dropout(fg_x)

        fg_graph = self.att_pool(fg_x, fg_batch)
        return fg_graph

class FragmentGNNEncoder(nn.Module):
    def __init__(self, args: Namespace, num_tasks=1):
        super().__init__()
        self.args = args
        self.emb_dim = args.latent_dim

        self.fragment_gnn = FragmentGNN(
            input_dim=args.fg_input_dim,
            hidden_per_head=args.latent_dim // args.encoder_head,
            num_layers=args.num_layers,
            heads=args.encoder_head
        )

        self.encoder = SerGINE(
            latent_dim=self.emb_dim,
            atom_dim=args.atom_dim,
            fg_dim=args.fg_dim,
            bond_dim=args.bond_dim,
            fg_edge_dim=args.fg_edge_dim,
            fragment_gnn=self.fragment_gnn,
            pool=args.graph_pooling,
            num_atom_layers=args.num_layers,
            num_fg_layers=args.num_layers,
            dropout=args.dropout
        )

    def forward(self, data):
        df = pd.DataFrame(data)
        new_data = process_data(df)
        pyg_dataset = MoleculeNetDataset(new_data)
        batched_data = Batch.from_data_list(pyg_dataset, follow_batch=["fg_x"]).to(self.args.device)
        return self.encoder(batched_data)

def process_data(df):
    dataset = []
    err_cnt = 0
    for i in range(len(df)):
        smiles_col = 0
        ids = df.iloc[i, smiles_col]
        y = df.iloc[i, smiles_col + 1:].values
        mol = Chem.MolFromSmiles(ids)
        if mol is None:
            print(f"[ERROR] Invalid SMILES: {ids}")
            err_cnt += 1
            continue

        atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list, fragment_subgraphs = mol_to_graphs(mol)
        if len(fg_features) == 0:
            print("[WARNING] No fragments found, using fallback FG")
            fg_features = [[0] * 73]
            fg_edge_list = []
            fg_edge_features = []
            atom2fg_list = [0] * mol.GetNumAtoms()

        dataset.append([atom_features, bond_list, bond_features, fg_features,
                        fg_edge_list, fg_edge_features, atom2fg_list, fragment_subgraphs, y, 1])

    print(f"[INFO] {err_cnt} molecules could not be parsed")
    return dataset


import os
from MacFrag import MacFrag
from dataset import MolDataset, visualize_mol_data, build_frag_graph, draw_frag_graph
from rdkit import Chem

if __name__ == "__main__":
    # === CONFIGURATION ===
    dataset_root = '.'  # adjust if using a different base path
    dataset_name = 'bbbp'  # corresponds to raw/bbbp.csv
    task_type = 'classification'
    maxBlocks = 4
    maxSR = 8
    minFragAtoms = 1

    print(f"\n🔍 Loading dataset: {dataset_name}...")
    dataset = MolDataset(
        root=dataset_root,
        dataset=dataset_name,
        task_type=task_type,
        tasks=None
    )

    # === SAMPLE GRAPH ===
    sample_idx = 6
    data_sample = dataset[sample_idx]
    print(f"\n📦 Loaded sample {sample_idx} from dataset")
    print(f"SMILES: {data_sample.smiles}")
    print(f"Label (y): {data_sample.y.item()}")

    # === VISUALIZE FULL MOLECULE GRAPH ===
    print("\n📊 Visualizing molecular graph...")
    visualize_mol_data(data_sample, out_path='outputs/molecule_graph.jpg')

    # === VISUALIZE MacFrag SPLIT GRAPH ===
    print("\n🔬 Visualizing MacFrag-based fragment graph...")
    mol = Chem.MolFromSmiles(data_sample.smiles)
    G_frag, frags = build_frag_graph(
        data_sample.smiles,
        maxBlocks=maxBlocks,
        maxSR=maxSR,
        minFragAtoms=minFragAtoms
    )
    draw_frag_graph(G_frag, frags, out_file='outputs/macfrag_graph.jpg')

    print("\n✅ All done.")