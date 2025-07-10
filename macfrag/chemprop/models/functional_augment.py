import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Batch
dtype = torch.float32
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import GATConv, GCNConv
import numpy as np
from argparse import Namespace
from chemprop.models.loader import MoleculeNetDataset
from chemprop.new_features.chem import *
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Batch

'''
class FragmentGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super(FragmentGNN, self).__init__()
        self.convs = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        x = self.pool(x, batch)
        return x
'''

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
        for i, conv in enumerate(self.convs):
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
        # fg_x and fg_embeds: [num_fg, embed_dim]
        q = self.query(fg_x).unsqueeze(1)         # [num_fg, 1, d]
        k = self.key(torch.stack([fg_x, fg_embeds], dim=1))  # [num_fg, 2, d]
        v = self.value(torch.stack([fg_x, fg_embeds], dim=1))  # [num_fg, 2, d]

        scores = torch.matmul(q, k.transpose(-1, -2)) / (fg_x.size(-1) ** 0.5)  # [num_fg, 1, 2]
        attn_weights = self.softmax(scores)  # [num_fg, 1, 2]

        fused = torch.matmul(attn_weights, v).squeeze(1)  # [num_fg, d]
        return fused

class SerGINE(nn.Module):
    def __init__(self, num_atom_layers=3, num_fg_layers=2, latent_dim=128,
                 atom_dim=101, fg_dim=73, bond_dim=11, fg_edge_dim=101, fragment_gnn=None,
                 atom2fg_reduce='mean', pool='mean', dropout=0, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_atom_layers = num_atom_layers
        self.num_fg_layers = num_fg_layers
        self.atom2fg_reduce = atom2fg_reduce
        self.fragment_gnn = fragment_gnn

        # embedding
        self.atom_embedding = nn.Linear(atom_dim, latent_dim)
        self.fg_embedding = nn.Linear(fg_dim, latent_dim)
        self.bond_embedding = nn.ModuleList(
            [nn.Linear(bond_dim, latent_dim) for _ in range(num_atom_layers)]
        )
        self.fg_edge_embedding = nn.ModuleList(
            [nn.Linear(fg_edge_dim, latent_dim) for _ in range(num_fg_layers)]
        )

        # gnn
        self.atom_gin = nn.ModuleList(
        [GINEConv(
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim*2),
                nn.BatchNorm1d(latent_dim*2),
                nn.ReLU(),
                nn.Linear(latent_dim*2, latent_dim)
            ),
            edge_dim = latent_dim
        ) for _ in range(num_atom_layers)]
    )

        self.atom_bn = nn.ModuleList(
            [nn.BatchNorm1d(latent_dim) for _ in range(num_atom_layers)]
        )
        self.fg_gin = nn.ModuleList(
            [GINEConv(
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim*2) , nn.ReLU(), nn.Linear(latent_dim*2, latent_dim)
                )
            ) for _ in range(num_fg_layers)]
        )
        self.fg_bn = nn.ModuleList(
            [nn.BatchNorm1d(latent_dim) for _ in range(num_fg_layers)]
        )
        self.atom2fg_lin = nn.Linear(latent_dim, latent_dim)
        # pooling
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'sum':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling!")
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

        # flatten list of fragment subgraphs (if still nested)
        flat_subgraphs = [fg for mol_subgraphs in subgraphs for fg in mol_subgraphs]

        # batch the fragment-level subgraphs
        batched_fgs = Batch.from_data_list(flat_subgraphs)

        # pass to FragmentGNN
        fg_embeds = self.fragment_gnn(batched_fgs)
        # one-hot to vec
        #atom_x = self.atom_embedding(atom_x)
        fg_x = self.fg_embedding(fg_x) 
        fg_x = fg_x +fg_embeds
        '''
        # atom-level gnn
        for i in range(self.num_atom_layers):
            bond_embedding = self.bond_embedding[i](atom_edge_attr)
            atom_x = self.atom_gin[i](atom_x, atom_edge_index, bond_embedding)
            atom_x = self.atom_bn[i](atom_x)
            if i != self.num_atom_layers-1:
                atom_x = self.relu(atom_x)
            atom_x = self.dropout(atom_x)

        # atom-level to FG-level
        atom2fg_x = scatter(atom_x[atom_idx], index=fg_idx, dim=0, dim_size=fg_x.size(0), reduce=self.atom2fg_reduce)
        atom2fg_x = self.atom2fg_lin(atom2fg_x)
        
        fg_x = fg_x + atom2fg_x
        '''

        # fg-level gnn
        for i in range(self.num_fg_layers):
            fg_x = self.fg_gin[i](fg_x, fg_edge_index, self.fg_edge_embedding[i](fg_edge_attr))
            #fg_x = self.fg_bn[i](fg_x)
            if i != self.num_fg_layers-1:
                fg_x = self.relu(fg_x)
            fg_x = self.dropout(fg_x)

        fg_graph = self.att_pool(fg_x, fg_batch)

        return fg_graph

class FragmentGNNEncoder(nn.Module):
    def __init__(self, args: Namespace, num_tasks=1):
        super().__init__()
        self.args = args
        self.emb_dim = args.latent_dim
        # Fragment encoder
        self.fragment_gnn = FragmentGNN(input_dim=101, hidden_per_head =args.latent_dim // args.encoder_head, num_layers=args.num_layers, heads=args.encoder_head)
        
        # Atom + FG GNN
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
        #Turns data into a MoleculeNetDataset
        data = pd.DataFrame(data)
        new_data = process_data(data)
        new_data = MoleculeNetDataset(new_data) 
        batched_data = Batch.from_data_list(new_data, follow_batch=['fg_x']).to(self.args.device)
        graph_embed = self.encoder(batched_data)
        return graph_embed
    
def process_data(df):
    dataset = []
    err_cnt = 0
    for i in range(len(df)):
        smiles_col = 0
        # value_list = np.array(df.iloc[:,smiles_col+1:]).tolist()
        ids = df.iloc[i, smiles_col]
        y = df.iloc[i, smiles_col + 1:].values

        mol = Chem.MolFromSmiles(ids)
        w = 1
        if mol is None:
            print(f"'{ids}' cannot be convert to graph")
            err_cnt += 1
            continue
        atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list, fragment_subgraphs = mol_to_graphs(mol)
        if len(fg_features) == 0:
            print("AHHH")
        # Fallback: treat the whole molecule as one "dummy FG"
            fg_features = [[0] * 73]
            fg_edge_list = []
            fg_edge_features = []
            atom2fg_list = [0] * mol.GetNumAtoms()
            print(fragment_subgraphs)
        
        dataset.append([atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list, fragment_subgraphs, y, w])
    #print(f"{err_cnt} data can't be convert to graph")
    return dataset

