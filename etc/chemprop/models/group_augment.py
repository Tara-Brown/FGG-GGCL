import torch
import torch.nn as nn
from .data_loader import SmilesDataset, get_vocab_descriptors,  get_vocab_macc, get_vocab_data, FragmentDataset
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops
from torch.nn import functional as F
import os
import numpy as np
import pandas as pd
from argparse import Namespace
from .graph_embed import GNN_mol
from rdkit import Chem
from .prepare_group_graph import get_vocab, sanitize
from torch_geometric.data import Batch

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

class HierGnnEncoder(nn.Module):

    def __init__(self, args):
        super(HierGnnEncoder, self).__init__()
        self.vocab = args.vocab_size
        self.num_tasks = args.num_tasks
        self.device = args.device
        self.num_layers = args.num_layers
        self.emb_dim = args.hidden_size
        self.drop_ratio = args.dropout
        self.gnn_type = args.gnn_type
        self.graph_gnn = GNN_mol(self.num_layers, self.emb_dim, self.gnn_type, drop_ratio=0)
        self.graph_pooling = args.graph_pooling

        self.fc_gat = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
        )
        self.fpn = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        self.attr_embed = torch.nn.Sequential(
            torch.nn.Linear(270, self.emb_dim),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.Dropout(0.1),
        )

        self.node_embed = torch.nn.Sequential(
            torch.nn.Linear(args.vocab_size[1], self.emb_dim),
            nn.ReLU(),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.Dropout(0),
        )

        self.mol_embed = torch.nn.Sequential(
            torch.nn.Linear(2 * self.emb_dim, self.emb_dim),
            nn.ReLU(),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.Dropout(self.drop_ratio),
        )

        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool

        self.graph_pred_linear = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.Linear(self.emb_dim, args.num_tasks)
        )
        self.fusion_proj = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.empty_fusion_proj = nn.Linear(self.emb_dim, 2 * self.emb_dim)

    def forward(self, vocab_datas, batch_data):
        # Move vocab_datas to device
        vocab_datas = vocab_datas.to(self.device)  # added

        x, edge_index, edge_attr, inter_tensor, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, \
                                                        batch_data.inter_tensor, batch_data.batch
        
        # Move inputs to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        inter_tensor = inter_tensor.to(self.device)
        batch = batch.to(self.device)

        # vocab_datas is now on device, so indexing works properly
        x_input_1 = vocab_datas.index_select(index=x.squeeze(), dim=0)

        x_tensor = self.node_embed(x_input_1)

        edge_attr_0 = torch.cat([inter_tensor[:, 0, :], edge_attr[:, 0:2]], dim=1).to(torch.float)
        edge_attr_1 = torch.cat([inter_tensor[:, 1, :], edge_attr[:, 2:4]], dim=1).to(torch.float)
        edge_attr = torch.cat([edge_attr_0, edge_attr_1], dim=1)
        edge_attr_trans = torch.cat([edge_attr_1, edge_attr_0], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr_trans], dim=0)  # add reverse edge info

        edge_index_trans = torch.stack([edge_index[1, :], edge_index[0, :]], dim=0)
        edge_index = torch.cat([edge_index, edge_index_trans], dim=1)  # add reverse edges
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        edge_index = edge_index.to(self.device)  # added

        self_loop_attr = torch.zeros(x.size(0), 270, device=self.device, dtype=edge_attr.dtype)  # create directly on device
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_attr = self.attr_embed(edge_attr)

        node_representation = self.graph_gnn(x_tensor, edge_index, edge_attr)
        gat_out = self.pool(node_representation, batch)

        mol_pred = self.graph_pred_linear(gat_out)

        return mol_pred, gat_out


class Predictor(nn.Module):
    def __init__(self, args: Namespace, vocab_datas):
        super().__init__()
        self.args = args
        self.vocab_datas = vocab_datas.to(self.args.device)  # move once on init
        self.encoder = HierGnnEncoder(args=args)

    def forward(self, batch_data):
        batch_data = pd.DataFrame(batch_data)
        g_list, vocab_df = get_vocab(batch_data, ncpu=8)
        vocab_list = vocab_df['smiles'].tolist()
        dataset = SmilesDataset(vocab_list, g_list, self.args.path, data_type='frag_datas')
        batched_data = Batch.from_data_list(dataset).to(self.args.device)  # move batch to device here
        
        _, emb1 = self.encoder(self.vocab_datas, batched_data)

        return emb1
