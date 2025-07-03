import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple
import torch
import pandas as pd
import numpy as np

from chemprop.train.run_training import pre_training
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.torchlight import initialize_exp

def pretrain(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    pre_training(args, logger)

if __name__ == '__main__':
    args = parse_train_args()
    args.data_path = "/home/tara/group-graph-testing/group-graph/dataset/class_task/clintox/clintox.csv"
    args.path = "/home/tara/group-graph-testing/group-graph/dataset/class_task/clintox"
    args.gpu = 1
    args.runs = 1
    args.start_epochs = 0
    args.end_epochs = 1000
    args.batch_size = 32
    args.exp_id = 'pretrain'
    args.device = 'cuda:1'
    args.dataset_type = 'classification'
    args.logger = True
    args.ffn_drop_out = 0.1
    args.dropout = 0
    args.pretrain = True
    args.encoder_drop_out = 0.1
    args.encoder_head = 4
    args.gru_head = 6
    args.add_step=''
    args.step =''
    args.num_tasks = 2
    args.gnn_type = 'gin'
    args.graph_pooling = 'mean'
    args.fg_dim = 73
    args.atom_dim = 101
    args.bond_dim=11
    args.fg_edge_dim=101
    args.latent_dim = 300
    args.early_stop=True
    args.hidden_size = 300
    args.num_layers = 3
    args.save_BRICS_f = False
    args.atom_messages = True
    modify_train_args(args)
    args.checkpoint_paths = None
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    pretrain(args, logger)

'''
if task == "tox21":
            num_tasks = 12
        elif task == "pcba":
            num_tasks = 128
        elif task == "muv":
            num_tasks = 17
        elif task == "toxcast":
            num_tasks = 617
        elif task == "sider":
            num_tasks = 27
        elif task == "clintox":
            num_tasks = 2
        else:
            num_tasks = 1
'''