import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple
import numpy as np

from chemprop.train.run_training import run_training
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.torchlight import initialize_exp


def run_stat(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)
    info(f'Run scaffold {args.runs}')
    args.save_dir = os.path.join(save_dir, f'run_{args.seed}')
    makedirs(args.save_dir)
    model_scores = run_training(args, args.pretrain, logger)
    info(f'{args.runs}-times runs')
    info(f'Scaffold {args.runs} ==> test {args.metric} = {model_scores:.6f}')

    return model_scores


if __name__ == '__main__':
    args = parse_train_args()
    args.data_path = "/home/tara/group-graph-testing/ContrastiveFGG/data/bace.csv"
    args.dataset = "bace"
    args.root_path = "/home/tara/group-graph-testing/ContrastiveFGG/data"
    args.metric = "auc"
    args.dataset_type = "classification"
    args.split_type = "scaffold_balanced"# scaffold_balanced
    args.runs = 2
    args.encoder = False
    args.exp_name = "finetune"
    args.exp_id = "esol" 
    args.checkpoint_path = "/home/tara/group-graph-testing/ContrastiveFGG/dumped/0702-finetune/pretrain/model/func_MoleculeModel_0702_1404_270th_epoch.pkl"
    args.gpu = 1
    args.epochs = 100
    args.pretrain = False
    args.early_stop = True
    args.atom_messages =  True
    args.increase_parm = 1 
    args.init_lr = 1e-4
    args.max_lr = 1e-3
    args.final_lr = 1e-4
    args.warmup_epochs =2
    args.gnn_type = 'gin'
    args.graph_pooling = 'mean'
    args.fg_dim = 73
    args.atom_dim = 101
    args.bond_dim=11
    args.fg_edge_dim=101
    args.latent_dim = 160
    args.hidden_size = 160
    args.add_reactive = False
    args.add_step = 'concat_mol_frag_attention'
    args.step = ''
    args.early_stop = False
    args.patience = 30
    args.last_early_stop = 0
    args.batch_size = 256
    args.ffn_num_layers = 2
    args.encoder_name = "FuncGNN" 
    args.dropout = 0.1
    args.l2_norm = 0
    args.depth = 3
    args.encoder_head = 4
    args.num_attention = 2
    args.num_layers = 3
    modify_train_args(args)
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    model_scores = run_stat(args, logger)
    print(f'Scaffold-{args.runs} Results: {model_scores:.5f}')
