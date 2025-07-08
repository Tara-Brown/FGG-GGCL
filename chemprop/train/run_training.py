from argparse import Namespace
import csv
from logging import Logger
import os
from typing import List
from chemprop.models.data_loader import SmilesDataset, get_vocab_descriptors,  get_vocab_macc, get_vocab_data, FragmentDataset
from chemprop.models.loader import MoleculeNetDataset
import numpy as np
import torch
import pandas as pd
from chemprop.new_features.chem import *
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data,load_data
from chemprop.models import build_model, build_pretrain_model,add_FUNC_prompt
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint,Early_stop
from chemprop.data import MoleculeDataset
from tqdm import tqdm
from chemprop.models import ContrastiveLoss
from chemprop.torchlight import initialize_exp, snapshot
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import pickle
from rdkit import Chem
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import sigmoid
from sklearn.metrics import roc_auc_score

def run_training(args: Namespace, pretrain: bool, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")

    # Print args
# =============================================================================
#     debug(pformat(vars(args)))
# =============================================================================

    # Get data
    info('Loading data')
    # args.vocab = Vocab(args)
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    info(f'Number of tasks = {args.num_tasks}')
    
    
    # Split data
    debug(f'Load data from {args.exp_id} for Scaffold-{args.runs}')
    if 0<=args.runs<3:
        train_data, val_data, test_data = load_data(data,args,logger)
    else:
        print('='*100)
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))
    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        # Load/build model
        if args.checkpoint_path is not None:
            debug(f'Loading model from {args.checkpoint_path}')
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model = build_model(args, encoder_name=args.encoder_name)
            model_state_dict = model.encoder.state_dict() if args.encoder else model.state_dict()
            pretrained_state_dict = {}
            for param_name in checkpoint.keys():
                if param_name not in model_state_dict:
                    print(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
                elif model_state_dict[param_name].shape != checkpoint[param_name].shape:
                    print(f'Pretrained parameter "{param_name}" '
                    f'of shape {checkpoint[param_name].shape} does not match corresponding '
                    f'model parameter of shape {model_state_dict[param_name].shape}.')
                else:
                    pretrained_state_dict[param_name] = checkpoint[param_name]
            model_state_dict.update(pretrained_state_dict)
            if args.encoder:
                model.encoder.load_state_dict(model_state_dict)
            else:
                model.load_state_dict(model_state_dict)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args, encoder_name=args.encoder_name)
        
        if args.step == 'func_prompt':
            add_FUNC_prompt(model, args)
        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Early_stop
        early_stop = False

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        if args.early_stop:
            stopper = Early_stop(patience=args.patience,minimize_score=args.minimize_score)
        for epoch in range(args.epochs):
            avg_loss = train(
                model=model,
                pretrain=pretrain,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            
            val_scores = evaluate(
                model=model,
                pretrain=pretrain,
                data=val_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            test_preds = predict(
                model=model,
                pretrain=pretrain,
                data=test_data,
                batch_size=args.batch_size,
                scaler=scaler
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            # Average test score
            avg_test_score = np.nanmean(test_scores)
            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args) 
        
            if args.early_stop and epoch>= args.last_early_stop:
                early_stop = stopper.step(avg_val_score)
                info(f'Epoch{epoch+1}/{args.epochs},train loss:{avg_loss:.4f},valid_{args.metric} = {avg_val_score:.6f},test_{args.metric} = {avg_test_score:.6},\
                    best_epoch = {best_epoch+1},patience = {stopper.counter}')
            else:
                info(f'Epoch{epoch+1}/{args.epochs},train loss:{avg_loss:.4f},valid_{args.metric} = {avg_val_score:.6f},test_{args.metric} = {avg_test_score:.6},\
                    best_epoch = {best_epoch+1}')
            if args.early_stop and early_stop:
                break
        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'),current_args=args, cuda=args.cuda, logger=logger)
        
        test_preds = predict(
            model=model,
            pretrain=pretrain,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )
        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)
        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    # Average ensemble score
    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')

    # Individual ensemble scores
    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')

    return avg_ensemble_test_score

def pre_training(args: Namespace, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        args.device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
        torch.cuda.set_device(args.gpu)

    # Print args
# =============================================================================
#     debug(pformat(vars(args)))
# =============================================================================

    # Get data
    debug('Loading data')
    data = get_data(path=args.data_path, args=args, logger=logger)


    args.data_size = len(data)
    
    debug(f'Total size = {len(data)}')

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model1 = load_checkpoint(args.checkpoint_paths[0], args=args, logger=logger,encoder_name='GroupGNN')
            model2 = load_checkpoint(args.checkpoint_paths[1], args=args, logger=logger,encoder_name='FuncGNN')
            model3 = load_checkpoint(args.checkpoint_paths[2], args=args, logger=logger,encoder_name='MacFrag')
        else:

            debug(f'Building model {model_idx}')
            # model = build_model(args)
            model1 = build_pretrain_model(args, encoder_name='GroupGNN') #One of these will be used for group graph (im thinking this one)
            model2 = build_pretrain_model(args, encoder_name='FuncGNN') # This one will be used for the functional group graph
            model3 = build_pretrain_model(args, encoder_name='MacFrag') # This one will be used for the functional group graph
            model4 = build_pretrain_model(args, encoder_name='CMPNN') #One of these will be used for group graph (im thinking this one)
        

        debug(model1)
        debug(f'Number of M1 parameters = {param_count(model1):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model1 = model1.cuda()

        debug(model2)
        debug(f'Number of M2 parameters = {param_count(model2):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model2 = model2.cuda()

        #logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{args.save_dir}/model'
        
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        args.device = device
        cl_criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args).cuda()
        fin_criterion = get_loss_func(args)
        score = get_metric_func(metric=args.metric)
        optimizer = Adam([{"params": model3.parameters()},{"params": model4.parameters()}], lr=3e-5)
        scheduler = ExponentialLR(optimizer, 0.99, -1)
        step_per_schedule = 500
        global_step = 0
        mol = MoleculeDataset(data)

        class SmilesTargetDataset(Dataset):
            def __init__(self, smiles_list, targets_list):
                assert len(smiles_list) == len(targets_list)
                self.smiles = smiles_list
                self.targets = torch.tensor([
            [np.nan if t is None else float(t) for t in row]
            for row in targets_list
            ] )
            def __len__(self):
                return len(self.smiles)

            def __getitem__(self, idx):
                return self.smiles[idx], self.targets[idx]

        # Prepare dataset
        smiles, targets = mol.smiles(), mol.targets()

        # Create dataset instance
        full_dataset = SmilesTargetDataset(smiles, targets)

        # Split indices
        train_size = int(0.90 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

        # Run training

        best_loss = float('-inf')
        patience = args.patience if hasattr(args, 'patience') else 10  # or hardcode
        epochs_no_improve = 0
        early_stop = False
        best_epoch = 0
        fusion_proj = nn.Linear(2 * args.latent_dim, args.num_tasks).to(device)
        for epoch in range(args.start_epochs,args.end_epochs):
            debug(f'Epoch {epoch}')
            if early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            debug = logger.debug if logger is not None else print
            total_loss = 0
            
            with tqdm(total=len(train_loader)) as t:
                for batch, target in train_loader:
                    model3.train()
                    model4.train()
                    # Run model
                    emb1 = model3(batch).to(device) # Group Graph
                    emb2 = model4(step, False, batch, None) # Functional Group graph + y (for now)
                    print(f"Aug1 emb mean: {emb1.mean():.4f}, std: {emb1.std():.4f}")
                    print(f"Aug2 emb mean: {emb2.mean():.4f}, std: {emb2.std():.4f}")
                    labels = target.to(device).view(-1, args.num_tasks)
                    labels_masked = labels.clone()
                    mask = ~torch.isnan(labels)
                    labels_masked[~mask] = 0
                    output = fusion_proj(torch.cat([emb1, emb2], dim=-1))
                    loss_raw = fin_criterion(output, labels_masked)  # shape (batch_size, num_tasks)
                    loss_masked = loss_raw * mask.float()        # zero out loss for missing labels
                    auc_loss = loss_masked.sum() / mask.sum()        # average loss over valid labels only
                    loss = 0.3 * cl_criterion(emb1, emb2) + .7*auc_loss
                    y_true_np = labels.cpu().numpy()
                    y_score_np = torch.sigmoid(output).detach().cpu().numpy()
                    auc = masked_roc_auc_score(y_true_np, y_score_np)
                    print(f"Train AUC: {auc:.4f}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    global_step += 1
                    t.set_description('Epoch[{}/{}]'.format(epoch+1, args.end_epochs))
                    t.set_postfix(train_loss=loss.item())
                    t.update()
                    total_loss+=loss.item() 
                    if global_step % step_per_schedule == 0:
                        scheduler.step()
            # save model   
            model3.eval()
            model4.eval()
            total_test_loss = 0
            y_true_all = []
            y_score_all = []
            best_auc = float('-inf')

            for test_batch, test_target in test_loader:
                with torch.no_grad():
                    emb1 = model3(test_batch)
                    emb2 = model4(step, False, test_batch, None)
                    labels = test_target.to(device).view(-1, args.num_tasks)
                    mask = ~torch.isnan(labels)
                    labels_masked = labels.clone().masked_fill(~mask, 0)

                    output = fusion_proj(torch.cat([emb1, emb2], dim=-1))
                    loss_raw = fin_criterion(output, labels_masked)
                    loss_masked = loss_raw * mask.float()
                    test_loss = 0.3 * cl_criterion(emb1, emb2) + 0.7 * loss_masked.sum() / mask.sum()

                    total_test_loss += test_loss.item()
                    y_true_all.append(labels.cpu().numpy())
                    y_score_all.append(torch.sigmoid(output).cpu().numpy())

            avg_test_loss = total_test_loss / len(test_loader)
            y_true_np = np.concatenate(y_true_all, axis=0)
            y_score_np = np.concatenate(y_score_all, axis=0)
            test_auc = masked_roc_auc_score(y_true_np, y_score_np)

            snapshot(model3, epoch, dump_folder, 'MacFrag')
            snapshot(model4, epoch, dump_folder, 'CMPNN')

            logger.info(f"Epoch {epoch+1}/{args.end_epochs} | Step {global_step} | "
                        f" Avg Test Loss: {avg_test_loss:.4f} | "
                        f"Test ROC-AUC: {test_auc:.4f}")

            # ---------- Early Stopping ----------
            if test_auc > best_auc:
                best_auc = test_auc
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model3.state_dict(), f'{dump_folder}/best_model_macfrag.pt')
                torch.save(model4.state_dict(), f'{dump_folder}/best_model_cmpnn.pt')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best ROC-AUC: {best_auc:.4f})")
                    early_stop = True




def masked_roc_auc_score(y_true, y_score):
    # y_true, y_score are numpy arrays, shape (batch_size, num_tasks)
    num_tasks = y_true.shape[1]
    aucs = []
    for i in range(num_tasks):
        mask = ~np.isnan(y_true[:, i])
        if np.sum(mask) == 0:
            aucs.append(float('nan'))
            continue
        try:
            auc = roc_auc_score(y_true[mask, i], y_score[mask, i])
            aucs.append(auc)
        except ValueError as e:
            print(f"[WARN] ROC AUC calculation failed for task {i}: {e}")
            aucs.append(float('nan'))
    # Return average ignoring NaNs:
    return np.nanmean(aucs)

