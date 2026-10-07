import sys

# OS
import os
import argparse, json, shutil
import gc
from argparse import Namespace
import pickle
from copy import deepcopy
from tqdm import tqdm

# mathematics
import numpy as np
import random, math

# data handling
from typing import List, Tuple, Optional, Sequence
import pandas as pd

# Torch
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# custom
sys.path.append('LBN')

from LBN.model.net import make_model
from LBN.model.train_swag import train_epoch, evaluate
from LBN.misc.loader import TrajectoryDataset, load_dataset, make_folds_kfold, dataset_to_langevin_data
from LBN.misc.utils import save_checkpoint, type_or_none


def main(args):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print("Random seed:", args.seed, "cv-idx:", args.cv_idx)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    # set training parameters based on model type
    args.epochs = args.epochs_drift if args.model_type == 'drift' else args.epochs_diff
    args.lr = args.lr_drift if args.model_type == 'drift' else args.lr_diff

    # data load and processing
    latent_data, data_content = load_dataset(args.data)

    # Only for synthetic data: we specify the sampling interval and total duration to control the difficulty of the learning task.
    if 'synthetic' in args.data:
        args.simulation_time_step = latent_data['dt']

        if args.protocol == 'A':
            print("Protocol A: Fixed sampling interval, varying total duration.")
            latent_data = latent_data[args.total_duration][args.sampling_interval]
        elif args.protocol == 'B':
            print("Protocol B: Fixed total duration & number of transitions, varying sampling intervals.")
            latent_data = latent_data[args.total_transitions][args.sampling_interval]

        args.time_step = args.simulation_time_step * args.sampling_interval
        print(f"Loaded synthetic latent data with sampling interval {args.sampling_interval}, simulation time step {args.simulation_time_step}, "
              f"and effective time step {args.time_step}, shape {latent_data.shape}")

    # convert latent data to training data

    ## For real-world data, we need to specify the id_col, state_cols, time_col, and time_unit in the arguments.
    if data_content == 'unknown':
        if 'vdem_electoral_academ' in args.data:
            args.id_col = 'iso3'
            args.state_cols = ['v2x_polyarchy', 'v2xca_academ']
            args.time_col = 'year'
            args.time_unit = 1.0
            args.include_time = False
        # Please add more elif clauses here for other real-world datasets with unknown data content.
        else:
            raise ValueError(f"Unknown data content for {args.data}. Please specify the id_col, state_cols, time_col, and time_unit in the arguments.")

    langevin_data = dataset_to_langevin_data(data_content, latent_data, args)
    xs, dxs, dts, series_slices, valid_ids, valid_times = (
        langevin_data['xs'],
        langevin_data['dxs'],
        langevin_data['dts'],
        langevin_data['series_slices'],
        langevin_data['valid_ids'],
        langevin_data['valid_times']
    )

    args.dim, args.input_dim = dxs.shape[-1], xs.shape[-1]
    args.n_cond_onehot = args.input_dim - args.dim
    if args.include_time:
        args.n_cond_onehot -= 1

    # cross-validation split (using name-wise K-fold)
    folds = make_folds_kfold(series_slices, K=args.n_folds)

    print(folds[args.cv_idx]['train_idx'][:10], folds[args.cv_idx]['valid_idx'][:10])
    train_dataset = TrajectoryDataset(args.Langevin_type, args.model_type, xs, dts, dxs=dxs, indices=folds[args.cv_idx]['train_idx'])
    valid_dataset = TrajectoryDataset(args.Langevin_type, args.model_type, xs, dts, dxs=dxs, indices=folds[args.cv_idx]['valid_idx'])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    if len(train_dataset) < args.batch_size:
        args.batch_size = len(train_dataset)
        print(f"Warning: training dataset size {len(train_dataset)} is smaller than batch size, set batch size to {args.batch_size} instead.")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers)

    print("Cross-validation fold:", args.cv_idx, f" out of {args.n_folds} folds")
    print(f"Langevin type: {args.Langevin_type}")
    print(f"Dimension: {args.dim}, Input dimension: {args.input_dim}, Cond.One-Hot: {args.n_cond_onehot}, Model type: {args.model_type}")
    print(f"Include time as input feature: {bool(args.include_time)}")
    if args.model_type == 'diff':
        print(f"Ensure PSD: {args.ensure_psd}")
    if args.use_LN == 0:
        print("Not using layer normalization in the model.")
    print(f"Total data size: {len(xs)}, Training data size: {len(train_dataset)}, Validation data size: {len(valid_dataset)}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, Learning rate: {args.lr}, Weight decay: {args.weight_decay}")

    # build model
    model = make_model(args, model_type=args.model_type).to(args.device)
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model from {args.load_path}")

    # optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler
    if args.scheduler == 'cosine':
        args.warmup_epochs = args.warmup_epochs if args.warmup_epochs >= 0 else 0
        
        sched_warmup = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=args.warmup_epochs)
        sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs - args.warmup_epochs, eta_min=1e-9)

        optim_scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [sched_warmup, sched_cosine], milestones=[args.warmup_epochs])
        print(f"Using cosine scheduler with warmup for {args.warmup_epochs} epochs.")
        
    else:
        # others (not implemented)
        print("No scheduler is used.")
        optim_scheduler = None

    # -------- ***Training loop*** --------
    initial_valid_loss = evaluate(model, 
                                  valid_loader, 
                                  args.model_type, 
                                  device=args.device
                                  )

    best_valid_loss = initial_valid_loss
    best_model, best_epoch = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0

    train_losses, reg_losses, valid_losses = [], [], [initial_valid_loss]
    print(f"Initial valid loss: {initial_valid_loss:.6f}")
    for epoch in range(1, args.epochs + 1):
        
        train_res = train_epoch(model, 
                                train_loader, 
                                optim, 
                                args.model_type, 
                                device=args.device, 
                                time_step=args.time_step, 
                                time_reg_lambda=args.time_reg_lambda, 
                                grad_clip=args.grad_clip, 
                                include_time=args.include_time
                                )

        train_losses += train_res['train_loss']
        reg_losses += train_res['reg_loss']

        # -------- Evaluation --------
        if (epoch) % args.record_interval == 0:
            if args.grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            valid_loss = evaluate(model, 
                                  valid_loader, 
                                  args.model_type, 
                                  device=args.device)

            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                print(f"New best valid loss: {valid_loss:.6f} at epoch {epoch}")
                best_valid_loss = valid_loss
                best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

            print(f"Epoch {epoch}/{args.epochs}, "
                f"Train Loss: {train_losses[-1]:.6f}, "
                f"Reg Loss: {reg_losses[-1]:.6f}, "
                f"Valid Loss: {valid_loss:.6f}, "
                f"Best Valid Loss: {best_valid_loss:.6f} at epoch {best_epoch}")
            if args.grad_clip > 0:
                print(f"Grad Norm: {total_norm}")

        if optim_scheduler is not None:
            optim_scheduler.step()
    
    # save the current model
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
    }, is_best=False, save_path=args.save_path, epoch=epoch, model_type=args.model_type)
    save_checkpoint({
        'epoch': best_epoch,
        'state_dict': best_model,
        'optimizer': optim.state_dict(),
    }, is_best=True, save_path=args.save_path, epoch=best_epoch, model_type=args.model_type)

    if args.save_swag:
        from LBN.misc.utils import collect_swag_snapshots, build_swag_stats

        model.load_state_dict(best_model)
        print(f"Loaded the best model at epoch {best_epoch} for SWAG.")

        snaps = collect_swag_snapshots(model, 
                                       train_loader, 
                                       valid_loader, 
                                       args.model_type,
                                       device=args.device,
                                       # training hyperparameters
                                       time_step=args.time_step,
                                       time_reg_lambda=args.time_reg_lambda,
                                       grad_clip=args.grad_clip,
                                       include_time=args.include_time,
                                       # SWAG hyperparameters
                                       lr_swag=args.lr_swag, 
                                       n_snaps=50,
                                       snap_every=args.snap_every,
                                       start_from_swa_state=best_model
                                    )
        model_mu, model_diag, model_S = build_swag_stats(snaps, rank_k=20)
        swag_state = {'model_mu': model_mu, 'model_diag': model_diag, 'model_S': model_S}

        torch.save(swag_state, os.path.join(args.save_path, f'swag_model_{args.model_type}.pt'))
        print(f"Saved SWAG model to swag_model_{args.model_type}.pt")

    with open(os.path.join(args.save_path, f'args_{args.model_type}.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, default=str)
    np.savez(os.path.join(args.save_path, f'losses_{args.model_type}.npz'), train_losses=np.array(train_losses), reg_losses=np.array(reg_losses), valid_losses=np.array(valid_losses))
    print("Training completed.")

    if args.output_path is not None:
        shutil.copy(args.output_path, args.save_path + '/' + args.model_type + '_out.log')


if __name__ == "__main__":
    from LBN.misc.utils import build_parser

    parser = build_parser()     # the argument parser is defined in misc/utils.py
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    args.device = torch.device("cuda" if use_cuda else "cpu")
    print(args.device)
    
    main(args)