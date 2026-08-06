"""
Unified Multi-Dataset Bagging Training Runner
==============================================
Phase 1: Trains DeepOPINN on all 4 datasets
Phase 2: Trains Bagging_u on all 4 datasets (frozen DeepOPINN + frozen LAX)

Usage:
    python run_all_bagging.py
"""

import sys
import os

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]

os.environ['DDE_BACKEND'] = 'pytorch'

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from utils.arguments import get_DeepOPINN_args, get_Bagging_u_args
from dataloader.data_helper import load_XJTU_data, load_TJU_data, load_MIT_data, load_HUST_data
from Model.PI_nets import DeepOPINN
from Model.Combination_nets import Bagging_u

sys.argv = _original_argv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def resolve_lax_args(args):
    """Copy dataset-specific LAX args into generic names (mirrors run_all_datasets.py)."""
    ds = args.data
    suffix = {'XJTU': '_XJTU', 'TJU': '_TJU', 'MIT': '_MIT'}.get(ds, '_HUST')
    mapping = {
        'betha_LAX': f'betha_LAX{suffix}',
        'dual_LAX': f'dual_LAX{suffix}',
        'theta_LAX': f'theta_LAX{suffix}',
        'lr_net': f'lr_net_LAX{suffix}',
        'h_dim_LAX': f'h_dim_LAX{suffix}',
        'beta_LAX': f'beta_LAX{suffix}',
        'distance_block_LAX': f'distance_block_LAX{suffix}',
        'time_block_LAX': f'time_block_LAX{suffix}',
        'center_block_LAX': f'center_block_LAX{suffix}',
        'inside_distance_block_MLP_layers': f'inside_distance_block_MLP_layers_LAX{suffix}',
        'inside_theta_layers': f'inside_theta_layers_LAX{suffix}',
        'inside_multivar_theta_layers': f'inside_multivar_theta_layers_LAX{suffix}',
        'inside_h_star_layers': f'inside_h_star_layers_LAX{suffix}',
        'inside_phi_layers': f'inside_phi_layers_LAX{suffix}',
        'inside_g_layers': f'inside_g_layers_LAX{suffix}',
        'inside_S_MLP_layers': f'inside_S_MLP_layers_LAX{suffix}',
        'inside_betan_layers': f'inside_betan_layers_LAX{suffix}',
        'g_dim': f'g_dim_LAX{suffix}',
        'h_out_LAX': f'H_out_LAX{suffix}',
        'g_out_LAX': f'g_out_LAX{suffix}',
        'phi_out_LAX': f'phi_out_LAX{suffix}',
        'dim_output_LAX': f'dim_output_LAX{suffix}',
        'lr_F': f'lr_F{suffix}',
    }
    for generic, specific in mapping.items():
        if hasattr(args, specific):
            setattr(args, generic, getattr(args, specific))

DATASETS = {
    "XJTU": {
        "data": "XJTU",
        "batch": "All",
        "loader": load_XJTU_data,
        "data_path": "data/Processed",
    },
    "TJU": {
        "data": "TJU",
        "batch": "All",
        "loader": load_TJU_data,
        "data_path": "data/Processed",
    },
    "MIT": {
        "data": "MIT",
        "batch": "one_batch",
        "loader": load_MIT_data,
        "data_path": "data/Processed",
    },
    "HUST": {
        "data": "HUST",
        "batch": "one_batch",
        "loader": load_HUST_data,
        "data_path": "data/Processed",
    },
}


def detect_g_dim(data_path, dataset_name):
    root = os.path.join(data_path, dataset_name + " data")
    for dirpath, dirnames, filenames in os.walk(root):
        csv_files = sorted([f for f in filenames if f.endswith(".csv")])
        if csv_files:
            df = pd.read_csv(os.path.join(dirpath, csv_files[0]))
            return df.shape[1] - 1
    raise FileNotFoundError(f"No CSV files found under {root}")


# ============================================================================
# Phase 1: DeepOPINN Training
# ============================================================================
def train_deepopinn(dataset_name):
    cfg = DATASETS[dataset_name]
    results_dir = os.path.join(OUTPUTS_ROOT, dataset_name, "DeepOPINN")
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "best_model.pth")

    print(f"\n{'='*60}")
    print(f"  Phase 1: Training DeepOPINN on {dataset_name} ({cfg['batch']})")
    print(f"{'='*60}")

    if os.path.exists(ckpt_path):
        print(f"  Checkpoint exists at {ckpt_path}, skipping.")
        return ckpt_path

    args = get_DeepOPINN_args()
    args.data = cfg["data"]
    args.batch = cfg["batch"]
    args.batch_size = getattr(args, f"batch_size_{dataset_name}")
    args.save_folder = results_dir
    args.log_dir = "logging.txt"
    args.run_mode = "DeepOPINN"
    args.run_optuna = False
    args.run_samll_sample = False

    g_dim = detect_g_dim(cfg["data_path"], dataset_name)
    args.g_dim_LAX_XJTU = g_dim

    dataloader = cfg["loader"](args, data_path=cfg["data_path"])
    print(f"  Train: {len(dataloader['train'].dataset)} samples")
    print(f"  Valid: {len(dataloader['valid'].dataset)} samples")
    print(f"  Test:  {len(dataloader['test'].dataset)} samples")
    print(f"  g_dim: {g_dim}")
    print(f"  Training ...")

    model = DeepOPINN.Model(args)
    result = model.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test'],
    )
    print(f"  DeepOPINN training result: {result}")
    return ckpt_path


# ============================================================================
# Phase 2: Bagging Training
# ============================================================================
def train_bagging(dataset_name):
    cfg = DATASETS[dataset_name]
    results_dir = os.path.join(OUTPUTS_ROOT, dataset_name, "Bagging")
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "best_model.pth")

    deepopinn_ckpt = os.path.join(OUTPUTS_ROOT, dataset_name, "DeepOPINN", "best_model.pth")
    lax_ckpt = os.path.join(OUTPUTS_ROOT, dataset_name, "best_weights.pth")

    print(f"\n{'='*60}")
    print(f"  Phase 2: Training Bagging_u on {dataset_name} ({cfg['batch']})")
    print(f"{'='*60}")

    if not os.path.exists(deepopinn_ckpt):
        print(f"  ERROR: DeepOPINN checkpoint not found at {deepopinn_ckpt}")
        return None
    if not os.path.exists(lax_ckpt):
        print(f"  ERROR: LAX checkpoint not found at {lax_ckpt}")
        return None

    if os.path.exists(ckpt_path):
        print(f"  Checkpoint exists at {ckpt_path}, skipping.")
        return ckpt_path

    args = get_Bagging_u_args()
    args.data = cfg["data"]
    args.batch = cfg["batch"]
    args.batch_size = getattr(args, f"batch_size_{dataset_name}")
    args.save_folder = results_dir
    args.log_dir = "logging.txt"
    args.run_mode = "Bagging_u"
    args.run_optuna = False
    args.run_samll_sample = False
    setattr(args, 'run_for_DeepOPINN', False)
    setattr(args, 'run_for_LAX', False)
    args.DOP_weights = deepopinn_ckpt
    args.LAX_weights = lax_ckpt
    args.trained_by_y_opt_LAX = False

    g_dim = detect_g_dim(cfg["data_path"], dataset_name)
    args.g_dim_LAX_XJTU = g_dim
    args.g_dim_LAX_TJU = g_dim
    args.g_dim_LAX_MIT = g_dim
    args.g_dim_LAX_HUST = g_dim

    args.distance_block_LAX_XJTU = "MLP"
    args.distance_block_LAX_TJU = "MLP"
    args.distance_block_LAX_MIT = "MLP"
    args.distance_block_LAX_HUST = "MLP"
    args.h_dim_LAX_MIT = 30

    if dataset_name == "HUST":
        args.bagging_NN_lr_HUST = 0.02
        args.mono_bag_HUST = 0.1
        args.bag_hidden_dim_HUST = [100, 100]

    resolve_lax_args(args)

    dataloader = cfg["loader"](args, data_path=cfg["data_path"])
    print(f"  Train: {len(dataloader['train'].dataset)} samples")
    print(f"  Valid: {len(dataloader['valid'].dataset)} samples")
    print(f"  Test:  {len(dataloader['test'].dataset)} samples")
    print(f"  g_dim: {g_dim}")
    print(f"  DOP weights: {deepopinn_ckpt}")
    print(f"  LAX weights: {lax_ckpt}")
    print(f"  Training ...")

    model = Bagging_u.Model(args)
    result = model.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test'],
    )
    print(f"  Bagging training result: {result}")
    return ckpt_path


def main():
    print("=" * 60)
    print("  Unified Multi-Dataset Bagging Training Runner")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Output: {OUTPUTS_ROOT}")

    for dataset_name in DATASETS:
        try:
            train_deepopinn(dataset_name)
        except Exception as e:
            print(f"\n  ERROR on DeepOPINN {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    for dataset_name in DATASETS:
        try:
            train_bagging(dataset_name)
        except Exception as e:
            print(f"\n  ERROR on Bagging {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  All Bagging training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
