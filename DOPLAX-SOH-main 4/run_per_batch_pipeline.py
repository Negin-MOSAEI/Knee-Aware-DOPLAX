"""
Per-Batch Training + Post-Processing Pipeline
==============================================
1) Train LAX + DeepOPINN once per dataset (all data)
2) Train Bagging separately for each batch
3) Run post-processing & plots per batch

Usage:
    python run_per_batch_pipeline.py
"""

import sys, os
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
os.environ['DDE_BACKEND'] = 'pytorch'

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

from utils.arguments import get_Bagging_u_args
from dataloader.data_helper import load_XJTU_data, load_TJU_data

sys.argv = _original_argv

# NOTE: Do NOT import DeepOPINN or Bagging_u at module level —
# they import deepxde which sets torch default device to cuda,
# breaking DataLoader shuffle. Import them only inside functions.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUTS_ROOT = Path("outputs").resolve()

DATASETS = {"XJTU": load_XJTU_data, "TJU": load_TJU_data}

BATCHES = {
    "XJTU": ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"],
    "TJU":  ["NCA", "NCM", "NCM_NCA"],
}

def detect_g_dim(data_path, dataset_name):
    root = os.path.join(data_path, dataset_name + " data")
    for dirpath, _, filenames in os.walk(root):
        csvs = sorted([f for f in filenames if f.endswith(".csv")])
        if csvs:
            df = pd.read_csv(os.path.join(dirpath, csvs[0]))
            return df.shape[1] - 1
    return 16

def resolve_lax_args(args):
    ds = args.data
    suffix = {'XJTU': '_XJTU', 'TJU': '_TJU'}.get(ds, '_HUST')
    mapping = {
        'betha_LAX': f'betha_LAX{suffix}', 'dual_LAX': f'dual_LAX{suffix}',
        'theta_LAX': f'theta_LAX{suffix}', 'lr_net': f'lr_net_LAX{suffix}',
        'h_dim_LAX': f'h_dim_LAX{suffix}', 'beta_LAX': f'beta_LAX{suffix}',
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
        'g_dim': f'g_dim_LAX{suffix}', 'h_out_LAX': f'H_out_LAX{suffix}',
        'g_out_LAX': f'g_out_LAX{suffix}', 'phi_out_LAX': f'phi_out_LAX{suffix}',
        'dim_output_LAX': f'dim_output_LAX{suffix}', 'lr_F': f'lr_F{suffix}',
    }
    for generic, specific in mapping.items():
        if hasattr(args, specific):
            setattr(args, generic, getattr(args, specific))

def train_lax(dataset_name):
    """Train LAX once per dataset on all data."""
    from run_all_datasets import run_dataset
    print(f"\n{'='*60}")
    print(f"  Training LAX on {dataset_name} (all data)")
    print(f"{'='*60}")
    run_dataset(dataset_name)

def train_deepopinn(dataset_name):
    """Train DeepOPINN once per dataset on all data."""
    from run_all_bagging import train_deepopinn as _train_dop
    _train_dop(dataset_name)

def train_bagging_per_batch(dataset_name, batch_name):
    """Train Bagging_u for a specific batch."""
    results_dir = OUTPUTS_ROOT / dataset_name / batch_name / "Bagging"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = results_dir / "best_model.pth"

    deepopinn_ckpt = OUTPUTS_ROOT / dataset_name / "DeepOPINN" / "best_model.pth"
    lax_ckpt = OUTPUTS_ROOT / dataset_name / "best_weights.pth"

    if not deepopinn_ckpt.exists():
        print(f"  ERROR: DeepOPINN checkpoint missing: {deepopinn_ckpt}")
        return None
    if not lax_ckpt.exists():
        print(f"  ERROR: LAX checkpoint missing: {lax_ckpt}")
        return None

    print(f"\n  Training Bagging for {dataset_name} [{batch_name}]")
    loader_fn = DATASETS[dataset_name]

    args = get_Bagging_u_args()
    args.data = dataset_name
    args.batch = batch_name
    args.batch_size = getattr(args, f"batch_size_{dataset_name}")
    args.save_folder = str(results_dir)
    args.log_dir = "logging.txt"
    args.run_mode = "Bagging_u"
    args.run_optuna = False
    args.run_samll_sample = False
    setattr(args, 'run_for_DeepOPINN', False)
    setattr(args, 'run_for_LAX', False)
    args.DOP_weights = str(deepopinn_ckpt)
    args.LAX_weights = str(lax_ckpt)
    args.trained_by_y_opt_LAX = False

    g_dim = detect_g_dim("data/Processed", dataset_name)
    setattr(args, 'g_dim_LAX_XJTU', g_dim)
    setattr(args, 'g_dim_LAX_TJU', g_dim)
    setattr(args, 'g_dim_LAX_MIT', g_dim)
    setattr(args, 'g_dim_LAX_HUST', g_dim)
    setattr(args, f'distance_block_LAX_{dataset_name}', "MLP")
    args.h_dim_LAX_MIT = 30

    resolve_lax_args(args)

    # ── DataLoader first (CPU device to avoid deepxde CUDA conflict) ──
    torch.set_default_device('cpu')
    dataloader = loader_fn(args, data_path="data/Processed")
    print(f"    Train: {len(dataloader['train'].dataset)} samples")
    print(f"    Valid: {len(dataloader['valid'].dataset)} samples")
    print(f"    Test:  {len(dataloader['test'].dataset)} samples")

    # ── Now import deepxde-dependent modules (restores CUDA default) ──
    from Model.Combination_nets import Bagging_u
    model = Bagging_u.Model(args)
    result = model.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test'],
    )
    print(f"    Bagging result: {result}")
    return ckpt_path

def run_post_processing():
    """Regenerate all plots and metrics."""
    print(f"\n{'='*60}")
    print(f"  Running post-processing pipeline")
    print(f"{'='*60}")
    from run_post_processing import main as pp_main
    pp_main()

def delete_outputs():
    """Delete TJU and XJTU outputs and all plots."""
    import shutil
    for d in ["TJU", "XJTU"]:
        dpath = OUTPUTS_ROOT / d
        if dpath.exists():
            shutil.rmtree(dpath)
            print(f"  Deleted {dpath}")
    ppath = OUTPUTS_ROOT / "plots"
    if ppath.exists():
        for f in ppath.glob("*"):
            if f.is_file():
                f.unlink()
        print(f"  Cleared {ppath}")
    comp_csv = OUTPUTS_ROOT / "overall_dataset_comparison.csv"
    if comp_csv.exists():
        comp_csv.unlink()
        print(f"  Deleted {comp_csv}")

def main():
    print("=" * 60)
    print("  Per-Batch Training + Post-Processing Pipeline")
    print("=" * 60)
    print(f"  Device: {device}")

    # ── Step 1: Delete old outputs ──
    print(f"\n{'='*60}")
    print(f"  Step 1: Delete old outputs")
    print(f"{'='*60}")
    delete_outputs()

    # ── Step 2: Train LAX (once per dataset, all data) ──
    print(f"\n{'='*60}")
    print(f"  Step 2: Train LAX")
    print(f"{'='*60}")
    for ds in DATASETS:
        train_lax(ds)

    # ── Step 3: Train DeepOPINN (once per dataset, all data) ──
    print(f"\n{'='*60}")
    print(f"  Step 3: Train DeepOPINN")
    print(f"{'='*60}")
    for ds in DATASETS:
        train_deepopinn(ds)

    # ── Step 4: Train Bagging per batch ──
    print(f"\n{'='*60}")
    print(f"  Step 4: Train Bagging per batch")
    print(f"{'='*60}")
    for ds in DATASETS:
        for bn in BATCHES[ds]:
            train_bagging_per_batch(ds, bn)

    # ── Step 5: Post-processing & plots ──
    print(f"\n{'='*60}")
    print(f"  Step 5: Post-processing & plots")
    print(f"{'='*60}")
    run_post_processing()

    print(f"\n{'='*60}")
    print(f"  All done!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
