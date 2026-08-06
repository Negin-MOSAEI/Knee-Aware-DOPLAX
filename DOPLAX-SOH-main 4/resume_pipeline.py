"""
Resume Pipeline: Train only what's missing.
1) DeepOPINN for TJU (missing)
2) Bagging per-batch for XJTU (6 batches)
3) Bagging per-batch for TJU (3 batches)
4) Post-processing
"""

import sys, os
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
os.environ['DDE_BACKEND'] = 'pytorch'

import torch
torch.set_default_device('cpu')
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

from utils.arguments import get_DeepOPINN_args, get_Bagging_u_args
from dataloader.data_helper import load_XJTU_data, load_TJU_data

sys.argv = _original_argv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUTS_ROOT = Path("outputs").resolve()

BATCHES = {
    "XJTU": ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"],
    "TJU":  ["NCA", "NCM", "NCM_NCA"],
}

DATASETS = {"XJTU": load_XJTU_data, "TJU": load_TJU_data}


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


def train_deepopinn(dataset_name):
    from Model.PI_nets import DeepOPINN
    ckpt_path = OUTPUTS_ROOT / dataset_name / "DeepOPINN" / "best_model.pth"
    if ckpt_path.exists():
        print(f"  [SKIP] DeepOPINN {dataset_name} already done")
        return

    print(f"\n{'='*60}")
    print(f"  Training DeepOPINN on {dataset_name}")
    print(f"{'='*60}")

    args = get_DeepOPINN_args()
    args.data = dataset_name
    args.batch = "All"
    args.batch_size = getattr(args, f"batch_size_{dataset_name}")
    args.save_folder = str(ckpt_path.parent)
    args.log_dir = "logging.txt"
    args.run_mode = "DeepOPINN"
    args.run_optuna = False
    args.run_samll_sample = False

    g_dim = detect_g_dim("data/Processed", dataset_name)
    args.g_dim_LAX_XJTU = g_dim

    dataloader = DATASETS[dataset_name](args, data_path="data/Processed")
    print(f"    Train: {len(dataloader['train'].dataset)} samples")
    print(f"    Valid: {len(dataloader['valid'].dataset)} samples")
    print(f"    Test:  {len(dataloader['test'].dataset)} samples")

    model = DeepOPINN.Model(args)
    result = model.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test'],
    )
    print(f"    DeepOPINN result: {result}")


def train_bagging_per_batch(dataset_name, batch_name):
    from Model.Combination_nets import Bagging_u

    results_dir = OUTPUTS_ROOT / dataset_name / batch_name / "Bagging"
    ckpt_path = results_dir / "best_model.pth"
    if ckpt_path.exists():
        print(f"  [SKIP] Bagging {dataset_name} [{batch_name}] already done")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    deepopinn_ckpt = OUTPUTS_ROOT / dataset_name / "DeepOPINN" / "best_model.pth"
    lax_ckpt = OUTPUTS_ROOT / dataset_name / "best_weights.pth"

    if not deepopinn_ckpt.exists():
        print(f"  ERROR: DeepOPINN checkpoint missing: {deepopinn_ckpt}")
        return
    if not lax_ckpt.exists():
        print(f"  ERROR: LAX checkpoint missing: {lax_ckpt}")
        return

    print(f"\n  Training Bagging for {dataset_name} [{batch_name}]")

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

    torch.set_default_device('cpu')
    dataloader = DATASETS[dataset_name](args, data_path="data/Processed")
    print(f"    Train: {len(dataloader['train'].dataset)} samples")
    print(f"    Valid: {len(dataloader['valid'].dataset)} samples")
    print(f"    Test:  {len(dataloader['test'].dataset)} samples")

    model = Bagging_u.Model(args)
    result = model.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test'],
    )
    print(f"    Bagging result: {result}")


def run_post_processing():
    print(f"\n{'='*60}")
    print(f"  Running post-processing pipeline")
    print(f"{'='*60}")
    from run_post_processing import main as pp_main
    pp_main()


def main():
    print("=" * 60)
    print("  Resume Pipeline (missing steps only)")
    print("=" * 60)
    print(f"  Device: {device}")

    # ── Step 1: DeepOPINN for TJU (missing) ──
    if not (OUTPUTS_ROOT / "TJU" / "DeepOPINN" / "best_model.pth").exists():
        train_deepopinn("TJU")
    else:
        print("  [SKIP] DeepOPINN TJU already done")

    # ── Step 2: Bagging per-batch for XJTU ──
    print(f"\n{'='*60}")
    print(f"  Step 2: Bagging per-batch for XJTU")
    print(f"{'='*60}")
    for bn in BATCHES["XJTU"]:
        train_bagging_per_batch("XJTU", bn)

    # ── Step 3: Bagging per-batch for TJU ──
    print(f"\n{'='*60}")
    print(f"  Step 3: Bagging per-batch for TJU")
    print(f"{'='*60}")
    for bn in BATCHES["TJU"]:
        train_bagging_per_batch("TJU", bn)

    # ── Step 4: Post-processing ──
    run_post_processing()

    print(f"\n{'='*60}")
    print(f"  All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
