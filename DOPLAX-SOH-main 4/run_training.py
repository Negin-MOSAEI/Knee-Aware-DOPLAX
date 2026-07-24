"""
Standalone LAX Model Training Runner
=====================================
Trains the LAX (Learnable Activation eXponential) model on XJTU 2C battery data
with knee-point-distance features. Saves all artifacts to outputs/.
"""

import sys
import os
import shutil

# Clear sys.argv so argparse inside get_LAX_args doesn't choke on our script args
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

from utils.arguments import get_LAX_args
from dataloader.data_helper import load_XJTU_data
from Model.PI_nets.LAX import run_training_and_evaluation, save_LAX_results
from knee_point_detection import KneePointDetector, knee_aware_branch_weights

sys.argv = _original_argv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_args():
    args = get_LAX_args()
    args.data = 'XJTU'
    args.batch = '2C'
    args.run_mode = 'LAX'
    args.run_optuna = False
    args.run_samll_sample = False

    args.results_path = str(Path('outputs').resolve())
    Path(args.results_path).mkdir(parents=True, exist_ok=True)

    log_dir = 'logging.txt'
    args.log_dir = log_dir

    sample_csv = pd.read_csv(os.path.join('data', 'Processed', 'XJTU data', '2C_battery-1.csv'))
    n_features = sample_csv.shape[1] - 1
    args.g_dim_LAX_XJTU = n_features

    return args


def generate_knee_distance_curve(args):
    """Plot the knee-point-distance for each test battery."""
    data_root = os.path.join('data', 'Processed', 'XJTU data')
    detector = KneePointDetector(n_percent=0.5)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, fname in enumerate(sorted(os.listdir(data_root))):
        if not fname.endswith('.csv'):
            continue
        if idx >= 8:
            break
        fpath = os.path.join(data_root, fname)
        df = pd.read_csv(fpath)
        if 'knee_point_distance' in df.columns:
            kd = df['knee_point_distance'].values
            axes[idx].plot(kd, linewidth=1.2, color='#2196F3')
            axes[idx].set_title(fname.replace('.csv', ''), fontsize=10)
            axes[idx].set_xlabel('Cycle Index')
            axes[idx].set_ylabel('Knee Distance')
            axes[idx].set_ylim(-0.05, 1.1)
            axes[idx].grid(True, alpha=0.3)

            knee_idx = detector.detect_knee_point(df['capacity'].values)
            if knee_idx is not None and knee_idx < len(kd):
                axes[idx].axvline(x=knee_idx, color='red', linestyle='--',
                                  linewidth=1, alpha=0.7, label=f'Knee@{knee_idx}')
                axes[idx].legend(fontsize=8)

    plt.suptitle('Knee Point Distance Curves — XJTU 2C Batteries', fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(args.results_path, 'plots', 'knee_distance_curve.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Knee distance curve saved to {out_path}")


def generate_combined_loss_plot(args):
    """Create a combined train/val loss plot from the individual loss plots."""
    plots_dir = os.path.join(args.results_path, 'plots')

    metric_files = {
        'MSE': 'loss_mse.png',
        'RMSE': 'loss_rmse.png',
        'MAPE': 'loss_mape.png',
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, fname) in zip(axes, metric_files.items()):
        fpath = os.path.join(args.results_path, fname)
        if os.path.exists(fpath):
            img = plt.imread(fpath)
            ax.imshow(img)
            ax.set_title(f'{name} Loss', fontsize=12)
        ax.axis('off')

    plt.suptitle('Training & Validation Loss Curves', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(plots_dir, 'train_val_loss.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined loss plot saved to {out_path}")


def generate_predicted_vs_actual_plot(args):
    """Copy the capacity prediction plot into the plots directory."""
    src = os.path.join(args.results_path, '2C-Experiments', 'capacity_prediction_2C.png')
    dst = os.path.join(args.results_path, 'plots', 'predicted_vs_actual.png')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Predicted vs actual plot saved to {dst}")
    else:
        print(f"Warning: capacity prediction plot not found at {src}")


def generate_metrics_csv(args, test_metrics):
    """Write a clean metrics.csv summary."""
    metrics_dir = os.path.join(args.results_path)
    rows = []
    for key in ['mse', 'rmse', 'mape', 'mae', 'dual', 'pde']:
        if key in test_metrics:
            val = test_metrics[key]
            rows.append({
                'Metric': key.upper(),
                'Value': val if isinstance(val, float) else float(val)
            })
    df = pd.DataFrame(rows)
    out_path = os.path.join(metrics_dir, 'metrics.csv')
    df.to_csv(out_path, index=False)
    print(f"Metrics saved to {out_path}")


def generate_logging_txt(args, test_metrics):
    """Write a human-readable logging.txt."""
    lines = [
        "=" * 60,
        "LAX Training Run — Knee-Aware DOPLAX",
        "=" * 60,
        f"Dataset:       {args.data}",
        f"Batch:         {args.batch}",
        f"Epochs (net):  {args.epoch_net}",
        f"Batch size:    {args.batch_size}",
        f"Device:        {device}",
        f"Results path:  {args.results_path}",
        "",
        "--- Test Metrics ---",
    ]
    for key in ['mse', 'rmse', 'mape', 'mae', 'dual', 'pde']:
        if key in test_metrics:
            val = test_metrics[key]
            lines.append(f"  {key.upper():>6s}: {val:.8f}" if isinstance(val, float) else f"  {key.upper():>6s}: {val}")
    lines.append("=" * 60)

    out_path = os.path.join(args.results_path, 'logging.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Logging saved to {out_path}")


def main():
    print("=" * 60)
    print("  Knee-Aware LAX Training — XJTU 2C Dataset")
    print("=" * 60)

    args = setup_args()

    print(f"\nConfiguration:")
    print(f"  Data:         {args.data}")
    print(f"  Batch:        {args.batch}")
    print(f"  Epochs:       {args.epoch_net}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Results path: {args.results_path}")
    print(f"  Device:       {device}")
    print()

    data_path = os.path.join('data', 'Processed')
    print(f"Loading data from {data_path} ...")
    dataloader = load_XJTU_data(args, data_path=data_path)
    print(f"  Train batches: {len(dataloader['train'])}")
    print(f"  Valid batches: {len(dataloader['valid'])}")
    print(f"  Test  batches: {len(dataloader['test'])}")
    print()

    print("Starting LAX training ...")
    test_metrics = run_training_and_evaluation(
        args=args,
        dataloader=dataloader,
        batch_name=args.batch,
        experiment_id=0,
    )
    print(f"\nTraining complete. Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k.upper():>6s}: {v:.8f}" if isinstance(v, float) else f"  {k.upper():>6s}: {v}")

    print("\nGenerating output artifacts ...")
    generate_knee_distance_curve(args)
    generate_combined_loss_plot(args)
    generate_predicted_vs_actual_plot(args)
    generate_metrics_csv(args, test_metrics)
    generate_logging_txt(args, test_metrics)

    print("\n" + "=" * 60)
    print("  All artifacts saved to outputs/")
    print("=" * 60)
    print("\nContents:")
    for root, dirs, files in os.walk(args.results_path):
        level = root.replace(args.results_path, '').count(os.sep)
        indent = '  ' * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = '  ' * (level + 1)
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f))
            print(f'{subindent}{f}  ({size:,} bytes)')


if __name__ == '__main__':
    main()
