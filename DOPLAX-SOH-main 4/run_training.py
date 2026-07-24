"""
Standalone LAX Model Training Runner
=====================================
Trains the LAX (Learnable Activation eXponential) model on XJTU 2C battery data
with knee-point-distance features. Saves all artifacts to outputs/.
"""

import sys
import os
import shutil

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from utils.arguments import get_LAX_args
from dataloader.data_helper import load_XJTU_data
from dataloader.dataloader import DF
from Model.PI_nets.LAX import (
    run_training_and_evaluation,
    OptimizationNetwork,
    save_LAX_results,
)
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


def _apply_lax_args_for_xjtU(args):
    """Replicate the XJTU arg assignment from run_training_and_evaluation."""
    args.betha_LAX = args.betha_LAX_XJTU
    args.dual_LAX = args.dual_LAX_XJTU
    args.theta_LAX = args.theta_LAX_XJTU
    args.lr_net = args.lr_net_LAX_XJTU
    args.h_dim_LAX = args.h_dim_LAX_XJTU
    args.beta_LAX = args.beta_LAX_XJTU
    args.distance_block_LAX = args.distance_block_LAX_XJTU
    args.time_block_LAX = args.time_block_LAX_XJTU
    args.center_block_LAX = args.center_block_LAX_XJTU
    args.inside_distance_block_MLP_layers = args.inside_distance_block_MLP_layers_LAX_XJTU
    args.inside_theta_layers = args.inside_theta_layers_LAX_XJTU
    args.inside_multivar_theta_layers = args.inside_multivar_theta_layers_LAX_XJTU
    args.inside_h_star_layers = args.inside_h_star_layers_LAX_XJTU
    args.inside_phi_layers = args.inside_phi_layers_LAX_XJTU
    args.inside_g_layers = args.inside_g_layers_LAX_XJTU
    args.inside_S_MLP_layers = args.inside_S_MLP_layers_LAX_XJTU
    args.inside_betan_layers = args.inside_betan_layers_LAX_XJTU
    args.g_dim = args.g_dim_LAX_XJTU
    args.h_out_LAX = args.H_out_LAX_XJTU
    args.g_out_LAX = args.g_out_LAX_XJTU
    args.phi_out_LAX = args.phi_out_LAX_XJTU
    args.dim_output_LAX = args.dim_output_LAX_XJTU
    args.lr_F = args.lr_F_XJTU
    return args


def _load_single_battery_tensors(csv_path, df_helper):
    """Load one processed CSV and return (x1, y1, n_cycles) for evaluation.

    Follows the exact same preprocessing as load_one_battery / run_epoch:
      - read_one_csv  -> inserts cycle_index, normalises features, fillna
      - x = df.iloc[:, :-1]   (features + cycle_index, capacity removed)
      - y = df.iloc[:, -1]    (capacity)
      - x1 = x[:-1], y1 = y[:-1]  (shifted pairs, N-1 samples)
    """
    df = df_helper.read_one_csv(csv_path)
    n_cycles = len(df)

    x = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)

    x1 = x[:-1]
    y1 = y[:-1]
    return (
        torch.from_numpy(x1),
        torch.from_numpy(y1).view(-1, 1),
        n_cycles,
    )


def load_and_evaluate_per_battery(args, dataloader):
    """Load the best checkpoint, evaluate per-battery, return per-cell results.

    Returns
    -------
    cell_results : list[dict]
        Each dict has keys: name, n_cycles, y_true, y_pred, cycle_indices
    """
    _apply_lax_args_for_xjtU(args)

    train_loader = dataloader['train']
    sum_ = 0.0
    sum_sq = 0.0
    n_samples = 0
    x_dim = y_dim = 0
    for batch in train_loader:
        if len(batch) == 6:
            x1, x2 = batch[0], batch[1]
        else:
            x1, x2 = batch[0], batch[1]
        x_dim = y_dim = x1.shape[1] - 1
        x1f = x1[:, :x_dim]
        x2f = x2[:, :x_dim]
        sum_ += x2f.sum(dim=0)
        sum_sq += (x2f ** 2).sum(dim=0)
        n_samples += x2f.size(0)
    X_mean = sum_ / n_samples
    X_std = torch.sqrt((sum_sq / n_samples) - (X_mean ** 2))

    model = OptimizationNetwork(
        x_sts=(X_mean, X_std),
        y_dim=y_dim,
        x_dim=x_dim,
        args=args,
    ).to(device)

    ckpt_path = os.path.join(args.results_path, 'best_weights.pth')
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state']

    if 'y' in state_dict and state_dict['y'].shape != model.y.shape:
        state_dict['y'] = model.y.data

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded best model from {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")

    df_helper = DF(args)
    data_root = os.path.join('data', 'Processed', 'XJTU data')
    csv_files = sorted([f for f in os.listdir(data_root) if f.endswith('.csv')])

    cell_results = []
    for fname in csv_files:
        csv_path = os.path.join(data_root, fname)
        x1, y1_true, n_cycles = _load_single_battery_tensors(csv_path, df_helper)

        ds = TensorDataset(x1, y1_true)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

        preds = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                x_in = batch_x[:, :-1]
                t_in = x_in[:, -1]
                pred, _ = model(x=x_in, t=t_in.squeeze(), epoch=args.epoch_net, return_f=True)
                preds.append(pred.squeeze(-1).cpu())

        y_pred = torch.cat(preds).numpy()
        y_true = y1_true.numpy().flatten()

        cycle_indices = np.arange(len(y_true))

        cell_results.append({
            'name': fname.replace('.csv', ''),
            'n_cycles': n_cycles,
            'y_true': y_true,
            'y_pred': y_pred,
            'cycle_indices': cycle_indices,
        })
        print(f"  Evaluated {fname}: {len(y_true)} samples")

    return cell_results


def generate_per_cell_prediction_plots(args, cell_results):
    """Generate a 2x4 subplot grid: one subplot per battery cell."""
    plots_dir = os.path.join(args.results_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for idx, cell in enumerate(cell_results):
        if idx >= 8:
            break
        ax = axes[idx]
        ax.plot(cell['cycle_indices'], cell['y_true'], label='True', linewidth=1.0, color='#1976D2')
        ax.plot(cell['cycle_indices'], cell['y_pred'], label='Predicted', linewidth=1.0, color='#FF6F00', alpha=0.85)
        ax.set_title(cell['name'], fontsize=10)
        ax.set_xlabel('Cycle Index', fontsize=8)
        ax.set_ylabel('Normalized Capacity', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Per-Cell Capacity Prediction — XJTU 2C (LAX)', fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(plots_dir, 'predicted_vs_actual_per_cell.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-cell prediction plot saved to {out_path}")


def generate_combined_prediction_plot(args, cell_results):
    """Generate a combined plot with all cells and vertical boundary lines."""
    plots_dir = os.path.join(args.results_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 6))

    global_offset = 0
    boundary_positions = []
    colors_true = ['#1976D2', '#388E3C', '#D32F2F', '#7B1FA2',
                   '#F57C00', '#00796B', '#5D4037', '#455A64']
    colors_pred = ['#90CAF9', '#A5D6A7', '#EF9A9A', '#CE93D8',
                   '#FFCC80', '#80CBC4', '#BCAAA4', '#B0BEC5']

    for idx, cell in enumerate(cell_results):
        n = len(cell['y_true'])
        x_axis = np.arange(n) + global_offset

        c_true = colors_true[idx % len(colors_true)]
        c_pred = colors_pred[idx % len(colors_pred)]

        ax.plot(x_axis, cell['y_true'], color=c_true, linewidth=0.9, label=f'{cell["name"]} (True)' if idx < 4 else None)
        ax.plot(x_axis, cell['y_pred'], color=c_pred, linewidth=0.9, linestyle='--', alpha=0.85,
                label=f'{cell["name"]} (Pred)' if idx < 4 else None)

        if idx > 0:
            boundary_positions.append(global_offset - 0.5)

        global_offset += n

    for bp in boundary_positions:
        ax.axvline(x=bp, color='red', linestyle=':', linewidth=0.8, alpha=0.6)

    ax.set_xlabel('Cycle Index (concatenated)', fontsize=11)
    ax.set_ylabel('Normalized Capacity', fontsize=11)
    ax.set_title('Combined Capacity Prediction with Cell Boundaries — XJTU 2C (LAX)', fontsize=13)
    ax.grid(True, alpha=0.2)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='gray', linewidth=1, label='True'),
                       Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Predicted'),
                       Line2D([0], [0], color='red', linestyle=':', linewidth=0.8, label='Cell Boundary')]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(plots_dir, 'predicted_vs_actual.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined prediction plot saved to {out_path}")


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

    print("\nGenerating per-cell evaluation ...")
    cell_results = load_and_evaluate_per_battery(args, dataloader)

    print("\nGenerating output artifacts ...")
    generate_per_cell_prediction_plots(args, cell_results)
    generate_combined_prediction_plot(args, cell_results)
    generate_knee_distance_curve(args)
    generate_combined_loss_plot(args)
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
