"""
Unified Multi-Dataset LAX Training Runner
==========================================
Trains the LAX model on XJTU, TJU, MIT, and HUST datasets.
Saves per-dataset artifacts and an overall comparison table.

Usage:
    python run_all_datasets.py
"""

import sys
import os

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
from scipy.stats import pearsonr

from utils.arguments import get_LAX_args
from dataloader.data_helper import load_XJTU_data, load_TJU_data, load_MIT_data, load_HUST_data
from dataloader.dataloader import DF
from Model.PI_nets.LAX import (
    run_training_and_evaluation,
    OptimizationNetwork,
    evaluate_on_set,
)

sys.argv = _original_argv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUTS_ROOT = Path("outputs").resolve()

# ============================================================================
# Dataset configurations
# ============================================================================
DATASETS = {
    "XJTU": {
        "data": "XJTU",
        "batch": "3C",
        "g_dim_attr": "g_dim_LAX_XJTU",
        "loader": load_XJTU_data,
        "data_path": "data/Processed",
    },
    "TJU": {
        "data": "TJU",
        "batch": "NCM_NCA",
        "g_dim_attr": "g_dim_LAX_TJU",
        "loader": load_TJU_data,
        "data_path": "data/Processed",
    },
    "MIT": {
        "data": "MIT",
        "batch": "one_batch",
        "g_dim_attr": "g_dim_LAX_MIT",
        "loader": load_MIT_data,
        "data_path": "data/Processed",
    },
    "HUST": {
        "data": "HUST",
        "batch": "one_batch",
        "g_dim_attr": "g_dim_LAX_HUST",
        "loader": load_HUST_data,
        "data_path": "data/Processed",
    },
}


def setup_args(dataset_name):
    """Configure args for a specific dataset."""
    cfg = DATASETS[dataset_name]
    args = get_LAX_args()
    args.data = cfg["data"]
    args.batch = cfg["batch"]
    args.run_mode = "LAX"
    args.run_optuna = False
    args.run_samll_sample = False

    results_dir = OUTPUTS_ROOT / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    args.results_path = str(results_dir)
    args.log_dir = "logging.txt"

    return args


def find_first_csv(data_path, dataset_name, batch):
    """Recursively find a sample CSV to determine feature count."""
    cfg = DATASETS[dataset_name]
    root = os.path.join(data_path, cfg["data"] + " data")
    if not os.path.isdir(root):
        root = os.path.join(data_path, cfg["data"] + " data")

    for dirpath, dirnames, filenames in os.walk(root):
        csv_files = sorted([f for f in filenames if f.endswith(".csv")])
        if csv_files:
            return os.path.join(dirpath, csv_files[0])

    raise FileNotFoundError(f"No CSV files found under {root}")


def detect_g_dim(data_path, dataset_name, batch):
    """Read a sample CSV to determine feature count for g_dim."""
    csv_path = find_first_csv(data_path, dataset_name, batch)
    df = pd.read_csv(csv_path)
    return df.shape[1] - 1  # minus capacity column


def compute_pearson_r(model, dataloader, args):
    """Run inference on test set and compute Pearson R."""
    model.eval()
    all_y_true = []
    all_y_pred = []

    for batch in dataloader:
        if len(batch) == 6:
            x1, x2, y1, y2 = batch[0], batch[1], batch[2], batch[3]
        else:
            x1, x2, y1, y2 = batch
        x1 = x1.to(device)
        y1 = y1.to(device)

        x_in = x1[:, :-1]
        t_in = x_in[:, -1]
        with torch.no_grad():
            pred, _ = model(x=x_in, t=t_in.squeeze(), epoch=args.epoch_net, return_f=True)
        all_y_true.append(y1.detach().cpu().numpy().flatten())
        all_y_pred.append(pred.detach().cpu().numpy().flatten())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    r, _ = pearsonr(y_true, y_pred)
    return r


def resolve_lax_args(args):
    """Copy dataset-specific LAX args into generic names (mirrors run_training_and_evaluation)."""
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


def compute_x_mean_std(dataloader):
    """Compute feature mean/std from training data for model initialization."""
    sum_ = 0.0
    sum_sq = 0.0
    n_samples = 0
    x_dim = 0

    for batch in dataloader:
        if len(batch) == 6:
            x1, x2 = batch[0], batch[1]
        else:
            x1, x2 = batch[0], batch[1]
        x_dim = x1.shape[1] - 1
        x1f = x1[:, :x_dim]
        x2f = x2[:, :x_dim]
        sum_ += x2f.sum(dim=0)
        sum_sq += (x2f ** 2).sum(dim=0)
        n_samples += x2f.size(0)

    X_mean = sum_ / n_samples
    X_std = torch.sqrt((sum_sq / n_samples) - (X_mean ** 2))
    return X_mean, X_std, x_dim


def run_dataset(dataset_name, epoch_limit=None):
    """Train and evaluate LAX on a single dataset. Returns metrics dict."""
    cfg = DATASETS[dataset_name]
    results_dir = OUTPUTS_ROOT / dataset_name
    ckpt_path = os.path.join(str(results_dir), 'best_weights.pth')
    metrics_csv = os.path.join(str(results_dir), 'metrics.csv')

    print(f"\n{'='*60}")
    print(f"  Training LAX on {dataset_name} ({cfg['batch']})")
    print(f"{'='*60}")

    args = setup_args(dataset_name)
    data_path = cfg["data_path"]

    g_dim = detect_g_dim(data_path, dataset_name, cfg["batch"])
    setattr(args, cfg["g_dim_attr"], g_dim)

    print(f"  Data path:   {data_path}")
    print(f"  Batch:       {cfg['batch']}")
    print(f"  g_dim:       {g_dim}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Epochs:      {args.epoch_net}")
    print(f"  Results dir: {args.results_path}")

    need_training = not os.path.exists(ckpt_path)
    need_post_eval = not os.path.exists(metrics_csv)

    if need_training:
        dataloader = cfg["loader"](args, data_path=data_path)
        print(f"  Train: {len(dataloader['train'].dataset)} samples")
        print(f"  Valid: {len(dataloader['valid'].dataset)} samples")
        print(f"  Test:  {len(dataloader['test'].dataset)} samples")

        print(f"\n  Training ...")
        test_metrics = run_training_and_evaluation(
            args=args,
            dataloader=dataloader,
            batch_name=cfg["batch"],
            experiment_id=0,
        )
    else:
        print(f"  Checkpoint exists, skipping training.")
        test_metrics = {}
        tm_csv = os.path.join(str(results_dir), 'test_metrics.csv')
        if os.path.exists(tm_csv):
            df = pd.read_csv(tm_csv)
            for _, row in df.iterrows():
                test_metrics[row['Metric'].lower()] = row['Value']
            print(f"  Loaded existing test_metrics.csv")
        else:
            print(f"  No test_metrics.csv, will run test evaluation ...")
            need_training_eval = True
            resolve_lax_args(args)
            dataloader = cfg["loader"](args, data_path=data_path)
            X_mean, X_std, x_dim = compute_x_mean_std(dataloader['train'])
            model = OptimizationNetwork(
                x_sts=(X_mean, X_std),
                y_dim=x_dim,
                x_dim=x_dim,
                args=args,
            ).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state']
            if 'y' in state_dict and state_dict['y'].shape != model.y.shape:
                state_dict['y'] = model.y.data
            model.load_state_dict(state_dict)
            model.eval()
            test_metrics = evaluate_on_set(model, dataloader['test'], args,
                                           experiment_folder_path=str(results_dir),
                                           phase='test', test=True, make_csv=True)
            test_metrics['pearson_r'] = 0.0
            test_metrics['final_loss'] = checkpoint.get('val_loss', test_metrics.get('mse', 0))

    if need_post_eval:
        print(f"  Computing Pearson R + saving artifacts ...")
        resolve_lax_args(args)
        dataloader = cfg["loader"](args, data_path=data_path)
        X_mean, X_std, x_dim = compute_x_mean_std(dataloader['train'])
        model = OptimizationNetwork(
            x_sts=(X_mean, X_std),
            y_dim=x_dim,
            x_dim=x_dim,
            args=args,
        ).to(device)

        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state']
            if 'y' in state_dict and state_dict['y'].shape != model.y.shape:
                state_dict['y'] = model.y.data
            model.load_state_dict(state_dict)
            model.eval()

            pearson_r = compute_pearson_r(model, dataloader['test'], args)
            final_loss = checkpoint.get('val_loss', test_metrics.get('mse', 0))
        else:
            pearson_r = 0.0
            final_loss = test_metrics.get('mse', 0)
            print(f"  WARNING: No checkpoint found")

        test_metrics['pearson_r'] = pearson_r
        test_metrics['final_loss'] = final_loss
        test_metrics['dataset'] = dataset_name
        test_metrics['batch'] = cfg['batch']
        test_metrics['g_dim'] = g_dim

        metrics_df = pd.DataFrame([
            {"Metric": "MSE", "Value": test_metrics.get('mse', 0)},
            {"Metric": "RMSE", "Value": test_metrics.get('rmse', 0)},
            {"Metric": "MAPE", "Value": test_metrics.get('mape', 0)},
            {"Metric": "MAE", "Value": test_metrics.get('mae', 0)},
            {"Metric": "DUAL", "Value": test_metrics.get('dual', 0)},
            {"Metric": "PDE", "Value": test_metrics.get('pde', 0)},
            {"Metric": "Pearson_R", "Value": pearson_r},
            {"Metric": "Final_Val_Loss", "Value": final_loss},
        ])
        metrics_df.to_csv(metrics_csv, index=False)

        log_lines = [
            "=" * 60,
            f"LAX Training Run — {dataset_name}",
            "=" * 60,
            f"Dataset:       {args.data}",
            f"Batch:         {cfg['batch']}",
            f"g_dim:         {g_dim}",
            f"Epochs (net):  {args.epoch_net}",
            f"Batch size:    {args.batch_size}",
            f"Device:        {device}",
            f"Results path:  {args.results_path}",
            "",
            "--- Test Metrics ---",
        ]
        for key in ['mse', 'rmse', 'mape', 'mae', 'dual', 'pde', 'pearson_r', 'final_loss']:
            val = test_metrics.get(key, 0)
            log_lines.append(f"  {key.upper():>14s}: {val:.8f}")
        log_lines.append("=" * 60)
        with open(os.path.join(args.results_path, 'logging.txt'), 'w') as f:
            f.write('\n'.join(log_lines) + '\n')
    else:
        print(f"  metrics.csv already exists, skipping post-eval.")
        df = pd.read_csv(metrics_csv)
        for _, row in df.iterrows():
            k = row['Metric'].lower()
            test_metrics[k] = row['Value']
        test_metrics['dataset'] = dataset_name
        test_metrics['batch'] = cfg['batch']
        test_metrics['g_dim'] = g_dim

    print(f"\n  Results for {dataset_name}:")
    print(f"    MSE:      {test_metrics.get('mse', 0):.6f}")
    print(f"    RMSE:     {test_metrics.get('rmse', 0):.6f}")
    print(f"    MAPE:     {test_metrics.get('mape', 0):.6f}")
    print(f"    MAE:      {test_metrics.get('mae', 0):.6f}")
    print(f"    Pearson R: {test_metrics.get('pearson_r', 0):.6f}")
    print(f"    Final Loss: {test_metrics.get('final_loss', 0):.6f}")

    return test_metrics


def generate_comparison_table(all_results):
    """Create overall_dataset_comparison.csv and print terminal summary."""
    rows = []
    for r in all_results:
        rows.append({
            "Dataset": r["dataset"],
            "Batch": r["batch"],
            "g_dim": r["g_dim"],
            "MSE": r.get("mse", 0),
            "RMSE": r.get("rmse", 0),
            "MAPE": r.get("mape", 0),
            "MAE": r.get("mae", 0),
            "Pearson_R": r.get("pearson_r", 0),
            "Final_Loss": r.get("final_loss", r.get("final_val_loss", 0)),
            "DUAL": r.get("dual", 0),
            "PDE": r.get("pde", 0),
        })

    df = pd.DataFrame(rows)
    out_path = OUTPUTS_ROOT / "overall_dataset_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nComparison table saved to {out_path}")

    # Terminal summary
    print("\n" + "=" * 90)
    print("  OVERALL DATASET COMPARISON — LAX Model")
    print("=" * 90)
    header = f"{'Dataset':<10s} {'Batch':<12s} {'MSE':>10s} {'RMSE':>10s} {'MAPE':>10s} {'MAE':>10s} {'Pearson R':>10s} {'Final Loss':>12s}"
    print(header)
    print("-" * 90)
    for r in rows:
        print(f"{r['Dataset']:<10s} {r['Batch']:<12s} {r['MSE']:>10.6f} {r['RMSE']:>10.6f} {r['MAPE']:>10.6f} {r['MAE']:>10.6f} {r['Pearson_R']:>10.6f} {r['Final_Loss']:>12.6f}")
    print("=" * 90)


def main():
    print("=" * 60)
    print("  Unified Multi-Dataset LAX Training Runner")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Output: {OUTPUTS_ROOT}")

    all_results = []
    for dataset_name in DATASETS:
        try:
            result = run_dataset(dataset_name)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        generate_comparison_table(all_results)
    else:
        print("\nNo datasets completed successfully.")

    print("\nDone.")


if __name__ == "__main__":
    main()
