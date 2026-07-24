"""
Unified Plotting Script — Per-Battery Capacity Prediction
==========================================================
Generates:
  1. Per-battery capacity prediction curves (individual subplots)
  2. Per-battery scatter plots
  3. Combined 2x2 grid across all datasets
  4. Training/validation loss curves
  5. Cross-dataset comparison bar charts

Usage:
    python plot_all_datasets.py
"""
import sys
sys.argv = [sys.argv[0]]

import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

from utils.arguments import get_LAX_args
from dataloader.dataloader import XJTUdata, TJUdata, MITdata, HUSTdata
from Model.PI_nets.LAX import OptimizationNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUTS_ROOT = Path("outputs").resolve()
PLOTS_DIR = OUTPUTS_ROOT / "plots"

COLORS = {"XJTU": "#1f77b4", "TJU": "#ff7f0e", "MIT": "#2ca02c", "HUST": "#d62728"}

DATASETS = {
    "XJTU": {"data": "XJTU", "batch": "3C", "g_dim_attr": "g_dim_LAX_XJTU"},
    "TJU":  {"data": "TJU", "batch": "NCM_NCA", "g_dim_attr": "g_dim_LAX_TJU"},
    "MIT":  {"data": "MIT", "batch": "one_batch", "g_dim_attr": "g_dim_LAX_MIT"},
    "HUST": {"data": "HUST", "batch": "one_batch", "g_dim_attr": "g_dim_LAX_HUST"},
}


def resolve_lax_args(args):
    ds = args.data
    suffix = {"XJTU": "_XJTU", "TJU": "_TJU", "MIT": "_MIT"}.get(ds, "_HUST")
    mapping = {
        "betha_LAX": f"betha_LAX{suffix}",
        "dual_LAX": f"dual_LAX{suffix}",
        "theta_LAX": f"theta_LAX{suffix}",
        "lr_net": f"lr_net_LAX{suffix}",
        "h_dim_LAX": f"h_dim_LAX{suffix}",
        "beta_LAX": f"beta_LAX{suffix}",
        "distance_block_LAX": f"distance_block_LAX{suffix}",
        "time_block_LAX": f"time_block_LAX{suffix}",
        "center_block_LAX": f"center_block_LAX{suffix}",
        "inside_distance_block_MLP_layers": f"inside_distance_block_MLP_layers_LAX{suffix}",
        "inside_theta_layers": f"inside_theta_layers_LAX{suffix}",
        "inside_multivar_theta_layers": f"inside_multivar_theta_layers_LAX{suffix}",
        "inside_h_star_layers": f"inside_h_star_layers_LAX{suffix}",
        "inside_phi_layers": f"inside_phi_layers_LAX{suffix}",
        "inside_g_layers": f"inside_g_layers_LAX{suffix}",
        "inside_S_MLP_layers": f"inside_S_MLP_layers_LAX{suffix}",
        "inside_betan_layers": f"inside_betan_layers_LAX{suffix}",
        "g_dim": f"g_dim_LAX{suffix}",
        "h_out_LAX": f"H_out_LAX{suffix}",
        "g_out_LAX": f"g_out_LAX{suffix}",
        "phi_out_LAX": f"phi_out_LAX{suffix}",
        "dim_output_LAX": f"dim_output_LAX{suffix}",
        "lr_F": f"lr_F{suffix}",
    }
    for generic, specific in mapping.items():
        if hasattr(args, specific):
            setattr(args, generic, getattr(args, specific))


def detect_g_dim(data_path, dataset_name):
    cfg = DATASETS[dataset_name]
    root = os.path.join(data_path, cfg["data"] + " data")
    for dirpath, _, filenames in os.walk(root):
        csvs = sorted([f for f in filenames if f.endswith(".csv")])
        if csvs:
            df = pd.read_csv(os.path.join(dirpath, csvs[0]))
            return df.shape[1] - 1
    return 16


def get_test_file_list(dataset_name):
    """Get list of test battery CSV file paths for each dataset."""
    cfg = DATASETS[dataset_name]
    root = "data/Processed/" + cfg["data"] + " data"

    if dataset_name == "XJTU":
        files = sorted([f for f in os.listdir(root) if f.endswith('.csv') and cfg["batch"] in f])
        return [os.path.join(root, f) for f in files if '4' in f or '8' in f]

    elif dataset_name == "TJU":
        batch_root = os.path.join(root, "Dataset_3_NCM_NCA_battery")
        files = sorted(os.listdir(batch_root))
        test_list = []
        for i, f in enumerate(files):
            if (i + 1) % 10 == 5 or (i + 1) % 10 == 9:
                test_list.append(os.path.join(batch_root, f))
        return test_list

    elif dataset_name == "MIT":
        test_list = []
        for batch in ['2017-05-12', '2017-06-30', '2018-04-12']:
            batch_root = os.path.join(root, batch)
            if not os.path.isdir(batch_root):
                continue
            for f in sorted(os.listdir(batch_root)):
                if not f.endswith('.csv'):
                    continue
                fid = int(f.split('-')[-1].split('.')[0])
                if fid % 5 == 0:
                    test_list.append(os.path.join(batch_root, f))
        return test_list

    elif dataset_name == "HUST":
        test_id = ['1-4','1-8','2-4','2-8','3-4','3-8','4-4','4-8',
                   '5-4','5-7','6-4','6-8','7-4','7-8','8-4','8-8',
                   '9-4','9-8','10-4','10-8']
        files = sorted([f for f in os.listdir(root) if f.endswith('.csv')])
        return [os.path.join(root, f) for f in files if f[:-4] in test_id]

    return []


def load_args(dataset_name):
    cfg = DATASETS[dataset_name]
    args = get_LAX_args()
    args.data = cfg["data"]
    args.batch = cfg["batch"]
    args.run_mode = "LAX"
    args.run_optuna = False
    args.run_samll_sample = False

    g_dim = detect_g_dim("data/Processed", dataset_name)
    setattr(args, cfg["g_dim_attr"], g_dim)
    resolve_lax_args(args)
    return args


def load_model(dataset_name, args):
    g_dim = detect_g_dim("data/Processed", dataset_name)
    x_dim = g_dim

    sum_ = 0.0
    sum_sq = 0.0
    n_samples = 0

    test_files = get_test_file_list(dataset_name)
    root = "data/Processed/" + DATASETS[dataset_name]["data"] + " data"

    data_cls_map = {"XJTU": XJTUdata, "TJU": TJUdata, "MIT": MITdata, "HUST": HUSTdata}
    data_obj = data_cls_map[dataset_name](root=root, args=args)

    for path in test_files:
        df = data_obj.read_one_csv(path)
        x = df.iloc[:, :-1].values
        x2 = x[1:, :x_dim]
        n_samples += x2.shape[0]
        sum_ += x2.sum(axis=0)
        sum_sq += (x2 ** 2).sum(axis=0)

    if n_samples == 0:
        raise ValueError(f"No test samples found for {dataset_name}")

    X_mean = sum_ / n_samples
    X_std = np.sqrt((sum_sq / n_samples) - (X_mean ** 2))
    X_mean_t = torch.from_numpy(X_mean).float()
    X_std_t = torch.from_numpy(X_std).float()

    model = OptimizationNetwork(x_sts=(X_mean_t, X_std_t), y_dim=x_dim, x_dim=x_dim, args=args).to(device)
    ckpt_path = OUTPUTS_ROOT / dataset_name / "best_weights.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"]
    if "y" in state_dict and state_dict["y"].shape != model.y.shape:
        state_dict["y"] = model.y.data
    model.load_state_dict(state_dict)
    model.eval()
    return model, data_obj


def predict_per_battery(model, data_obj, file_paths, args):
    """Load each battery CSV individually and predict."""
    battery_results = []
    batch_list = getattr(data_obj, 'batchs', None)
    nom_caps = getattr(data_obj, 'nominal_capacities', None)
    default_nc = getattr(data_obj, 'nominal_capacity', None)

    def _nc_for_path(path):
        if nom_caps and batch_list:
            for i, b in enumerate(batch_list):
                if b in path:
                    return nom_caps[i]
        return default_nc

    for path in file_paths:
        battery_name = Path(path).stem
        df = data_obj.read_one_csv(path, nominal_capacity=_nc_for_path(path))
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        x_tensor = torch.from_numpy(x).float().to(device)
        x_in = x_tensor[:, :-1]
        t_in = x_in[:, -1]

        with torch.no_grad():
            pred, _ = model(x=x_in, t=t_in.squeeze(), epoch=args.epoch_net, return_f=True)

        y_true = y
        y_pred = pred.detach().cpu().numpy().flatten()
        battery_results.append({"name": battery_name, "y_true": y_true, "y_pred": y_pred})
    return battery_results


# ============================================================================
# Plot: Per-Battery Capacity Prediction (grid of subplots)
# ============================================================================
def plot_per_battery_capacity(dataset_name, battery_results):
    n = len(battery_results)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"{dataset_name} ({DATASETS[dataset_name]['batch']}) — Per-Battery Capacity Prediction",
                 fontsize=16, fontweight="bold", y=1.02)

    for idx, br in enumerate(battery_results):
        r, _ = pearsonr(br["y_true"], br["y_pred"])
        mae = np.mean(np.abs(br["y_true"] - br["y_pred"]))
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.plot(br["y_true"], linewidth=1.5, color=COLORS[dataset_name], label="True")
        ax.plot(br["y_pred"], linewidth=1.5, color="red", linestyle="--", alpha=0.8, label="Predicted")
        ax.set_title(f"{br['name']}\nR={r:.4f} MAE={mae:.4f}", fontsize=9)
        ax.set_xlabel("Cycle", fontsize=8)
        ax.set_ylabel("Capacity", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"capacity_prediction_per_battery_{dataset_name}.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved capacity_prediction_per_battery_{dataset_name}.png ({n} batteries)")


# ============================================================================
# Plot: Per-Battery Scatter
# ============================================================================
def plot_per_battery_scatter(dataset_name, battery_results):
    n = len(battery_results)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    fig.suptitle(f"{dataset_name} — Per-Battery Predicted vs Actual",
                 fontsize=16, fontweight="bold", y=1.02)

    for idx, br in enumerate(battery_results):
        r, _ = pearsonr(br["y_true"], br["y_pred"])
        mae = np.mean(np.abs(br["y_true"] - br["y_pred"]))
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.scatter(br["y_true"], br["y_pred"], s=4, alpha=0.4, color=COLORS[dataset_name])
        lims = [min(br["y_true"].min(), br["y_pred"].min()) - 0.01,
                max(br["y_true"].max(), br["y_pred"].max()) + 0.01]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=0.8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"{br['name']}\nR={r:.4f} MAE={mae:.4f}", fontsize=9)
        ax.set_xlabel("Actual", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"scatter_per_battery_{dataset_name}.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter_per_battery_{dataset_name}.png ({n} batteries)")


# ============================================================================
# Plot: Combined 2x2 overview (best battery from each dataset)
# ============================================================================
def plot_combined_overview(all_battery_results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("LAX Model — Capacity Prediction (One Best-Performing Battery per Dataset)",
                 fontsize=16, fontweight="bold")

    for ax, (ds, batteries) in zip(axes.flatten(), all_battery_results.items()):
        if not batteries:
            ax.text(0.5, 0.5, f"No data\nfor {ds}", ha="center", va="center", fontsize=14)
            continue

        best = max(batteries, key=lambda b: pearsonr(b["y_true"], b["y_pred"])[0])
        r, _ = pearsonr(best["y_true"], best["y_pred"])
        mae = np.mean(np.abs(best["y_true"] - best["y_pred"]))
        ax.plot(best["y_true"], label="True", linewidth=1.5, color=COLORS[ds])
        ax.plot(best["y_pred"], label="Predicted", linewidth=1.5, color="red", linestyle="--", alpha=0.8)
        ax.set_title(f"{ds} ({best['name']})\nR={r:.4f}  MAE={mae:.4f}", fontsize=11)
        ax.set_xlabel("Cycle Index")
        ax.set_ylabel("Normalized Capacity")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "capacity_prediction_all_datasets.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved capacity_prediction_all_datasets.png")


# ============================================================================
# Plot: Combined scatter (all batteries overlaid per dataset)
# ============================================================================
def plot_combined_scatter(all_battery_results):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Predicted vs Actual — All Test Batteries Overlaid", fontsize=16, fontweight="bold")

    for ax, (ds, batteries) in zip(axes, all_battery_results.items()):
        all_true = np.concatenate([b["y_true"] for b in batteries])
        all_pred = np.concatenate([b["y_pred"] for b in batteries])
        r, _ = pearsonr(all_true, all_pred)
        mae = np.mean(np.abs(all_true - all_pred))
        ax.scatter(all_true, all_pred, s=2, alpha=0.2, color=COLORS[ds])
        lims = [min(all_true.min(), all_pred.min()) - 0.01,
                max(all_true.max(), all_pred.max()) + 0.01]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"{ds} ({len(batteries)} batteries)\nR={r:.4f} | MAE={mae:.4f}", fontsize=11)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "predicted_vs_actual_all.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved predicted_vs_actual_all.png")


# ============================================================================
# Plot: Training Loss Curves
# ============================================================================
def parse_training_metrics(dataset_name):
    cfg = DATASETS[dataset_name]
    exp_dir = OUTPUTS_ROOT / dataset_name / f"{cfg['batch']}-Experiments"
    tm_path = exp_dir / "training_metrics.txt"
    if not tm_path.exists():
        return None

    records = []
    pattern = re.compile(
        r"^\s*(train|valid)\s+loss\s+(\w+(?:\s+\w+)?)\s+on\s+epoch\s+(\d+)\s*:\s+([\d.e+-]+)\s*$",
        re.IGNORECASE)
    with open(tm_path) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                records.append({
                    "phase": m.group(1), "metric": m.group(2),
                    "epoch": int(m.group(3)), "value": float(m.group(4))
                })
    return pd.DataFrame(records)


def plot_training_loss_individual(dataset_name, df):
    if df is None:
        return
    metrics = ["mse", "mae", "mape"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{dataset_name} — Training & Validation Loss Curves", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        train = df[(df["phase"] == "train") & (df["metric"] == metric)]
        valid = df[(df["phase"] == "valid") & (df["metric"] == metric)]
        if not train.empty:
            ax.plot(train["epoch"], train["value"], label="Train", linewidth=1.2, color=COLORS[dataset_name])
        if not valid.empty:
            ax.plot(valid["epoch"], valid["value"], label="Valid", linewidth=1.2,
                    color=COLORS[dataset_name], linestyle="--", alpha=0.7)
        ax.set_title(metric.upper(), fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"training_loss_{dataset_name}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved training_loss_{dataset_name}.png")


# ============================================================================
# Plot: Cross-Dataset Comparison Bar Charts
# ============================================================================
def plot_comparison_bars(comparison_path):
    df = pd.read_csv(comparison_path)
    if df.empty:
        return

    bar_metrics = [("MSE", "mse"), ("RMSE", "rmse"), ("MAE", "mae"), ("MAPE", "mape")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Cross-Dataset Metric Comparison (LAX Model)", fontsize=16, fontweight="bold")

    for ax, (label, col) in zip(axes, bar_metrics):
        vals = df[col].values
        ds_names = df["Dataset"].values
        bar_colors = [COLORS.get(d, "#333") for d in ds_names]
        bars = ax.bar(ds_names, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_title(label, fontsize=13)
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comparison_bar_chart.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison_bar_chart.png")

    pearson_vals = df["pearson_r"].values
    ds_names = df["Dataset"].values
    bar_colors = [COLORS.get(d, "#333") for d in ds_names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(ds_names, pearson_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, pearson_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.title("Pearson R Comparison (LAX Model)", fontsize=14, fontweight="bold")
    plt.ylabel("Pearson R")
    plt.ylim(0.85, 1.0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pearson_r_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved pearson_r_comparison.png")


# ============================================================================
# Main
# ============================================================================
def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Generating Per-Battery Plots for All 4 Datasets")
    print("=" * 60)

    all_battery_results = {}

    for ds_name in DATASETS:
        print(f"\n  Processing {ds_name} ...")
        try:
            args = load_args(ds_name)
            model, data_obj = load_model(ds_name, args)
            test_files = get_test_file_list(ds_name)
            print(f"    Found {len(test_files)} test batteries")

            battery_results = predict_per_battery(model, data_obj, test_files, args)
            all_battery_results[ds_name] = battery_results

            plot_per_battery_capacity(ds_name, battery_results)
            plot_per_battery_scatter(ds_name, battery_results)

            df = parse_training_metrics(ds_name)
            plot_training_loss_individual(ds_name, df)

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR on {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_battery_results:
        print(f"\n  Generating combined plots ...")
        plot_combined_overview(all_battery_results)
        plot_combined_scatter(all_battery_results)

    comparison_csv = OUTPUTS_ROOT / "overall_dataset_comparison.csv"
    if comparison_csv.exists():
        plot_comparison_bars(comparison_csv)

    print(f"\n  All plots saved to: {PLOTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
