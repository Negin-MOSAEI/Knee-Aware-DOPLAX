"""
Unified Plotting Script — Knee-Aware DOPLAX Model Prediction
=============================================================
Loads the full ensemble (extractor + solution_u + LAX + bagging_NN)
from outputs/{dataset}/Bagging/best_model.pth and generates:
  1. Per-battery capacity prediction (True vs DeepOPINN vs LAX vs Knee-Aware DOPLAX)
  2. Per-battery scatter (predicted vs actual for each sub-model)
  3. Combined 2x2 grid across all datasets
  4. Combined scatter all batteries overlaid
  5. Training/validation loss curves
  6. Cross-dataset comparison bar charts (all 3 sub-models)
  7. Metrics summary table (CSV)

Usage:
    python plot_all_datasets.py
"""
import sys
sys.argv = [sys.argv[0]]

import os
os.environ['DDE_BACKEND'] = 'pytorch'

import re
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

import deepxde as dde
from utils.arguments import get_Bagging_u_args
from dataloader.dataloader import XJTUdata, TJUdata, MITdata, HUSTdata
from Model.PI_nets.LAX import OptimizationNetwork
from Model.Auxiliary_nets.Solution_u import Solution_u
from Model.Auxiliary_nets.MLP import MLP
from Model.Combination_nets.Bagging_u import MLP_Bagging_NN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUTS_ROOT = Path("outputs").resolve()
PLOTS_DIR = OUTPUTS_ROOT / "plots"

COLORS = {"XJTU": "#1f77b4", "TJU": "#ff7f0e", "MIT": "#2ca02c", "HUST": "#d62728"}
MODEL_COLORS = {"DeepOPINN": "#1f77b4", "LAX": "#ff7f0e", "Knee-Aware DOPLAX": "#d62728"}
MODEL_STYLES = {"DeepOPINN": (":", 1.2), "LAX": ("--", 1.2), "Knee-Aware DOPLAX": ("-", 2.0)}

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
    cfg = DATASETS[dataset_name]
    root = "data/Processed/" + cfg["data"] + " data"

    if dataset_name == "XJTU":
        files = sorted([f for f in os.listdir(root) if f.endswith('.csv') and (cfg["batch"] == 'All' or cfg["batch"] in f)])
        return [os.path.join(root, f) for f in files if '4' in f or '8' in f]

    elif dataset_name == "TJU":
        folders = ["Dataset_1_NCA_battery", "Dataset_2_NCM_battery", "Dataset_3_NCM_NCA_battery"] if cfg["batch"] == 'All' else ["Dataset_3_NCM_NCA_battery"]
        test_list = []
        for folder in folders:
            batch_root = os.path.join(root, folder)
            files = sorted(os.listdir(batch_root))
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


def load_bagging_args(dataset_name):
    cfg = DATASETS[dataset_name]
    args = get_Bagging_u_args()
    args.data = cfg["data"]
    args.batch = cfg["batch"]
    args.batch_size = getattr(args, f"batch_size_{dataset_name}")
    args.save_folder = str(OUTPUTS_ROOT / dataset_name / "Bagging")
    args.log_dir = "logging.txt"
    args.run_mode = "Bagging_u"
    args.run_optuna = False
    args.run_samll_sample = False
    args.run_for_DeepOPINN = False
    args.run_for_LAX = False
    args.trained_by_y_opt_LAX = False
    args.dim_x = 1

    ds = dataset_name
    args.F_hidden_dim = getattr(args, f"F_hidden_dim_{ds}")
    args.F_layers_num = getattr(args, f"F_layers_num_{ds}")
    args.dropout = getattr(args, f"dropout_{ds}")
    args.bag_hidden_dim = getattr(args, f"bag_hidden_dim_{ds}")
    if ds == "HUST":
        args.bag_hidden_dim = [100, 100]

    g_dim = detect_g_dim("data/Processed", dataset_name)
    args.g_dim_LAX_XJTU = g_dim
    args.g_dim_LAX_TJU = g_dim
    args.g_dim_LAX_MIT = g_dim
    args.g_dim_LAX_HUST = g_dim

    args.distance_block_LAX_XJTU = "MLP"
    args.distance_block_LAX_TJU = "MLP"
    args.distance_block_LAX_MIT = "MLP"
    args.distance_block_LAX_HUST = "MLP"
    args.h_dim_LAX_MIT = 30

    resolve_lax_args(args)
    return args


def load_bagging_model(dataset_name, args):
    ckpt_path = OUTPUTS_ROOT / dataset_name / "Bagging" / "best_model.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    m = checkpoint['input_dimention']

    g_dim = detect_g_dim("data/Processed", dataset_name)
    x_dim = g_dim

    data_cls_map = {"XJTU": XJTUdata, "TJU": TJUdata, "MIT": MITdata, "HUST": HUSTdata}
    root = "data/Processed/" + DATASETS[dataset_name]["data"] + " data"
    data_obj = data_cls_map[dataset_name](root=root, args=args)

    sum_ = 0.0
    sum_sq = 0.0
    n_samples = 0
    test_files = get_test_file_list(dataset_name)
    for path in test_files:
        df = data_obj.read_one_csv(path)
        x = df.iloc[:, :-1].values
        x2 = x[1:, :x_dim]
        n_samples += x2.shape[0]
        sum_ += x2.sum(axis=0)
        sum_sq += (x2 ** 2).sum(axis=0)

    X_mean = sum_ / n_samples
    X_std = np.sqrt((sum_sq / n_samples) - (X_mean ** 2))
    X_mean_t = torch.from_numpy(X_mean).float()
    X_std_t = torch.from_numpy(X_std).float()

    layer_sizes_branch = [m] + [args.F_hidden_dim] * (args.F_layers_num - 1)
    layer_sizes_trunk = [args.dim_x] + [args.F_hidden_dim] * (args.F_layers_num - 1)
    extractor = dde.nn.DeepONetCartesianProd(
        layer_sizes_branch=layer_sizes_branch,
        layer_sizes_trunk=layer_sizes_trunk,
        activation="relu",
        kernel_initializer="Glorot normal"
    ).to(device)
    extractor.load_state_dict(checkpoint['feature_extractor'], strict=False)
    extractor.eval()

    solution_u = Solution_u(
        input_dim=m,
        layers_num=args.F_layers_num,
        hidden_dim=60,
        dropout=args.dropout
    ).to(device)
    solution_u.load_state_dict(checkpoint['solution_u'])
    solution_u.eval()

    lax_model = OptimizationNetwork(
        x_sts=(X_mean_t, X_std_t),
        y_dim=x_dim,
        x_dim=x_dim,
        args=args
    ).to(device)
    lax_state = dict(checkpoint['LAX'])
    if "y" in lax_state and lax_state["y"].shape != lax_model.y.shape:
        lax_state["y"] = lax_model.y.data
    lax_model.load_state_dict(lax_state, strict=False)
    lax_model.eval()

    bagging_NN = MLP_Bagging_NN(
        input_dim=3,
        output_dim=1,
        hidden_dim=args.bag_hidden_dim,
        dropout=args.dropout
    ).to(device)
    bagging_NN.load_state_dict(checkpoint['bagging_u'])
    bagging_NN.eval()

    return extractor, solution_u, lax_model, bagging_NN, m, data_obj


def kneaware_inference(extractor, solution_u, lax_model, bagging_NN, m, x_tensor, kd_tensor, epoch=0):
    """Run full Knee-Aware DOPLAX inference, returning all 3 sub-model predictions."""
    with torch.no_grad():
        B, L = x_tensor.shape
        metadata = x_tensor[:, -1].unsqueeze(1)
        coordinates = torch.linspace(0, 1, m - 1).reshape(-1, 1).to(device)
        features = extractor((x_tensor, coordinates))
        xt = torch.cat([features, metadata], dim=1)

        _, u_pinn = solution_u(xt)

        u_lax = lax_model(x=x_tensor[:, :-1], t=x_tensor[:, -1], epoch=epoch)
        if u_lax.dim() == 1:
            u_lax = u_lax.unsqueeze(1)

        pred = bagging_NN(torch.cat([u_pinn, u_lax, kd_tensor], dim=1))

    return (u_pinn.detach().cpu().numpy().flatten(),
            u_lax.detach().cpu().numpy().flatten(),
            pred.detach().cpu().numpy().flatten())


def predict_per_battery(extractor, solution_u, lax_model, bagging_NN, m, data_obj, file_paths, args):
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
        y = df.iloc[:, -1].values

        feat_cols = [c for c in df.columns if c not in ['capacity', 'knee_point_distance']]
        x = df[feat_cols].values
        x_tensor = torch.from_numpy(x).float().to(device)

        if 'knee_point_distance' in df.columns:
            kd = df['knee_point_distance'].values
            kd_tensor = torch.from_numpy(kd).float().view(-1, 1).to(device)
        else:
            kd_tensor = torch.zeros(x_tensor.shape[0], 1, device=device)

        u_pinn, u_lax, y_pred = kneaware_inference(
            extractor, solution_u, lax_model, bagging_NN, m, x_tensor, kd_tensor, epoch=0
        )
        battery_results.append({
            "name": battery_name,
            "y_true": y,
            "y_pinn": u_pinn,
            "y_lax": u_lax,
            "y_pred": y_pred,
        })
    return battery_results


# ============================================================================
# Metrics helper
# ============================================================================
def compute_metrics(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return {"Pearson_R": r, "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


# ============================================================================
# Plot: Per-Battery Capacity Prediction — all 3 models overlaid
# ============================================================================
def plot_per_battery_capacity(dataset_name, battery_results):
    n = len(battery_results)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    fig.suptitle(f"{dataset_name} ({DATASETS[dataset_name]['batch']}) — Knee-Aware DOPLAX Capacity Prediction",
                 fontsize=16, fontweight="bold", y=1.02)

    for idx, br in enumerate(battery_results):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.plot(br["y_true"], linewidth=1.8, color="black", label="True", zorder=5)
        ax.plot(br["y_pinn"], linewidth=1.0, color=MODEL_COLORS["DeepOPINN"],
                linestyle=MODEL_STYLES["DeepOPINN"][0], alpha=0.7, label="DeepOPINN")
        ax.plot(br["y_lax"], linewidth=1.0, color=MODEL_COLORS["LAX"],
                linestyle=MODEL_STYLES["LAX"][0], alpha=0.7, label="LAX")
        ax.plot(br["y_pred"], linewidth=1.5, color=MODEL_COLORS["Knee-Aware DOPLAX"],
                linestyle="-", alpha=0.9, label="Knee-Aware DOPLAX")

        m_deep = compute_metrics(br["y_true"], br["y_pinn"])
        m_lax = compute_metrics(br["y_true"], br["y_lax"])
        m_kad = compute_metrics(br["y_true"], br["y_pred"])
        ax.set_title(f"{br['name']}\n"
                     f"DOPINN R={m_deep['Pearson_R']:.3f} | "
                     f"LAX R={m_lax['Pearson_R']:.3f} | "
                     f"KAD R={m_kad['Pearson_R']:.3f}", fontsize=8)
        ax.set_xlabel("Cycle", fontsize=8)
        ax.set_ylabel("Capacity", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"kneaware_capacity_per_battery_{dataset_name}.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved kneaware_capacity_per_battery_{dataset_name}.png ({n} batteries)")


# ============================================================================
# Plot: Per-Battery Scatter — 3-column grid (DeepOPINN, LAX, Knee-Aware DOPLAX)
# ============================================================================
def plot_per_battery_scatter(dataset_name, battery_results):
    n = len(battery_results)
    if n == 0:
        return

    model_keys = [("DeepOPINN", "y_pinn"), ("LAX", "y_lax"), ("Knee-Aware DOPLAX", "y_pred")]
    fig, axes = plt.subplots(n, 3, figsize=(18, 3.5 * n), squeeze=False)
    fig.suptitle(f"{dataset_name} — Predicted vs Actual (All Sub-Models)",
                 fontsize=16, fontweight="bold", y=1.005)

    for idx, br in enumerate(battery_results):
        for col, (model_name, key) in enumerate(model_keys):
            y_true = br["y_true"]
            y_pred = br[key]
            m = compute_metrics(y_true, y_pred)
            ax = axes[idx][col]
            ax.scatter(y_true, y_pred, s=4, alpha=0.4, color=MODEL_COLORS[model_name])
            lims = [min(y_true.min(), y_pred.min()) - 0.01,
                    max(y_true.max(), y_pred.max()) + 0.01]
            ax.plot(lims, lims, "k--", alpha=0.5, linewidth=0.8)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            if idx == 0:
                ax.set_title(model_name, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{br['name']}\nR={m['Pearson_R']:.3f}\nRMSE={m['RMSE']:.4f}",
                              fontsize=7)
            ax.set_xlabel("Actual" if idx == n - 1 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"kneaware_scatter_per_battery_{dataset_name}.png",
                dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved kneaware_scatter_per_battery_{dataset_name}.png ({n} batteries)")


# ============================================================================
# Plot: Combined 2x2 overview (best battery from each dataset, all 3 models)
# ============================================================================
def plot_combined_overview(all_battery_results):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Knee-Aware DOPLAX — Capacity Prediction (Best Battery per Dataset)",
                 fontsize=16, fontweight="bold")

    for ax, (ds, batteries) in zip(axes.flatten(), all_battery_results.items()):
        if not batteries:
            ax.text(0.5, 0.5, f"No data\nfor {ds}", ha="center", va="center", fontsize=14)
            continue

        best = max(batteries, key=lambda b: compute_metrics(b["y_true"], b["y_pred"])["Pearson_R"])
        m_deep = compute_metrics(best["y_true"], best["y_pinn"])
        m_lax = compute_metrics(best["y_true"], best["y_lax"])
        m_kad = compute_metrics(best["y_true"], best["y_pred"])

        ax.plot(best["y_true"], label="True", linewidth=1.8, color="black")
        ax.plot(best["y_pinn"], label=f"DeepOPINN (R={m_deep['Pearson_R']:.3f})",
                linewidth=1.0, color=MODEL_COLORS["DeepOPINN"], linestyle=":", alpha=0.7)
        ax.plot(best["y_lax"], label=f"LAX (R={m_lax['Pearson_R']:.3f})",
                linewidth=1.0, color=MODEL_COLORS["LAX"], linestyle="--", alpha=0.7)
        ax.plot(best["y_pred"], label=f"Knee-Aware DOPLAX (R={m_kad['Pearson_R']:.3f})",
                linewidth=1.5, color=MODEL_COLORS["Knee-Aware DOPLAX"], alpha=0.9)
        ax.set_title(f"{ds} ({best['name']})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Cycle Index")
        ax.set_ylabel("Normalized Capacity")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kneaware_combined_overview.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved kneaware_combined_overview.png")


# ============================================================================
# Plot: Combined scatter — 3 rows (one per model) x 4 cols (one per dataset)
# ============================================================================
def plot_combined_scatter(all_battery_results):
    model_keys = [("DeepOPINN", "y_pinn"), ("LAX", "y_lax"), ("Knee-Aware DOPLAX", "y_pred")]
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle("Knee-Aware DOPLAX — Predicted vs Actual — All Test Batteries",
                 fontsize=16, fontweight="bold")

    for row, (model_name, key) in enumerate(model_keys):
        for col, (ds, batteries) in enumerate(all_battery_results.items()):
            ax = axes[row][col]
            if not batteries:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue
            all_true = np.concatenate([b["y_true"] for b in batteries])
            all_pred = np.concatenate([b[key] for b in batteries])
            m = compute_metrics(all_true, all_pred)
            ax.scatter(all_true, all_pred, s=2, alpha=0.2, color=MODEL_COLORS[model_name])
            lims = [min(all_true.min(), all_pred.min()) - 0.01,
                    max(all_true.max(), all_pred.max()) + 0.01]
            ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            if row == 0:
                ax.set_title(f"{ds} ({len(batteries)} batteries)", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{model_name}\nR={m['Pearson_R']:.4f}\nRMSE={m['RMSE']:.4f}",
                              fontsize=9, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Actual", fontsize=10)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kneaware_predicted_vs_actual_all.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved kneaware_predicted_vs_actual_all.png")


# ============================================================================
# Plot: Per-Dataset Comparison Bar Chart (RMSE, MAE, MAPE, R for all 3 models)
# ============================================================================
def plot_model_comparison_bars(all_battery_results):
    models = ["DeepOPINN", "LAX", "Knee-Aware DOPLAX"]
    keys = ["y_pinn", "y_lax", "y_pred"]
    metric_names = ["RMSE", "MAE", "MAPE", "Pearson_R"]

    rows = []
    for ds, batteries in all_battery_results.items():
        if not batteries:
            continue
        for model_name, key in zip(models, keys):
            all_true = np.concatenate([b["y_true"] for b in batteries])
            all_pred = np.concatenate([b[key] for b in batteries])
            m = compute_metrics(all_true, all_pred)
            m["Dataset"] = ds
            m["Model"] = model_name
            rows.append(m)

    if not rows:
        return

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Knee-Aware DOPLAX — Cross-Dataset Model Comparison",
                 fontsize=16, fontweight="bold")

    bar_width = 0.25
    for ax, metric in zip(axes, metric_names):
        ds_names = sorted(all_battery_results.keys())
        x = np.arange(len(ds_names))
        for i, (model_name, color) in enumerate(MODEL_COLORS.items()):
            vals = []
            for ds in ds_names:
                row = df[(df["Dataset"] == ds) & (df["Model"] == model_name)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            bars = ax.bar(x + i * bar_width, vals, bar_width, label=model_name,
                         color=color, edgecolor="black", linewidth=0.5, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.4f}" if metric != "MAPE" else f"{v:.2f}%",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(ds_names, fontsize=10)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kneaware_model_comparison_bars.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved kneaware_model_comparison_bars.png")

    return df


# ============================================================================
# Plot: Metrics Summary Table (saved as CSV and as a figure)
# ============================================================================
def save_metrics_table(all_battery_results):
    models = ["DeepOPINN", "LAX", "Knee-Aware DOPLAX"]
    keys = ["y_pinn", "y_lax", "y_pred"]
    metric_names = ["RMSE", "MAE", "MAPE", "Pearson_R"]

    rows = []
    for ds, batteries in all_battery_results.items():
        if not batteries:
            continue
        for model_name, key in zip(models, keys):
            all_true = np.concatenate([b["y_true"] for b in batteries])
            all_pred = np.concatenate([b[key] for b in batteries])
            m = compute_metrics(all_true, all_pred)
            rows.append({
                "Dataset": ds,
                "Model": model_name,
                "N_batteries": len(batteries),
                "N_samples": len(all_true),
                **{k: round(v, 6) for k, v in m.items()},
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    csv_path = PLOTS_DIR / "kneaware_metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path.name}")

    fig, ax = plt.subplots(figsize=(18, max(3, len(rows) * 0.5 + 2)))
    ax.axis("off")
    ax.set_title("Knee-Aware DOPLAX — Metrics Summary", fontsize=14, fontweight="bold", pad=20)

    table_data = []
    for r in rows:
        table_data.append([
            r["Dataset"], r["Model"], str(r["N_batteries"]),
            f"{r['RMSE']:.4f}", f"{r['MAE']:.4f}",
            f"{r['MAPE']:.2f}%", f"{r['Pearson_R']:.4f}"
        ])

    col_labels = ["Dataset", "Model", "#Batt", "RMSE", "MAE", "MAPE", "Pearson R"]
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_idx > 0:
            model_name = rows[row_idx - 1]["Model"] if row_idx - 1 < len(rows) else ""
            if "Knee-Aware" in model_name:
                cell.set_facecolor("#FFF2CC")
            elif "LAX" in model_name:
                cell.set_facecolor("#FCE4D6")
            elif "DeepOPINN" in model_name:
                cell.set_facecolor("#D6E4F0")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kneaware_metrics_table.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved kneaware_metrics_table.png")


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
# Main
# ============================================================================
def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Knee-Aware DOPLAX — Generating Plots & Metrics")
    print("=" * 60)

    all_battery_results = {}

    for ds_name in DATASETS:
        print(f"\n  Processing {ds_name} ...")
        try:
            args = load_bagging_args(ds_name)
            extractor, solution_u, lax_model, bagging_NN, m, data_obj = load_bagging_model(ds_name, args)
            test_files = get_test_file_list(ds_name)
            print(f"    Found {len(test_files)} test batteries")

            battery_results = predict_per_battery(
                extractor, solution_u, lax_model, bagging_NN, m, data_obj, test_files, args
            )
            all_battery_results[ds_name] = battery_results

            plot_per_battery_capacity(ds_name, battery_results)
            plot_per_battery_scatter(ds_name, battery_results)

            df = parse_training_metrics(ds_name)
            plot_training_loss_individual(ds_name, df)

            del extractor, solution_u, lax_model, bagging_NN
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR on {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_battery_results:
        print(f"\n  Generating combined plots ...")
        plot_combined_overview(all_battery_results)
        plot_combined_scatter(all_battery_results)
        plot_model_comparison_bars(all_battery_results)
        save_metrics_table(all_battery_results)

    print(f"\n  All plots saved to: {PLOTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
