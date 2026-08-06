"""
Unified Per-Batch Post-Processing + Plotting + Metrics Pipeline
================================================================
Loads trained Bagging models for all datasets, evaluates per-batch,
applies enhanced post-processing (polyorder=4 SavGol + Hampel + IsotonicRegression),
generates per-batch plots and metrics, plus cross-batch comparison.

Usage:
    python run_post_processing.py
"""
import sys
sys.argv = [sys.argv[0]]

import os
os.environ['DDE_BACKEND'] = 'pytorch'

import re
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

import deepxde as dde
from utils.arguments import get_Bagging_u_args
from dataloader.dataloader import XJTUdata, TJUdata, MITdata, HUSTdata
from Model.PI_nets.LAX import OptimizationNetwork
from Model.Auxiliary_nets.Solution_u import Solution_u
from Model.Combination_nets.Bagging_u import MLP_Bagging_NN
from post_processing import postprocess_capacity, _mape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUTS_ROOT = Path("outputs").resolve()
PLOTS_DIR = OUTPUTS_ROOT / "plots"

MODEL_CLRS = {"DeepOPINN": "#1f77b4", "LAX": "#ff7f0e", "Knee-Aware DOPLAX": "#d62728", "Post-Processed": "#2ca02c"}
COLORS = {"XJTU": "#1f77b4", "TJU": "#ff7f0e", "MIT": "#2ca02c", "HUST": "#d62728"}

BATCH_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# ============================================================================
# Per-dataset batch definitions
# ============================================================================
TJU_DIR_MAP = {
    "NCA": "Dataset_1_NCA_battery",
    "NCM": "Dataset_2_NCM_battery",
    "NCM_NCA": "Dataset_3_NCM_NCA_battery",
}

def get_batches(dataset_name):
    if dataset_name == "XJTU":
        return ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]
    elif dataset_name == "TJU":
        return ["NCA", "NCM", "NCM_NCA"]
    elif dataset_name == "MIT":
        return ["2017-05-12", "2017-06-30", "2018-04-12"]
    elif dataset_name == "HUST":
        return ["one_batch"]
    return []

def get_batch_name(dataset_name, batch):
    if dataset_name in ("TJU", "MIT", "XJTU", "HUST"):
        return batch if dataset_name != "HUST" else "HUST"
    return batch

# ============================================================================
# Test file list (per-batch)
# ============================================================================
def get_test_file_list(dataset_name, batch):
    root = "data/Processed/" + dataset_name + " data"
    if dataset_name == "XJTU":
        files = sorted([f for f in os.listdir(root) if f.endswith('.csv') and f.startswith(batch)])
        return [os.path.join(root, f) for f in files if any(s in f for s in ['-4.', '-8.', '-14.', '-18.'])]
    elif dataset_name == "TJU":
        tju_dir = TJU_DIR_MAP.get(batch, batch)
        batch_root = os.path.join(root, tju_dir)
        if not os.path.isdir(batch_root):
            return []
        files = sorted(os.listdir(batch_root))
        test_list = []
        for i, f in enumerate(files):
            if (i + 1) % 10 == 5 or (i + 1) % 10 == 9:
                test_list.append(os.path.join(batch_root, f))
        return test_list
    elif dataset_name == "MIT":
        batch_root = os.path.join(root, batch)
        if not os.path.isdir(batch_root):
            return []
        test_list = []
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

# ============================================================================
# Model loading
# ============================================================================
def resolve_lax_args(args):
    ds = args.data
    suffix = {"XJTU": "_XJTU", "TJU": "_TJU", "MIT": "_MIT"}.get(ds, "_HUST")
    mapping = {
        "betha_LAX": f"betha_LAX{suffix}", "dual_LAX": f"dual_LAX{suffix}",
        "theta_LAX": f"theta_LAX{suffix}", "lr_net": f"lr_net_LAX{suffix}",
        "h_dim_LAX": f"h_dim_LAX{suffix}", "beta_LAX": f"beta_LAX{suffix}",
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
        "g_dim": f"g_dim_LAX{suffix}", "h_out_LAX": f"H_out_LAX{suffix}",
        "g_out_LAX": f"g_out_LAX{suffix}", "phi_out_LAX": f"phi_out_LAX{suffix}",
        "dim_output_LAX": f"dim_output_LAX{suffix}", "lr_F": f"lr_F{suffix}",
    }
    for generic, specific in mapping.items():
        if hasattr(args, specific):
            setattr(args, generic, getattr(args, specific))

def detect_g_dim(data_path, dataset_name):
    root = os.path.join(data_path, dataset_name + " data")
    for dirpath, _, filenames in os.walk(root):
        csvs = sorted([f for f in filenames if f.endswith(".csv")])
        if csvs:
            df = pd.read_csv(os.path.join(dirpath, csvs[0]))
            return df.shape[1] - 1
    return 16

def load_bagging_args(dataset_name):
    args = get_Bagging_u_args()
    args.data = dataset_name
    args.batch = "All"
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
    for k in ["XJTU", "TJU", "MIT", "HUST"]:
        setattr(args, f"g_dim_LAX_{k}", g_dim)
        setattr(args, f"distance_block_LAX_{k}", "MLP")
    args.h_dim_LAX_MIT = 30
    resolve_lax_args(args)
    return args

def load_bagging_model(dataset_name, args, batch_name=None):
    if batch_name:
        ckpt_path = OUTPUTS_ROOT / dataset_name / batch_name / "Bagging" / "best_model.pth"
    else:
        ckpt_path = OUTPUTS_ROOT / dataset_name / "Bagging" / "best_model.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = checkpoint['input_dimention']
    g_dim = detect_g_dim("data/Processed", dataset_name)
    x_dim = g_dim
    data_cls_map = {"XJTU": XJTUdata, "TJU": TJUdata, "MIT": MITdata, "HUST": HUSTdata}
    root = "data/Processed/" + dataset_name + " data"
    data_obj = data_cls_map[dataset_name](root=root, args=args)
    all_test_files = []
    for b in get_batches(dataset_name):
        all_test_files.extend(get_test_file_list(dataset_name, b))
    sum_ = 0.0; sum_sq = 0.0; n_samples = 0
    for path in all_test_files:
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
        layer_sizes_branch=layer_sizes_branch, layer_sizes_trunk=layer_sizes_trunk,
        activation="relu", kernel_initializer="Glorot normal"
    ).to(device)
    extractor.load_state_dict(checkpoint['feature_extractor'], strict=False)
    extractor.eval()
    solution_u = Solution_u(input_dim=m, layers_num=args.F_layers_num, hidden_dim=60, dropout=args.dropout).to(device)
    solution_u.load_state_dict(checkpoint['solution_u'])
    solution_u.eval()
    lax_model = OptimizationNetwork(x_sts=(X_mean_t, X_std_t), y_dim=x_dim, x_dim=x_dim, args=args).to(device)
    lax_state = dict(checkpoint['LAX'])
    if "y" in lax_state and lax_state["y"].shape != lax_model.y.shape:
        lax_state["y"] = lax_model.y.data
    lax_model.load_state_dict(lax_state, strict=False)
    lax_model.eval()
    bagging_NN = MLP_Bagging_NN(input_dim=3, output_dim=1, hidden_dim=args.bag_hidden_dim, dropout=args.dropout).to(device)
    bagging_NN.load_state_dict(checkpoint['bagging_u'])
    bagging_NN.eval()
    return extractor, solution_u, lax_model, bagging_NN, m, data_obj

def kneaware_inference(extractor, solution_u, lax_model, bagging_NN, m, x_tensor, kd_tensor, epoch=0):
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
        y_post = postprocess_capacity(
            y_pred, jump_thresh=0.02, hampel_window=21, savgol_window=101,
            polyorder=3, isotonic=True, clip=None,
        )
        battery_results.append({
            "name": battery_name, "y_true": y,
            "y_pinn": u_pinn, "y_lax": u_lax, "y_pred": y_pred, "y_post": y_post,
        })
    return battery_results

# ============================================================================
# Metrics
# ============================================================================
def compute_metrics(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    denom = np.clip(np.abs(y_true), 1e-12, None)
    mape = np.mean(np.abs((y_true - y_pred) / denom))  # fraction, not %
    return {"Pearson_R": r, "MAE": mae, "MSE": mse, "RMSE": rmse,
            "MAPE": mape, "MAPE%": mape * 100}

# ============================================================================
# Plot: Per-Batch Capacity (one subplot per battery in that batch)
# ============================================================================
def plot_batch_capacity(dataset_name, batch_name, battery_results, save_dir):
    n = len(battery_results)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    fig.suptitle(f"{dataset_name} [{batch_name}] — Post-Processed Capacity", fontsize=16, fontweight="bold", y=1.02)
    for idx, br in enumerate(battery_results):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.plot(br["y_true"], linewidth=1.8, color="black", label="True", zorder=5)
        ax.plot(br["y_pred"], linewidth=1.0, color=MODEL_CLRS["Knee-Aware DOPLAX"], linestyle="-", alpha=0.5, label="KAD")
        ax.plot(br["y_post"], linewidth=2.0, color=MODEL_CLRS["Post-Processed"], linestyle="-", alpha=1.0, label="KAD+Post")
        m_kad = compute_metrics(br["y_true"], br["y_pred"])
        m_post = compute_metrics(br["y_true"], br["y_post"])
        ax.set_title(f"{br['name']}\nKAD R={m_kad['Pearson_R']:.3f} | Post R={m_post['Pearson_R']:.3f}", fontsize=7)
        ax.set_xlabel("Cycle", fontsize=7); ax.set_ylabel("Capacity", fontsize=7)
        ax.tick_params(labelsize=6); ax.legend(fontsize=6, loc="best"); ax.grid(True, alpha=0.3)
        ax.set_ylim(br["y_true"].min() - 0.05, br["y_true"].max() + 0.05)
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)
    plt.tight_layout()
    safe_batch = re.sub(r'[\\/*?:"<>|]', '_', str(batch_name))
    plt.savefig(save_dir / f"postproc_{dataset_name}_{safe_batch}_capacity.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved {dataset_name}_{safe_batch}_capacity.png ({n} batteries)")

# ============================================================================
# Plot: Per-Batch Scatter (KAD vs Post-Proc only)
# ============================================================================
def plot_batch_scatter(dataset_name, batch_name, battery_results, save_dir):
    n = len(battery_results)
    if n == 0:
        return
    model_keys = [("Knee-Aware DOPLAX", "y_pred"), ("Post-Processed", "y_post")]
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.5 * n), squeeze=False)
    fig.suptitle(f"{dataset_name} [{batch_name}] — KAD vs Post-Proc", fontsize=16, fontweight="bold", y=1.005)
    for idx, br in enumerate(battery_results):
        for col, (mn, key) in enumerate(model_keys):
            yt = br["y_true"]; yp = br[key]; m = compute_metrics(yt, yp)
            ax = axes[idx][col]
            ax.scatter(yt, yp, s=4, alpha=0.4, color=MODEL_CLRS[mn])
            lo = min(yt.min(), yp.min()) - 0.01
            hi = max(yt.max(), yp.max()) + 0.01
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=0.8)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            if idx == 0: ax.set_title(mn, fontsize=10, fontweight="bold")
            if col == 0: ax.set_ylabel(f"{br['name']}\nR={m['Pearson_R']:.3f}\nRMSE={m['RMSE']:.4f}", fontsize=7)
            ax.set_xlabel("Actual" if idx == n - 1 else "", fontsize=8)
            ax.tick_params(labelsize=7); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_batch = re.sub(r'[\\/*?:"<>|]', '_', str(batch_name))
    plt.savefig(save_dir / f"postproc_{dataset_name}_{safe_batch}_scatter.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    Saved {dataset_name}_{safe_batch}_scatter.png ({n} batteries)")

# ============================================================================
# Per-dataset per-batch metrics CSV
# ============================================================================
def save_batch_metrics(dataset_name, batch_name, battery_results, save_dir):
    models = [("DeepOPINN", "y_pinn"), ("LAX", "y_lax"),
              ("Knee-Aware DOPLAX", "y_pred"), ("Post-Processed", "y_post")]
    rows = []
    for br in battery_results:
        for mn, key in models:
            m = compute_metrics(br["y_true"], br[key])
            rows.append({"battery": br["name"], "model": mn, **{k: round(v, 6) for k, v in m.items()}})
    df_detail = pd.DataFrame(rows) if rows else pd.DataFrame()
    overall = []
    for mn, key in models:
        all_t = np.concatenate([b["y_true"] for b in battery_results]) if battery_results else np.array([])
        all_p = np.concatenate([b[key] for b in battery_results]) if battery_results else np.array([])
        if len(all_t) == 0: continue
        m = compute_metrics(all_t, all_p)
        overall.append({"battery": "OVERALL", "model": mn, **{k: round(v, 6) for k, v in m.items()}})
    df_overall = pd.DataFrame(overall)
    df = pd.concat([df_detail, df_overall], ignore_index=True) if not df_detail.empty else df_overall
    safe_batch = re.sub(r'[\\/*?:"<>|]', '_', str(batch_name))
    path = save_dir / f"postproc_{dataset_name}_{safe_batch}_metrics.csv"
    df.to_csv(path, index=False)
    print(f"    Saved metrics to {path.name}")
    return df_overall

# ============================================================================
# Batch comparison bar chart (per dataset: each batch, KAD vs Post-Proc)
# ============================================================================
def plot_batch_comparison(dataset_name, batch_summaries, save_dir):
    if not batch_summaries:
        return
    models = ["Knee-Aware DOPLAX", "Post-Processed"]
    keys = ["y_pred", "y_post"]
    metric_names = ["RMSE", "MAE", "MAPE", "Pearson_R"]
    batch_names = [s["batch"] for s in batch_summaries]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{dataset_name} — Per-Batch Comparison (KAD vs Post-Proc)", fontsize=14, fontweight="bold")
    bar_width = 0.3
    x = np.arange(len(batch_names))
    for ax, metric in zip(axes, metric_names):
        for i, (mn, key) in enumerate(zip(models, keys)):
            vals = []
            for s in batch_summaries:
                r = s["results"]
                all_t = np.concatenate([b["y_true"] for b in r])
                all_p = np.concatenate([b[key] for b in r])
                m = compute_metrics(all_t, all_p)
                vals.append(m[metric])
            bars = ax.bar(x + i * bar_width, vals, bar_width, label=mn,
                         color=MODEL_CLRS[mn], edgecolor="black", linewidth=0.5, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.4f}" if metric != "MAPE" else f"{v:.4f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(batch_names, fontsize=8, rotation=15, ha="right")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel("Value")
        ax.legend(fontsize=7); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"postproc_{dataset_name}_batch_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {dataset_name}_batch_comparison.png")

# ============================================================================
# Cross-dataset bar chart (Post-Proc only, best batch per dataset)
# ============================================================================
def plot_cross_dataset_comparison(all_dataset_summaries):
    models = ["Knee-Aware DOPLAX", "Post-Processed"]
    keys = ["y_pred", "y_post"]
    metric_names = ["RMSE", "MAE", "MAPE", "Pearson_R"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Cross-Dataset Comparison — KAD vs Post-Processed", fontsize=16, fontweight="bold")
    bar_width = 0.3
    ds_names = sorted(all_dataset_summaries.keys())
    x = np.arange(len(ds_names))
    for ax, metric in zip(axes, metric_names):
        for i, (mn, key) in enumerate(zip(models, keys)):
            vals = []
            for ds in ds_names:
                all_t = np.concatenate([b["y_true"] for b in all_dataset_summaries[ds]])
                all_p = np.concatenate([b[key] for b in all_dataset_summaries[ds]])
                m = compute_metrics(all_t, all_p)
                vals.append(m[metric])
            bars = ax.bar(x + i * bar_width, vals, bar_width, label=mn,
                         color=MODEL_CLRS[mn], edgecolor="black", linewidth=0.5, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.4f}" if metric != "MAPE" else f"{v:.4f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(ds_names, fontsize=11)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "postproc_cross_dataset_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved postproc_cross_dataset_comparison.png")

# ============================================================================
# Cross-dataset metrics table
# ============================================================================
def save_cross_dataset_table(all_dataset_summaries):
    all_models = ["DeepOPINN", "LAX", "Knee-Aware DOPLAX", "Post-Processed"]
    all_keys = ["y_pinn", "y_lax", "y_pred", "y_post"]
    model_colors = {"DeepOPINN": "#D6E4F0", "LAX": "#FCE4D6",
                    "Knee-Aware DOPLAX": "#FFF2CC", "Post-Processed": "#C6EFCE"}

    # ── Aggregated summary (1 row per dataset × model) ──
    agg_rows = []
    for ds, batteries in all_dataset_summaries.items():
        for mn, key in zip(all_models, all_keys):
            all_t = np.concatenate([b["y_true"] for b in batteries])
            all_p = np.concatenate([b[key] for b in batteries])
            m = compute_metrics(all_t, all_p)
            agg_rows.append({"Dataset": ds, "Model": mn, "N_batteries": len(batteries),
                             "N_samples": len(all_t), **{k: round(v, 6) for k, v in m.items()}})
    if not agg_rows:
        return
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(PLOTS_DIR / "postproc_metrics_summary.csv", index=False)
    print(f"  Saved postproc_metrics_summary.csv")

    # ── Per-battery detail rows ──
    detail_rows = []
    for ds, batteries in all_dataset_summaries.items():
        for br in batteries:
            for mn, key in zip(all_models, all_keys):
                m = compute_metrics(br["y_true"], br[key])
                detail_rows.append({
                    "Dataset": ds, "Batch": br.get("batch", ""),
                    "Battery": br["name"], "Model": mn,
                    **{k: round(v, 6) for k, v in m.items()}
                })
    if not detail_rows:
        return
    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(PLOTS_DIR / "postproc_metrics_detail.csv", index=False)
    print(f"  Saved postproc_metrics_detail.csv")

    # ── Build the granular table plot ──
    display_metrics = ["RMSE", "MAE", "MAPE%", "Pearson_R"]
    n_rows = len(detail_rows)
    row_height = 0.035 if n_rows > 100 else 0.045
    fig_h = max(4, n_rows * row_height + 2)
    fig, ax = plt.subplots(figsize=(36, fig_h))
    ax.axis("off")
    ax.set_title("Per-Battery Metrics — All Models  (MAPE = fraction; MAPE% = fraction × 100)",
                 fontsize=13, fontweight="bold", pad=20)

    col_labels = ["Dataset", "Batch", "Battery", "Model",
                  "RMSE", "MAE", "MAPE%", "Pearson R"]
    table_data = []
    for r in detail_rows:
        table_data.append([
            r["Dataset"], r["Batch"], r["Battery"], r["Model"],
            f"{r['RMSE']:.4f}", f"{r['MAE']:.4f}",
            f"{r['MAPE%']:.2f}", f"{r['Pearson_R']:.4f}"
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.3)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#2F5496")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7)
            cell.set_height(0.06)
        else:
            cell.set_height(row_height * 0.9)
            cell.set_fontsize(5)
            mn = detail_rows[row_idx - 1]["Model"]
            clr = model_colors.get(mn, "white")
            if col_idx < 3:
                ds = detail_rows[row_idx - 1]["Dataset"]
                cell.set_facecolor(clr)
                is_first_of_group = (row_idx == 1 or
                                     detail_rows[row_idx - 1]["Dataset"] != detail_rows[row_idx - 2]["Dataset"] or
                                     detail_rows[row_idx - 1]["Batch"] != detail_rows[row_idx - 2]["Batch"] or
                                     detail_rows[row_idx - 1]["Battery"] != detail_rows[row_idx - 2]["Battery"])
            else:
                cell.set_facecolor(clr)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx > 0 and col_idx < 4:
            cell.set_fontsize(5)
        elif row_idx > 0:
            fmt_cell = cell.get_text().get_text()
            try:
                v = float(fmt_cell)
                if col_idx == 7:
                    cell.set_text_props(color="#006100" if v > 0.9 else "#9C0006")
                elif col_idx in (4, 5, 6):
                    cell.set_text_props(color="#006100" if v < 0.05 else "#9C0006")
            except ValueError:
                pass

    plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02)
    plt.savefig(PLOTS_DIR / "postproc_metrics_table.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved postproc_metrics_table.png ({n_rows} detail rows)")

# ============================================================================
# Cleanup
# ============================================================================
def cleanup():
    print("\n  Cleaning up old artifacts ...")
    for f in PLOTS_DIR.glob("postproc_*"):
        if f.name.endswith(".png") or f.name.endswith(".csv"):
            f.unlink(); print(f"    Deleted {f.name}")
    for f in Path(".").glob("summary_of_all_metrics.csv"):
        f.unlink(); print(f"    Deleted {f.name}")
    for f in Path(".").glob("averaged_losses_by_group.csv"):
        f.unlink(); print(f"    Deleted {f.name}")
    print("  Cleanup done.")

# ============================================================================
# Main
# ============================================================================
def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  Per-Batch Post-Processing Pipeline (SavGol+Hampel+Isotonic)")
    print("=" * 60)

    cleanup()

    all_dataset_summaries = {}

    for ds_name in ["XJTU", "TJU", "MIT", "HUST"]:
        print(f"\n  === {ds_name} ===")
        try:
            args = load_bagging_args(ds_name)
            args_cache = args

            batches = get_batches(ds_name)
            batch_summaries = []
            all_batteries = []
            ds_plot_dir = PLOTS_DIR
            ds_plot_dir.mkdir(parents=True, exist_ok=True)

            for batch in batches:
                bname = get_batch_name(ds_name, batch)
                test_files = get_test_file_list(ds_name, batch)
                if not test_files:
                    print(f"    [{bname}] No test files found")
                    continue
                print(f"  [{bname}] {len(test_files)} test batteries")

                # Load batch-specific Bagging model if it exists
                batch_ckpt = OUTPUTS_ROOT / ds_name / batch / "Bagging" / "best_model.pth"
                use_batch_model = batch_ckpt.exists()
                model_parts = load_bagging_model(ds_name, args_cache, batch_name=batch if use_batch_model else None)
                extractor, solution_u, lax_model, bagging_NN, m, data_obj = model_parts

                battery_results = predict_per_battery(
                    extractor, solution_u, lax_model, bagging_NN, m, data_obj, test_files, args_cache
                )

                # Tag each battery with batch and dataset info
                for br in battery_results:
                    br["batch"] = bname
                    br["dataset"] = ds_name

                # Per-batch plots
                plot_batch_capacity(ds_name, bname, battery_results, ds_plot_dir)
                plot_batch_scatter(ds_name, bname, battery_results, ds_plot_dir)

                # Per-batch metrics CSV
                overall_df = save_batch_metrics(ds_name, bname, battery_results, ds_plot_dir)

                batch_summaries.append({"batch": bname, "results": battery_results})
                all_batteries.extend(battery_results)

                # Save batch-specific artifacts to batch output folder
                if use_batch_model:
                    batch_out = OUTPUTS_ROOT / ds_name / batch
                    batch_out.mkdir(parents=True, exist_ok=True)
                    overall_df.to_csv(batch_out / "metrics.csv", index=False)
                    # Copy plots to batch folder
                    import shutil
                    for ext in ["capacity.png", "scatter.png"]:
                        src = ds_plot_dir / f"postproc_{ds_name}_{bname}_{ext}"
                        if src.exists():
                            shutil.copy2(src, batch_out / ext)
                    print(f"    Saved batch artifacts to {batch_out}")

                del extractor, solution_u, lax_model, bagging_NN
                torch.cuda.empty_cache()

            if all_batteries:
                # Per-dataset batch comparison
                plot_batch_comparison(ds_name, batch_summaries, PLOTS_DIR)
                all_dataset_summaries[ds_name] = all_batteries

        except Exception as e:
            print(f"  ERROR on {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_dataset_summaries:
        plot_cross_dataset_comparison(all_dataset_summaries)
        save_cross_dataset_table(all_dataset_summaries)

    print(f"\n  All outputs in: {PLOTS_DIR}")
    print("Done.")

if __name__ == "__main__":
    main()
