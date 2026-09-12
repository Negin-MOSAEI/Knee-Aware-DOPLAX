import os
import sys
import glob
import argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Style settings for publication-ready figures
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0


def plot_single_battery_soh(df: pd.DataFrame, battery_name: str, save_path: str):
    """
    Plots true vs. predicted SOH degradation trajectories for a single battery.
    Includes DeepOPINN, LAX, and Fused Knee-Aware DOPLAX curves with knee point markers.
    """
    cycles = df['cycle_index'].values
    y_true = df['true_soh'].values
    dop_pred = df['pred_soh_deepopinn'].values
    lax_pred = df['pred_soh_lax'].values
    fuse_post = df['pred_soh_fusion_post'].values
    true_knee = int(df['true_knee_cycle'].iloc[0])
    pred_knee = int(df['pred_knee_cycle'].iloc[0])
    split = df['split'].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Ground Truth
    ax.plot(cycles, y_true, color='#111111', linestyle='-', linewidth=2.5, label='Ground Truth SOH', zorder=5)

    # Baselines
    ax.plot(cycles, dop_pred, color='#1f77b4', linestyle='--', linewidth=1.8, alpha=0.85, label='DeepOPINN', zorder=3)
    ax.plot(cycles, lax_pred, color='#2ca02c', linestyle='-.', linewidth=1.8, alpha=0.85, label='LAX Solver', zorder=3)

    # Fused Knee-Aware Model
    ax.plot(cycles, fuse_post, color='#d62728', linestyle='-', linewidth=2.2, label='Knee-Aware DOPLAX (Fused)', zorder=4)

    # Knee Point Markers
    if true_knee < len(cycles):
        ax.axvline(x=true_knee, color='#7f7f7f', linestyle='--', linewidth=1.5, alpha=0.9, label=f'True Knee Point ($C={true_knee}$)')
    if pred_knee < len(cycles):
        ax.axvline(x=pred_knee, color='#e377c2', linestyle=':', linewidth=1.8, alpha=0.9, label=f'Pred Knee Point ($C={pred_knee}$)')

    # Labels and Aesthetics
    ax.set_title(f"Battery SOH Degradation: {battery_name} ({split.upper()} Set)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Cycle Index ($C$)", fontsize=12, labelpad=8)
    ax.set_ylabel("State-of-Health / Capacity (Normalized)", fontsize=12, labelpad=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()


def plot_single_battery_kpd(df: pd.DataFrame, battery_name: str, save_path: str):
    """
    Plots true vs. predicted Knee Point Distance (KPD = -arctan(C - C_knee)) trajectory.
    """
    cycles = df['cycle_index'].values
    true_kpd = df['true_kpd'].values
    pred_kpd = df['pred_kpd'].values
    true_knee = int(df['true_knee_cycle'].iloc[0])
    pred_knee = int(df['pred_knee_cycle'].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    ax.plot(cycles, true_kpd, color='#333333', linestyle='-', linewidth=2.2, label=r'True KPD: $-\arctan(C - C_{\mathrm{knee}})$')
    ax.plot(cycles, pred_kpd, color='#9467bd', linestyle='--', linewidth=2.0, label='TCN Predicted KPD')

    # Zero Line (Knee Point Boundary)
    ax.axhline(0, color='red', linestyle='-', linewidth=1.0, alpha=0.7)

    # Shaded Operating Regions
    ax.fill_between(cycles, 0, np.maximum(0, true_kpd), color='#2ca02c', alpha=0.08, label='Pre-Knee Regime ($d > 0$)')
    ax.fill_between(cycles, np.minimum(0, true_kpd), 0, color='#d62728', alpha=0.08, label='Post-Knee Regime ($d < 0$)')

    if true_knee < len(cycles):
        ax.axvline(x=true_knee, color='#7f7f7f', linestyle='--', linewidth=1.5, label=f'True Knee ($C={true_knee}$)')
    if pred_knee < len(cycles):
        ax.axvline(x=pred_knee, color='#e377c2', linestyle=':', linewidth=1.8, label=f'Pred Knee ($C={pred_knee}$)')

    ax.set_title(f"Knee Point Distance Dynamics: {battery_name}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Cycle Index ($C$)", fontsize=11)
    ax.set_ylabel("Knee Point Distance (KPD)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True, framealpha=0.95, fontsize=9.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()


def plot_batch_grid(inference_dir: str, dataset: str, batch: str, save_dir: str, split_filter: str = 'test'):
    """
    Plots a multi-battery comparison grid for all test batteries in a batch.
    """
    batch_dir = os.path.join(inference_dir, dataset, batch)
    csv_files = glob.glob(os.path.join(batch_dir, "*_full_pred.csv"))

    if not csv_files:
        return

    # Filter by split if requested
    selected_files = []
    for f in csv_files:
        df = pd.read_csv(f)
        if split_filter == 'all' or df['split'].iloc[0] == split_filter:
            selected_files.append((f, df))

    if not selected_files:
        selected_files = [(f, pd.read_csv(f)) for f in csv_files]

    n_batteries = len(selected_files)
    n_cols = min(3, n_batteries)
    n_rows = int(np.ceil(n_batteries / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), dpi=300, squeeze=False)

    for idx, (f_path, df) in enumerate(selected_files):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        b_name = os.path.basename(f_path).replace("_full_pred.csv", "")
        cycles = df['cycle_index'].values
        y_true = df['true_soh'].values
        fuse_post = df['pred_soh_fusion_post'].values
        dop_pred = df['pred_soh_deepopinn'].values
        lax_pred = df['pred_soh_lax'].values
        true_knee = int(df['true_knee_cycle'].iloc[0])

        ax.plot(cycles, y_true, color='#111111', linestyle='-', linewidth=2.0, label='True SOH')
        ax.plot(cycles, dop_pred, color='#1f77b4', linestyle='--', linewidth=1.4, alpha=0.7, label='DeepOPINN')
        ax.plot(cycles, lax_pred, color='#2ca02c', linestyle='-.', linewidth=1.4, alpha=0.7, label='LAX')
        ax.plot(cycles, fuse_post, color='#d62728', linestyle='-', linewidth=1.8, label='Fused SOH')

        if true_knee < len(cycles):
            ax.axvline(x=true_knee, color='#7f7f7f', linestyle='--', linewidth=1.2, alpha=0.8)

        rmse = np.sqrt(np.mean((fuse_post - y_true) ** 2))
        ax.set_title(f"{b_name} (RMSE: {rmse:.4f})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Cycle", fontsize=10)
        ax.set_ylabel("SOH", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        if idx == 0:
            ax.legend(loc='lower left', fontsize=8.5)

    # Hide extra empty subplots
    for idx in range(n_batteries, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    fig.suptitle(f"Benchmark Degradation Curves: [{dataset}] - [{batch}] ({split_filter.upper()} Batteries)", fontsize=14, fontweight='bold', y=1.01)
    
    out_path = os.path.join(save_dir, f"{dataset}_{batch}_grid.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready degradation curves and knee point plots.")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "XJTU", "MIT", "TJU", "HUST"],
                        help="Dataset name ('all' for all datasets, or specific: 'XJTU', 'MIT', 'TJU', 'HUST').")
    parser.add_argument("--batch", type=str, default="all",
                        help="Batch name ('all' for all batches in dataset, or specific batch).")
    parser.add_argument("--inference_dir", type=str, default="predictions/Inference", help="Directory where inference results are stored.")
    parser.add_argument("--output_dir", type=str, default="results/figures", help="Directory to save generated figures.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid", "train", "all"], help="Filter by split.")

    args = parser.parse_args()

    # Search for all available batch folders in inference_dir
    all_csvs = glob.glob(os.path.join(args.inference_dir, "*", "*", "*_full_pred.csv"))
    if not all_csvs:
        print(f"[!] No inference CSV files found in '{args.inference_dir}'. Run inference.py first!")
        return

    print("=" * 115)
    print(f"  Publication Plot Generator: Found {len(all_csvs)} battery prediction trajectories")
    print(f"  Target Figures Directory: {os.path.abspath(args.output_dir)}")
    print("=" * 115)

    # Group by (dataset, batch)
    batches_found = {}
    for f in all_csvs:
        parts = os.path.normpath(f).split(os.sep)
        d_name = parts[-3]
        b_name = parts[-2]
        key = (d_name, b_name)
        if key not in batches_found:
            batches_found[key] = []
        batches_found[key].append(f)

    for (d, b), files in batches_found.items():
        if args.dataset != "all" and d.upper() != args.dataset.upper():
            continue
        if args.batch != "all" and b != args.batch:
            continue

        print(f"\n>>> Generating plots for Batch: [{d}] - [{b}] ({len(files)} batteries) ...")
        batch_fig_dir = os.path.join(args.output_dir, d, b)

        for csv_path in files:
            b_name = os.path.basename(csv_path).replace("_full_pred.csv", "")
            df = pd.read_csv(csv_path)

            if args.split != "all" and df['split'].iloc[0] != args.split:
                continue

            # Plot single SOH
            soh_fig_path = os.path.join(batch_fig_dir, f"{b_name}_SOH.png")
            plot_single_battery_soh(df, b_name, soh_fig_path)

            # Plot single KPD
            kpd_fig_path = os.path.join(batch_fig_dir, f"{b_name}_KPD.png")
            plot_single_battery_kpd(df, b_name, kpd_fig_path)

        # Plot batch grid
        plot_batch_grid(args.inference_dir, d, b, args.output_dir, split_filter=args.split)
        print(f"    Saved figures to: {batch_fig_dir}")

    print("\n" + "=" * 115)
    print("All plots generated successfully in PNG (300 DPI) and vector SVG formats!")
    print(f"Figures saved under: {os.path.abspath(args.output_dir)}")
    print("=" * 115)


if __name__ == "__main__":
    main()
