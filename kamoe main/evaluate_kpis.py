import os
import sys
import json
import argparse
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd

from train_tcn import ALL_13_BATCHES
from inference import run_inference_for_batch


def load_or_run_batch_metrics(
    dataset_name: str,
    batch_name: str,
    inference_dir: str,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Loads batch summary from inference output directory, or triggers inference if missing.
    """
    summary_file = os.path.join(inference_dir, dataset_name, batch_name, "batch_inference_summary.json")
    if os.path.exists(summary_file) and not args.recompute:
        with open(summary_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[*] Running inference for [{dataset_name}] - [{batch_name}]...")
    inf_args = argparse.Namespace(
        dataset=dataset_name,
        batch=batch_name,
        checkpoints_dir="checkpoints",
        output_dir=inference_dir,
        data_root=args.data_root,
        kpd_root="predictions/TCN_KPD",
        force_train=False,
        epochs=None
    )
    return run_inference_for_batch(dataset_name, batch_name, inf_args)


def main():
    parser = argparse.ArgumentParser(description="Evaluate KPIs and generate benchmark comparison tables across 13 batches.")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "XJTU", "MIT", "TJU", "HUST"],
                        help="Dataset name ('all' for all datasets, or specific: 'XJTU', 'MIT', 'TJU', 'HUST').")
    parser.add_argument("--batch", type=str, default="all",
                        help="Batch name ('all' for all batches in dataset, or specific batch).")
    parser.add_argument("--inference_dir", type=str, default="predictions/Inference", help="Directory where inference results are stored.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save KPI reports.")
    parser.add_argument("--data_root", type=str, default="../DOPLAX-SOH-main 4/data/Processed", help="Root directory of processed data.")
    parser.add_argument("--recompute", action="store_true", help="Recompute inference metrics even if summary JSON exists.")

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Determine tasks
    tasks: list = []
    if args.dataset == "all":
        tasks = ALL_13_BATCHES
    else:
        target_d = args.dataset.upper()
        if args.batch == "all":
            tasks = [(d, b) for (d, b) in ALL_13_BATCHES if d.upper() == target_d]
        else:
            tasks = [(target_d, args.batch)]

    table_rows = []

    for d, b in tasks:
        try:
            summary = load_or_run_batch_metrics(d, b, args.inference_dir, args)
            
            dop_rmse = summary['test_DeepOPINN']['RMSE']
            dop_mae = summary['test_DeepOPINN']['MAE']
            dop_mape = summary['test_DeepOPINN']['MAPE']

            lax_rmse = summary['test_LAX']['RMSE']
            lax_mae = summary['test_LAX']['MAE']
            lax_mape = summary['test_LAX']['MAPE']

            raw_rmse = summary['test_Fusion_Raw']['RMSE']
            raw_mae = summary['test_Fusion_Raw']['MAE']
            raw_mape = summary['test_Fusion_Raw']['MAPE']

            post_rmse = summary['test_Fusion_Post']['RMSE']
            post_mae = summary['test_Fusion_Post']['MAE']
            post_mape = summary['test_Fusion_Post']['MAPE']

            kpd_rmse = summary['test_KPD']['RMSE']
            kpd_mae = summary['test_KPD']['MAE']
            knee_err = summary['test_KPD']['mean_knee_delta_cycles']

            # % improvement of Fused SOH (Post) over best individual physics model
            best_single_rmse = min(dop_rmse, lax_rmse) if min(dop_rmse, lax_rmse) > 0 else 1e-6
            rel_imp_rmse = ((best_single_rmse - post_rmse) / best_single_rmse) * 100.0

            table_rows.append({
                'Dataset': d,
                'Batch': b,
                'Total_Batteries': summary['total_batteries'],
                'Test_Batteries': summary['test_batteries'],
                'KPD_RMSE': kpd_rmse,
                'KPD_MAE': kpd_mae,
                'Knee_Delta_Cycles': knee_err,
                'DeepOPINN_RMSE': dop_rmse,
                'DeepOPINN_MAE': dop_mae,
                'DeepOPINN_MAPE': dop_mape,
                'LAX_RMSE': lax_rmse,
                'LAX_MAE': lax_mae,
                'LAX_MAPE': lax_mape,
                'Fusion_Raw_RMSE': raw_rmse,
                'Fusion_Raw_MAE': raw_mae,
                'Fusion_Raw_MAPE': raw_mape,
                'Fused_SOH_RMSE': post_rmse,
                'Fused_SOH_MAE': post_mae,
                'Fused_SOH_MAPE': post_mape,
                'Improvement_vs_Best_Single_%': rel_imp_rmse
            })
        except Exception as e:
            print(f"[!] Error loading metrics for [{d}] - [{b}]: {str(e)}")

    if not table_rows:
        print("[!] No metrics found. Run inference first!")
        return

    df_kpi = pd.DataFrame(table_rows)

    # Save CSV
    csv_path = os.path.join(args.results_dir, "benchmark_kpis.csv")
    df_kpi.to_csv(csv_path, index=False)

    # Save JSON
    json_path = os.path.join(args.results_dir, "benchmark_kpis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=4)

    # Format Markdown Table
    md_lines = [
        "# Benchmark KPI Evaluation Table: Knee-Aware DOPLAX vs. Baseline Models\n",
        f"Generated across **{len(table_rows)} batches**\n",
        "| # | Dataset | Batch Name | DeepOPINN RMSE | LAX RMSE | Fused SOH (Post) RMSE | Fused SOH MAE | Fused SOH MAPE (%) | Knee Delta (cyc) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for idx, r in enumerate(table_rows, 1):
        md_lines.append(
            f"| {idx} | {r['Dataset']} | {r['Batch']} | {r['DeepOPINN_RMSE']:.5f} | {r['LAX_RMSE']:.5f} | **{r['Fused_SOH_RMSE']:.5f}** | {r['Fused_SOH_MAE']:.5f} | {r['Fused_SOH_MAPE']:.2f}% | {r['Knee_Delta_Cycles']:.1f} |"
        )

    # Averages
    mean_dop_rmse = df_kpi['DeepOPINN_RMSE'].mean()
    mean_lax_rmse = df_kpi['LAX_RMSE'].mean()
    mean_fused_rmse = df_kpi['Fused_SOH_RMSE'].mean()
    mean_fused_mae = df_kpi['Fused_SOH_MAE'].mean()
    mean_fused_mape = df_kpi['Fused_SOH_MAPE'].mean()
    mean_knee_err = df_kpi['Knee_Delta_Cycles'].mean()

    md_lines.append(
        f"| **AVG** | **ALL** | **Overall Average** | **{mean_dop_rmse:.5f}** | **{mean_lax_rmse:.5f}** | **{mean_fused_rmse:.5f}** | **{mean_fused_mae:.5f}** | **{mean_fused_mape:.2f}%** | **{mean_knee_err:.1f}** |"
    )

    md_content = "\n".join(md_lines)
    md_path = os.path.join(args.results_dir, "benchmark_kpis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Print Console Table
    print("\n" + "=" * 135)
    print(f"{'#':<3} | {'Dataset':<6} | {'Batch Name':<28} | {'DeepOPINN RMSE':<16} | {'LAX RMSE':<12} | {'Fused SOH RMSE':<16} | {'Fused MAE':<12} | {'Knee Err (cyc)':<16}")
    print("=" * 135)
    for idx, r in enumerate(table_rows, 1):
        print(f"{idx:<3} | {r['Dataset']:<6} | {r['Batch']:<28} | {r['DeepOPINN_RMSE']:<16.6f} | {r['LAX_RMSE']:<12.6f} | {r['Fused_SOH_RMSE']:<16.6f} | {r['Fused_SOH_MAE']:<12.6f} | {r['Knee_Delta_Cycles']:<16.2f}")
    print("=" * 135)
    print(f"{'AVG':<3} | {'ALL':<6} | {'Overall Benchmark Average':<28} | {mean_dop_rmse:<16.6f} | {mean_lax_rmse:<12.6f} | {mean_fused_rmse:<16.6f} | {mean_fused_mae:<12.6f} | {mean_knee_err:<16.2f}")
    print("=" * 135)
    print(f"Reports saved successfully under '{os.path.abspath(args.results_dir)}':")
    print(f"  - CSV:      {csv_path}")
    print(f"  - Markdown: {md_path}")
    print(f"  - JSON:     {json_path}")
    print("=" * 135)


if __name__ == "__main__":
    main()
