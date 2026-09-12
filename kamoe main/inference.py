import os
import sys
import time
import json
import argparse
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataloader import load_battery_dataloader
from configs import (
    get_tcn_config, get_deepopinn_config, get_lax_config, get_fusion_config,
    TCNConfig, DeepOPINNConfig, LAXConfig, FusionMLPConfig
)
from Model import TemporalConvNet, DeepOPINN, OptimizationNetwork, FusionMLP
from Model.utils.util import eval_metrix, get_logger, AverageMeter
from post_processing import postprocess_capacity
from train_tcn import ALL_13_BATCHES, train_single_battery_batch
from train_deepopinn import train_single_deepopinn_batch
from train_lax import train_single_lax_batch, compute_dataset_feature_stats
from train_fusion import train_single_fusion_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def estimate_knee_cycle_from_kpd(cycles: np.ndarray, pred_kpd: np.ndarray) -> int:
    """Estimates predicted knee cycle index where |pred_kpd| is minimized."""
    kpd_flat = pred_kpd.flatten()
    cycles_flat = cycles.flatten()
    min_idx = np.argmin(np.abs(kpd_flat))
    return int(cycles_flat[min_idx])


def run_inference_for_batch(
    dataset_name: str,
    batch_name: str,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Executes end-to-end inference across all 4 models for all batteries in a batch.
    Applies post-processing without monotonic-only constraint.
    """
    out_dir = os.path.join(args.output_dir, dataset_name, batch_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(out_dir, filename="inference.log")

    logger.info("==================================================")
    logger.info(f"Running Full Inference for [{dataset_name}] - [{batch_name}]")
    logger.info("==================================================")

    # 1. Load Data
    loaders = load_battery_dataloader(
        dataset_name=dataset_name,
        batch_name=batch_name,
        data_root=args.data_root,
        normalization_method="min-max",
        shuffle_train=False
    )
    train_loader = loaders['train']

    # 2. Check & Load Checkpoints (Auto-train if missing)
    # A. TCN
    tcn_cfg = get_tcn_config(dataset_name)
    tcn_ckpt = os.path.join(args.checkpoints_dir, "TCN", dataset_name, batch_name, "best_tcn.pth")
    if not os.path.exists(tcn_ckpt) or args.force_train:
        print(f"[*] Training missing TCN for [{dataset_name}] - [{batch_name}]...")
        t_args = argparse.Namespace(epochs=args.epochs, lr=None, early_stop=None, data_root=args.data_root, save_dir=os.path.join(args.checkpoints_dir, "TCN"))
        train_single_battery_batch(dataset_name, batch_name, t_args, config=tcn_cfg)

    model_tcn = TemporalConvNet(
        num_inputs=tcn_cfg.num_inputs,
        num_channels=tcn_cfg.num_channels,
        kernel_size=tcn_cfg.kernel_size,
        dropout=tcn_cfg.dropout,
        output_dim=tcn_cfg.output_dim
    ).to(device)
    ckpt_tcn = torch.load(tcn_ckpt, map_location=device)
    model_tcn.load_state_dict({k: v.to(device) for k, v in ckpt_tcn['model_state_dict'].items()})
    model_tcn.eval()

    # B. DeepOPINN
    dop_cfg = get_deepopinn_config(dataset_name)
    dop_ckpt = os.path.join(args.checkpoints_dir, "DeepOPINN", dataset_name, batch_name, "best_deepopinn.pth")
    if not os.path.exists(dop_ckpt) or args.force_train:
        print(f"[*] Training missing DeepOPINN for [{dataset_name}] - [{batch_name}]...")
        d_args = argparse.Namespace(epochs=args.epochs, lr=None, early_stop=None, data_root=args.data_root, kpd_root=args.kpd_root, save_dir=os.path.join(args.checkpoints_dir, "DeepOPINN"))
        train_single_deepopinn_batch(dataset_name, batch_name, d_args, config=dop_cfg)

    model_dop = DeepOPINN(config=dop_cfg).to(device)
    ckpt_dop = torch.load(dop_ckpt, map_location=device)
    model_dop.load_state_dict(ckpt_dop['model_state_dict'])
    model_dop.eval()

    # C. LAX
    lax_cfg = get_lax_config(dataset_name)
    lax_ckpt = os.path.join(args.checkpoints_dir, "LAX", dataset_name, batch_name, "best_lax.pth")
    if not os.path.exists(lax_ckpt) or args.force_train:
        print(f"[*] Training missing LAX for [{dataset_name}] - [{batch_name}]...")
        l_args = argparse.Namespace(epochs=args.epochs, lr=None, early_stop=None, data_root=args.data_root, save_dir=os.path.join(args.checkpoints_dir, "LAX"))
        train_single_lax_batch(dataset_name, batch_name, l_args, config=lax_cfg)

    x_mean, x_std = compute_dataset_feature_stats(train_loader)
    model_lax = OptimizationNetwork(config=lax_cfg, x_sts=(x_mean, x_std)).to(device)
    ckpt_lax = torch.load(lax_ckpt, map_location=device)
    model_lax.load_state_dict(ckpt_lax['model_state_dict'])
    model_lax.eval()

    # D. FusionMLP
    fusion_cfg = get_fusion_config(dataset_name)
    fusion_ckpt = os.path.join(args.checkpoints_dir, "FusionMLP", dataset_name, batch_name, "best_fusion.pth")
    if not os.path.exists(fusion_ckpt) or args.force_train:
        print(f"[*] Training missing FusionMLP for [{dataset_name}] - [{batch_name}]...")
        f_args = argparse.Namespace(
            epochs=args.epochs, lr=None, early_stop=None, data_root=args.data_root,
            kpd_root=args.kpd_root, deepopinn_dir=os.path.join(args.checkpoints_dir, "DeepOPINN"),
            lax_dir=os.path.join(args.checkpoints_dir, "LAX"), save_dir=os.path.join(args.checkpoints_dir, "FusionMLP"),
            force_train=False
        )
        train_single_fusion_batch(dataset_name, batch_name, f_args, config=fusion_cfg)

    model_fusion = FusionMLP(config=fusion_cfg).to(device)
    ckpt_fusion = torch.load(fusion_ckpt, map_location=device)
    model_fusion.load_state_dict(ckpt_fusion['model_state_dict'])
    model_fusion.eval()

    # 3. Perform Inference across all splits
    battery_summaries = {}
    test_true_all = []
    test_dop_all = []
    test_lax_all = []
    test_raw_all = []
    test_post_all = []
    test_kpd_true_all = []
    test_kpd_pred_all = []
    test_knee_errors = []

    for split in ['train', 'valid', 'test']:
        loader = loaders[split]
        for sample in loader:
            b_name = sample['battery_name']
            clean_name = b_name[:-4] if b_name.endswith('.csv') else b_name
            x = sample['x'].to(device)
            y_true = sample['y'].cpu().numpy()
            cycles = sample['cycle_indices'].cpu().numpy()
            true_kpd = sample['knee_distance'].cpu().numpy()
            true_knee_cycle = sample['knee_point']

            with torch.no_grad():
                # TCN KPD
                pred_kpd = model_tcn(x).cpu().numpy()
                kpd_tensor = torch.from_numpy(pred_kpd).to(device)

                # DeepOPINN
                _, pred_dop_tensor = model_dop(x)
                pred_dop = pred_dop_tensor.cpu().numpy()

                # LAX
                pred_lax_tensor = model_lax(x)
                pred_lax = pred_lax_tensor.cpu().numpy()

                # FusionMLP (Raw)
                pred_fusion_tensor = model_fusion(
                    u_deepopinn=pred_dop_tensor,
                    u_lax=pred_lax_tensor,
                    knee_distance=kpd_tensor if fusion_cfg.include_knee_feature else None
                )
                pred_fusion_raw = pred_fusion_tensor.cpu().numpy()

            # Post-processing (NO monotonic-only decreasing constraint)
            pred_fusion_post = postprocess_capacity(
                pred_fusion_raw,
                jump_thresh=0.02,
                hampel_window=21,
                savgol_window=51,
                polyorder=2
            ).reshape(-1, 1)

            # Knee estimation
            pred_knee_cycle = estimate_knee_cycle_from_kpd(cycles, pred_kpd)
            knee_err = abs(pred_knee_cycle - true_knee_cycle)

            # Metrics for this battery
            [dop_mae, dop_mape, dop_mse, dop_rmse] = eval_metrix(pred_dop, y_true)
            [lax_mae, lax_mape, lax_mse, lax_rmse] = eval_metrix(pred_lax, y_true)
            [raw_mae, raw_mape, raw_mse, raw_rmse] = eval_metrix(pred_fusion_raw, y_true)
            [post_mae, post_mape, post_mse, post_rmse] = eval_metrix(pred_fusion_post, y_true)
            [kpd_mae, _, kpd_mse, kpd_rmse] = eval_metrix(pred_kpd, true_kpd)

            # Save full trajectory CSV
            df_out = pd.DataFrame({
                'cycle_index': cycles.flatten().astype(int),
                'true_soh': y_true.flatten(),
                'pred_soh_deepopinn': pred_dop.flatten(),
                'pred_soh_lax': pred_lax.flatten(),
                'pred_soh_fusion_raw': pred_fusion_raw.flatten(),
                'pred_soh_fusion_post': pred_fusion_post.flatten(),
                'true_kpd': true_kpd.flatten(),
                'pred_kpd': pred_kpd.flatten(),
                'true_knee_cycle': true_knee_cycle,
                'pred_knee_cycle': pred_knee_cycle,
                'split': split
            })
            csv_path = os.path.join(out_dir, f"{clean_name}_full_pred.csv")
            df_out.to_csv(csv_path, index=False)

            # Save numpy array of final post-processed SOH
            npy_path = os.path.join(out_dir, f"{clean_name}_soh_fused_post.npy")
            np.save(npy_path, pred_fusion_post)

            battery_summaries[clean_name] = {
                'split': split,
                'cycles': len(cycles),
                'true_knee_cycle': true_knee_cycle,
                'pred_knee_cycle': pred_knee_cycle,
                'knee_cycle_error': knee_err,
                'KPD_RMSE': kpd_rmse,
                'DeepOPINN_RMSE': dop_rmse,
                'LAX_RMSE': lax_rmse,
                'Fusion_Raw_RMSE': raw_rmse,
                'Fusion_Post_RMSE': post_rmse,
                'Fusion_Post_MAE': post_mae,
                'Fusion_Post_MAPE': post_mape,
                'csv_path': csv_path
            }

            if split == 'test':
                test_true_all.append(y_true)
                test_dop_all.append(pred_dop)
                test_lax_all.append(pred_lax)
                test_raw_all.append(pred_fusion_raw)
                test_post_all.append(pred_fusion_post)
                test_kpd_true_all.append(true_kpd)
                test_kpd_pred_all.append(pred_kpd)
                test_knee_errors.append(knee_err)

    # 4. Overall Test Metrics
    if test_true_all:
        y_test_cat = np.concatenate(test_true_all, axis=0)
        dop_test_cat = np.concatenate(test_dop_all, axis=0)
        lax_test_cat = np.concatenate(test_lax_all, axis=0)
        raw_test_cat = np.concatenate(test_raw_all, axis=0)
        post_test_cat = np.concatenate(test_post_all, axis=0)
        kpd_true_cat = np.concatenate(test_kpd_true_all, axis=0)
        kpd_pred_cat = np.concatenate(test_kpd_pred_all, axis=0)

        test_dop_metrics = eval_metrix(dop_test_cat, y_test_cat)
        test_lax_metrics = eval_metrix(lax_test_cat, y_test_cat)
        test_raw_metrics = eval_metrix(raw_test_cat, y_test_cat)
        test_post_metrics = eval_metrix(post_test_cat, y_test_cat)
        test_kpd_metrics = eval_metrix(kpd_pred_cat, kpd_true_cat)
        mean_knee_delta = float(np.mean(test_knee_errors))
    else:
        test_dop_metrics = [0.0, 0.0, 0.0, 0.0]
        test_lax_metrics = [0.0, 0.0, 0.0, 0.0]
        test_raw_metrics = [0.0, 0.0, 0.0, 0.0]
        test_post_metrics = [0.0, 0.0, 0.0, 0.0]
        test_kpd_metrics = [0.0, 0.0, 0.0, 0.0]
        mean_knee_delta = 0.0

    batch_summary = {
        'dataset': dataset_name,
        'batch': batch_name,
        'total_batteries': len(battery_summaries),
        'test_batteries': len(test_true_all),
        'test_DeepOPINN': {'MAE': test_dop_metrics[0], 'MAPE': test_dop_metrics[1], 'MSE': test_dop_metrics[2], 'RMSE': test_dop_metrics[3]},
        'test_LAX': {'MAE': test_lax_metrics[0], 'MAPE': test_lax_metrics[1], 'MSE': test_lax_metrics[2], 'RMSE': test_lax_metrics[3]},
        'test_Fusion_Raw': {'MAE': test_raw_metrics[0], 'MAPE': test_raw_metrics[1], 'MSE': test_raw_metrics[2], 'RMSE': test_raw_metrics[3]},
        'test_Fusion_Post': {'MAE': test_post_metrics[0], 'MAPE': test_post_metrics[1], 'MSE': test_post_metrics[2], 'RMSE': test_post_metrics[3]},
        'test_KPD': {'MAE': test_kpd_metrics[0], 'MSE': test_kpd_metrics[2], 'RMSE': test_kpd_metrics[3], 'mean_knee_delta_cycles': mean_knee_delta},
        'battery_summaries': battery_summaries
    }

    with open(os.path.join(out_dir, "batch_inference_summary.json"), "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=4)

    logger.info(f"Completed Batch [{dataset_name}] - [{batch_name}]:")
    logger.info(f"  Test SOH DeepOPINN   RMSE: {test_dop_metrics[3]:.6f} | MAE: {test_dop_metrics[0]:.6f}")
    logger.info(f"  Test SOH LAX         RMSE: {test_lax_metrics[3]:.6f} | MAE: {test_lax_metrics[0]:.6f}")
    logger.info(f"  Test SOH Fusion Raw  RMSE: {test_raw_metrics[3]:.6f} | MAE: {test_raw_metrics[0]:.6f}")
    logger.info(f"  Test SOH Fusion Post RMSE: {test_post_metrics[3]:.6f} | MAE: {test_post_metrics[0]:.6f}")
    logger.info(f"  Mean Knee Delta: {mean_knee_delta:.2f} cycles")

    return batch_summary


def main():
    parser = argparse.ArgumentParser(description="End-to-End Battery SOH Inference across 13 benchmark batches.")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "XJTU", "MIT", "TJU", "HUST"],
                        help="Dataset to evaluate ('all' for all datasets, or specific: 'XJTU', 'MIT', 'TJU', 'HUST').")
    parser.add_argument("--batch", type=str, default="all",
                        help="Batch to evaluate ('all' for all batches in dataset, or specific batch).")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Root directory containing model checkpoints.")
    parser.add_argument("--output_dir", type=str, default="predictions/Inference", help="Output directory for full predictions.")
    parser.add_argument("--data_root", type=str, default="../DOPLAX-SOH-main 4/data/Processed", help="Root directory of processed data.")
    parser.add_argument("--kpd_root", type=str, default="predictions/TCN_KPD", help="Directory for KPD predictions.")
    parser.add_argument("--force_train", action="store_true", help="Force re-training if checkpoints are missing.")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs if retraining is triggered.")

    args = parser.parse_args()

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

    print("=" * 125)
    print(f"  End-to-End Battery SOH Inference Pipeline: {len(tasks)} Batches Scheduled")
    print("=" * 125)
    for idx, (d, b) in enumerate(tasks, 1):
        print(f"  {idx:2d}. Dataset: {d:<6} | Batch: {b}")
    print("=" * 125)

    all_summaries = []
    for idx, (d, b) in enumerate(tasks, 1):
        print(f"\n>>> Running Inference on Batch {idx}/{len(tasks)}: [{d}] - [{b}] ...")
        try:
            res = run_inference_for_batch(d, b, args)
            all_summaries.append(res)
            post_rmse = res['test_Fusion_Post']['RMSE']
            post_mae = res['test_Fusion_Post']['MAE']
            knee_d = res['test_KPD']['mean_knee_delta_cycles']
            print(f"    [COMPLETED] Test Fused SOH RMSE: {post_rmse:.6f} | MAE: {post_mae:.6f} | Mean Knee Delta: {knee_d:.1f} cyc")
        except Exception as e:
            print(f"    [FAILED] Error on [{d}] - [{b}]: {str(e)}")

    # Print Final Summary Table
    print("\n" + "=" * 125)
    print(f"{'#':<3} | {'Dataset':<6} | {'Batch Name':<28} | {'DeepOPINN RMSE':<16} | {'LAX RMSE':<12} | {'Fused SOH RMSE':<16} | {'Knee Err (cyc)':<16}")
    print("=" * 125)
    for idx, r in enumerate(all_summaries, 1):
        dop_r = r['test_DeepOPINN']['RMSE']
        lax_r = r['test_LAX']['RMSE']
        fuse_r = r['test_Fusion_Post']['RMSE']
        kd = r['test_KPD']['mean_knee_delta_cycles']
        print(f"{idx:<3} | {r['dataset']:<6} | {r['batch']:<28} | {dop_r:<16.6f} | {lax_r:<12.6f} | {fuse_r:<16.6f} | {kd:<16.2f}")
    print("=" * 125)

    # Save overall summary
    summary_path = os.path.join(args.output_dir, "all_batches_inference_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=4)
    print(f"Overall Inference Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
