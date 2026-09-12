import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataloader import load_battery_dataloader
from configs import get_tcn_config, TCNConfig
from Model import TemporalConvNet
from Model.utils.util import eval_metrix, get_logger
from train_tcn import train_single_battery_batch, ALL_13_BATCHES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def estimate_knee_cycle_from_kpd(cycles: np.ndarray, pred_kpd: np.ndarray) -> int:
    """
    Estimates the predicted knee cycle index from the predicted KPD trajectory.
    Since KPD = -arctan(C - C_knee), the knee point occurs where KPD = 0
    (transition from positive in pre-knee to negative in post-knee).
    """
    kpd_flat = pred_kpd.flatten()
    cycles_flat = cycles.flatten()

    # Find the cycle where |kpd| is closest to 0
    min_idx = np.argmin(np.abs(kpd_flat))
    return int(cycles_flat[min_idx])


def predict_and_save_kpd_for_batch(
    dataset_name: str,
    batch_name: str,
    args: argparse.Namespace
) -> dict:
    """
    Predicts and saves Knee Point Distance (KPD) for all batteries (train, val, test) in a batch.
    If pre-trained model checkpoint does not exist (or force_train is True), trains it first.
    """
    cfg = get_tcn_config(dataset_name)
    ckpt_dir = os.path.join(args.checkpoints_dir, dataset_name, batch_name)
    ckpt_path = os.path.join(ckpt_dir, "best_tcn.pth")

    # If checkpoint does not exist or force_train is requested, train first
    if not os.path.exists(ckpt_path) or args.force_train:
        print(f"[*] Checkpoint not found at {ckpt_path}. Training TCN model first for [{dataset_name}] - [{batch_name}]...")
        train_args = argparse.Namespace(
            epochs=args.epochs,
            lr=args.lr,
            early_stop=args.early_stop,
            data_root=args.data_root,
            save_dir=args.checkpoints_dir
        )
        train_single_battery_batch(dataset_name, batch_name, train_args, config=cfg)

    # Load Model
    model = TemporalConvNet(
        num_inputs=cfg.num_inputs,
        num_channels=cfg.num_channels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
        output_dim=cfg.output_dim
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict({k: v.to(device) for k, v in checkpoint['model_state_dict'].items()})
    model.eval()

    # Setup output directory
    batch_out_dir = os.path.join(args.output_dir, dataset_name, batch_name)
    os.makedirs(batch_out_dir, exist_ok=True)
    logger = get_logger(batch_out_dir, filename="predict_kpd.log")

    logger.info(f"Loaded trained TCN from: {ckpt_path} (Trained Epoch: {checkpoint.get('epoch', 'N/A')}, Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f})")

    # Load All Battery Trajectories
    loaders = load_battery_dataloader(
        dataset_name=dataset_name,
        batch_name=batch_name,
        data_root=args.data_root,
        normalization_method="min-max",
        shuffle_train=False
    )

    battery_results = {}
    all_test_preds, all_test_targets = [], []
    all_knee_cycle_errors = []

    # Iterate through all splits
    for split_name in ['train', 'valid', 'test']:
        loader = loaders[split_name]
        for sample in loader:
            b_name = sample['battery_name']
            x = sample['x'].to(device)                           # [L, 17]
            true_kpd = sample['knee_distance'].cpu().numpy()     # [L, 1]
            cycles = sample['cycle_indices'].cpu().numpy()       # [L, 1]
            true_knee_cycle = sample['knee_point']               # int scalar
            num_cycles = sample['length']

            with torch.no_grad():
                pred_kpd = model(x).cpu().numpy()                # [L, 1]

            # Estimate predicted knee cycle index
            pred_knee_cycle = estimate_knee_cycle_from_kpd(cycles, pred_kpd)
            knee_cycle_err = abs(pred_knee_cycle - true_knee_cycle)
            all_knee_cycle_errors.append(knee_cycle_err)

            # Metrics for this battery
            [mae, mape, mse, rmse] = eval_metrix(pred_kpd, true_kpd)

            # Save individual battery KPD predictions to CSV
            df_out = pd.DataFrame({
                'cycle_index': cycles.flatten().astype(int),
                'true_kpd': true_kpd.flatten(),
                'pred_kpd': pred_kpd.flatten(),
                'kpd_error': (pred_kpd - true_kpd).flatten(),
                'true_knee_cycle': true_knee_cycle,
                'pred_knee_cycle': pred_knee_cycle,
                'split': split_name
            })
            csv_path = os.path.join(batch_out_dir, f"{b_name}_kpd.csv")
            df_out.to_csv(csv_path, index=False)

            # Also save numpy arrays for downstream models (DeepOPINN / Fusion MLP)
            np_path = os.path.join(batch_out_dir, f"{b_name}_pred_kpd.npy")
            np.save(np_path, pred_kpd)

            battery_results[b_name] = {
                'split': split_name,
                'cycles': num_cycles,
                'true_knee_cycle': true_knee_cycle,
                'pred_knee_cycle': pred_knee_cycle,
                'knee_cycle_error': knee_cycle_err,
                'kpd_RMSE': rmse,
                'kpd_MAE': mae,
                'kpd_MSE': mse,
                'csv_path': csv_path,
                'npy_path': np_path
            }

            if split_name == 'test':
                all_test_preds.append(pred_kpd)
                all_test_targets.append(true_kpd)

    # Compute overall test metrics for the batch
    if all_test_preds:
        test_pred_concat = np.concatenate(all_test_preds, axis=0)
        test_true_concat = np.concatenate(all_test_targets, axis=0)
        [test_mae, test_mape, test_mse, test_rmse] = eval_metrix(test_pred_concat, test_true_concat)
    else:
        test_mae, test_mape, test_mse, test_rmse = 0.0, 0.0, 0.0, 0.0

    mean_knee_err = float(np.mean(all_knee_cycle_errors))

    batch_summary = {
        'dataset': dataset_name,
        'batch': batch_name,
        'total_batteries': len(battery_results),
        'test_kpd_RMSE': test_rmse,
        'test_kpd_MAE': test_mae,
        'test_kpd_MSE': test_mse,
        'mean_knee_cycle_error': mean_knee_err,
        'battery_results': battery_results
    }

    # Save summary json
    with open(os.path.join(batch_out_dir, "batch_kpd_summary.json"), "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=4)

    logger.info(f"Completed KPD Predictions for [{dataset_name}] - [{batch_name}]:")
    logger.info(f"  Total Batteries: {len(battery_results)} (Train, Val, Test saved)")
    logger.info(f"  Test KPD RMSE:   {test_rmse:.6f} | Test KPD MAE: {test_mae:.6f}")
    logger.info(f"  Mean Knee Error: {mean_knee_err:.2f} cycles")
    logger.info(f"  Saved to:        {batch_out_dir}")

    return batch_summary


def main():
    parser = argparse.ArgumentParser(description="Predict and Save Knee Point Distance (KPD) for all batteries across 13 batches.")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "XJTU", "MIT", "TJU", "HUST"],
                        help="Dataset name ('all' for all 4 datasets, or specific: 'XJTU', 'MIT', 'TJU', 'HUST').")
    parser.add_argument("--batch", type=str, default="all",
                        help="Batch name ('all' for all batches in dataset, or specific batch).")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/TCN", help="Directory where TCN checkpoints are stored.")
    parser.add_argument("--output_dir", type=str, default="predictions/TCN_KPD", help="Directory to save KPD predictions.")
    parser.add_argument("--data_root", type=str, default="../DOPLAX-SOH-main 4/data/Processed", help="Root directory of processed data.")
    parser.add_argument("--force_train", action="store_true", help="Force re-training of TCN even if checkpoint exists.")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs if training is triggered.")
    parser.add_argument("--lr", type=float, default=None, help="LR if training is triggered.")
    parser.add_argument("--early_stop", type=int, default=None, help="Early stopping patience if training is triggered.")

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

    print("=" * 115)
    print(f"  Knee Point Distance (KPD) Prediction Pipeline: {len(tasks)} Batches Scheduled")
    print("=" * 115)
    for idx, (d, b) in enumerate(tasks, 1):
        print(f"  {idx:2d}. Dataset: {d:<6} | Batch: {b}")
    print("=" * 115)

    all_summaries = []
    total_batteries_processed = 0

    for idx, (d, b) in enumerate(tasks, 1):
        print(f"\n>>> Processing Batch {idx}/{len(tasks)}: [{d}] - [{b}] ...")
        try:
            res = predict_and_save_kpd_for_batch(d, b, args)
            all_summaries.append(res)
            total_batteries_processed += res['total_batteries']
            print(f"    [SAVED] {res['total_batteries']} batteries | Test KPD RMSE: {res['test_kpd_RMSE']:.6f} | Mean Knee Error: {res['mean_knee_cycle_error']:.1f} cycles")
        except Exception as e:
            print(f"    [FAILED] Error on [{d}] - [{b}]: {str(e)}")

    # Print Final Summary Table
    print("\n" + "=" * 115)
    print(f"{'#':<3} | {'Dataset':<6} | {'Batch Name':<28} | {'Batteries':<10} | {'Test KPD RMSE':<14} | {'Test KPD MAE':<14} | {'Mean Knee Err (cyc)':<20}")
    print("=" * 115)
    for idx, r in enumerate(all_summaries, 1):
        print(f"{idx:<3} | {r['dataset']:<6} | {r['batch']:<28} | {r['total_batteries']:<10} | {r['test_kpd_RMSE']:<14.6f} | {r['test_kpd_MAE']:<14.6f} | {r['mean_knee_cycle_error']:<20.2f}")
    print("=" * 115)
    print(f"SUMMARY: Processed and saved KPD for {total_batteries_processed} total batteries across {len(all_summaries)} batches.")
    print(f"Output files saved under: {os.path.abspath(args.output_dir)}")
    print("=" * 115)

    # Save aggregated overall summary
    summary_path = os.path.join(args.output_dir, "all_batches_kpd_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=4)


if __name__ == "__main__":
    main()
