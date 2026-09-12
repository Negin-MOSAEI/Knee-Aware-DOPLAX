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
from torch.optim import Adam

from dataloader import load_battery_dataloader
from configs import get_lax_config, LAXConfig
from Model import OptimizationNetwork
from Model.utils.util import eval_metrix, get_logger, AverageMeter
from Model.utils.lr_schedulers import LR_Scheduler
from train_tcn import ALL_13_BATCHES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_dataset_feature_stats(train_loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes feature-wise mean and standard deviation from the training loader.
    """
    sum_feat = None
    sum_sq_feat = None
    total_samples = 0

    for sample in train_loader:
        # Extract features (first 16 dimensions without time metadata)
        x = sample['x'][:, :16].to(device)
        if sum_feat is None:
            sum_feat = torch.zeros(x.shape[1], device=device)
            sum_sq_feat = torch.zeros(x.shape[1], device=device)

        sum_feat += x.sum(dim=0)
        sum_sq_feat += (x ** 2).sum(dim=0)
        total_samples += x.size(0)

    mean = sum_feat / max(1, total_samples)
    std = torch.sqrt(torch.clamp((sum_sq_feat / max(1, total_samples)) - (mean ** 2), min=1e-6))
    return mean.cpu(), std.cpu()


def train_single_lax_batch(
    dataset_name: str,
    batch_name: str,
    args: argparse.Namespace,
    config: Optional[LAXConfig] = None
) -> Dict[str, Any]:
    """
    Trains and evaluates an OptimizationNetwork (LAX solver) model for one dataset batch (e.g. XJTU 2C).
    """
    cfg = config or get_lax_config(dataset_name)
    epochs = args.epochs if args.epochs is not None else cfg.epochs
    lr = args.lr if args.lr is not None else cfg.lr_net
    early_stop_patience = args.early_stop if args.early_stop is not None else cfg.patience

    # Setup directories
    batch_save_dir = os.path.join(args.save_dir, dataset_name, batch_name)
    os.makedirs(batch_save_dir, exist_ok=True)
    logger = get_logger(batch_save_dir, filename="train_lax.log")

    logger.info("==================================================")
    logger.info(f"Starting LAX Solver Training for [{dataset_name}] - [{batch_name}]")
    logger.info(f"Device: {device} | Epochs: {epochs} | LR: {lr:.6f} | Mono Weight: {cfg.betha_LAX:.5f} | Center: {cfg.center_block}")
    logger.info("==================================================")

    # 1. Load DataLoaders
    loaders = load_battery_dataloader(
        dataset_name=dataset_name,
        batch_name=batch_name,
        data_root=args.data_root,
        normalization_method="min-max",
        shuffle_train=True
    )
    train_loader = loaders['train']
    valid_loader = loaders['valid']
    test_loader = loaders['test']

    logger.info(f"Data Splits: Train={len(loaders['train_dataset'])}, Val={len(loaders['valid_dataset'])}, Test={len(loaders['test_dataset'])}")

    # Compute Feature Statistics for Normalization
    x_mean, x_std = compute_dataset_feature_stats(train_loader)

    # 2. Build LAX Model
    model = OptimizationNetwork(
        config=cfg,
        x_sts=(x_mean, x_std)
    ).to(device)

    # Optimizer & Adaptive LR Scheduler
    warmup_epochs = cfg.warmup_epochs_net
    warmup_lr = lr * 0.1
    final_lr = cfg.min_lr_net

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = LR_Scheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        warmup_lr=warmup_lr,
        num_epochs=epochs,
        base_lr=lr,
        final_lr=final_lr
    )

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_ckpt_path = os.path.join(batch_save_dir, "best_lax.pth")

    train_losses = []
    val_losses = []
    start_time = time.time()

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        loss_mse_meter = AverageMeter()
        loss_mono_meter = AverageMeter()
        loss_total_meter = AverageMeter()

        # Step LR Scheduler
        curr_lr = scheduler.step(epoch)

        for sample in train_loader:
            x1 = sample['x1'].to(device)
            x2 = sample['x2'].to(device)
            y1 = sample['y1'].to(device)
            y2 = sample['y2'].to(device)

            loss_dict = model.compute_loss(
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2
            )
            total_loss = loss_dict['total_loss']

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_mse_meter.update(loss_dict['loss_mse'].item(), x1.size(0))
            loss_mono_meter.update(loss_dict['loss_mono'].item(), x1.size(0))
            loss_total_meter.update(total_loss.item(), x1.size(0))

        train_losses.append(loss_total_meter.avg)

        # 4. Validation Loop
        model.eval()
        val_loss_meter = AverageMeter()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for sample in valid_loader:
                x = sample['x'].to(device)
                y = sample['y'].to(device)
                u_pred = model(x)

                mse_val = nn.functional.mse_loss(u_pred, y)
                val_loss_meter.update(mse_val.item(), x.size(0))

                val_preds.append(u_pred.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        val_losses.append(val_loss_meter.avg)

        if val_preds:
            val_p_cat = np.concatenate(val_preds, axis=0)
            val_t_cat = np.concatenate(val_targets, axis=0)
            [v_mae, v_mape, v_mse, v_rmse] = eval_metrix(val_p_cat, val_t_cat)
        else:
            v_mae, v_mape, v_mse, v_rmse = 0.0, 0.0, val_loss_meter.avg, np.sqrt(val_loss_meter.avg)

        # 5. Checkpoint & Best Model Selection strictly on lowest validation loss
        is_best = val_loss_meter.avg < best_val_loss
        if is_best:
            best_val_loss = val_loss_meter.avg
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_rmse': v_rmse,
                'val_mae': v_mae,
                'val_mape': v_mape,
                'config': cfg.__dict__
            }, best_ckpt_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or is_best or epoch == 1:
            logger.info(
                f"Epoch [{epoch:4d}/{epochs:4d}] LR: {curr_lr:.6f} | "
                f"Train Loss: {loss_total_meter.avg:.6f} (MSE: {loss_mse_meter.avg:.5f}, Mono: {loss_mono_meter.avg:.5f}) | "
                f"Val MSE: {val_loss_meter.avg:.6f} (RMSE: {v_rmse:.4f}) "
                f"{'[BEST]' if is_best else ''}"
            )

        # Early Stopping
        if patience_counter >= early_stop_patience:
            logger.info(f"[*] Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val MSE: {best_val_loss:.6f}")
            break

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f}s. Loading best checkpoint from epoch {best_epoch} for final test evaluation...")

    # 6. Final Test Evaluation on Best Model Checkpoint
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds, test_targets = [], []
    battery_test_metrics = {}

    with torch.no_grad():
        for sample in test_loader:
            b_name = sample['battery_name']
            x = sample['x'].to(device)
            y = sample['y'].to(device)
            u_pred = model(x)

            pred_np = u_pred.cpu().numpy()
            true_np = y.cpu().numpy()

            [b_mae, b_mape, b_mse, b_rmse] = eval_metrix(pred_np, true_np)
            battery_test_metrics[b_name] = {
                'RMSE': b_rmse,
                'MAE': b_mae,
                'MAPE': b_mape,
                'MSE': b_mse
            }

            test_preds.append(pred_np)
            test_targets.append(true_np)

    if test_preds:
        test_p_cat = np.concatenate(test_preds, axis=0)
        test_t_cat = np.concatenate(test_targets, axis=0)
        [test_mae, test_mape, test_mse, test_rmse] = eval_metrix(test_p_cat, test_t_cat)
    else:
        test_mae, test_mape, test_mse, test_rmse = 0.0, 0.0, 0.0, 0.0

    logger.info("==================================================")
    logger.info(f"Test Results for [{dataset_name}] - [{batch_name}] (Best Epoch {best_epoch}):")
    logger.info(f"  Test RMSE: {test_rmse:.6f}")
    logger.info(f"  Test MAE:  {test_mae:.6f}")
    logger.info(f"  Test MAPE: {test_mape:.6f}")
    logger.info(f"  Test MSE:  {test_mse:.6f}")
    logger.info("==================================================")

    # Save summary metrics
    summary = {
        'dataset': dataset_name,
        'batch': batch_name,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_RMSE': test_rmse,
        'test_MAE': test_mae,
        'test_MAPE': test_mape,
        'test_MSE': test_mse,
        'battery_metrics': battery_test_metrics,
        'checkpoint_path': best_ckpt_path,
        'elapsed_seconds': elapsed
    }

    with open(os.path.join(batch_save_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train LAX Hopf-Lax solver network across 13 benchmark batches.")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "XJTU", "MIT", "TJU", "HUST"],
                        help="Dataset to train ('all' for all datasets, or specific: 'XJTU', 'MIT', 'TJU', 'HUST').")
    parser.add_argument("--batch", type=str, default="all",
                        help="Batch to train ('all' for all batches in dataset, or specific batch).")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--early_stop", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument("--data_root", type=str, default="../DOPLAX-SOH-main 4/data/Processed", help="Root directory of processed data.")
    parser.add_argument("--save_dir", type=str, default="checkpoints/LAX", help="Directory to save LAX checkpoints.")

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
    print(f"  LAX Hopf-Lax Solver Training Pipeline: {len(tasks)} Batches Scheduled")
    print("=" * 115)
    for idx, (d, b) in enumerate(tasks, 1):
        print(f"  {idx:2d}. Dataset: {d:<6} | Batch: {b}")
    print("=" * 115)

    all_results = []
    for idx, (d, b) in enumerate(tasks, 1):
        print(f"\n>>> Training Batch {idx}/{len(tasks)}: [{d}] - [{b}] ...")
        try:
            res = train_single_lax_batch(d, b, args)
            all_results.append(res)
            print(f"    [COMPLETED] Test RMSE: {res['test_RMSE']:.6f} | Test MAE: {res['test_MAE']:.6f} | Best Epoch: {res['best_epoch']}")
        except Exception as e:
            print(f"    [FAILED] Error on [{d}] - [{b}]: {str(e)}")

    # Print Final Summary Table
    print("\n" + "=" * 115)
    print(f"{'#':<3} | {'Dataset':<6} | {'Batch Name':<28} | {'Best Epoch':<12} | {'Test RMSE':<14} | {'Test MAE':<14} | {'Test MAPE':<12}")
    print("=" * 115)
    for idx, r in enumerate(all_results, 1):
        print(f"{idx:<3} | {r['dataset']:<6} | {r['batch']:<28} | {r['best_epoch']:<12} | {r['test_RMSE']:<14.6f} | {r['test_MAE']:<14.6f} | {r['test_MAPE']:<12.6f}")
    print("=" * 115)

    # Save overall summary
    summary_path = os.path.join(args.save_dir, "all_batches_lax_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"Overall LAX summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
