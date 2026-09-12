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
from configs import get_deepopinn_config, DeepOPINNConfig
from Model import DeepOPINN
from Model.utils.util import eval_metrix, get_logger, AverageMeter
from Model.utils.lr_schedulers import LR_Scheduler
from train_tcn import ALL_13_BATCHES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_kpd_for_battery(
    dataset_name: str,
    batch_name: str,
    battery_name: str,
    sample: Dict[str, Any],
    kpd_root: str = "predictions/TCN_KPD"
) -> torch.Tensor:
    """
    Loads predicted Knee Point Distance (KPD) for a battery from TCN predictions.
    Falls back to sample ground-truth knee distance if prediction file is absent.
    """
    clean_name = battery_name[:-4] if battery_name.endswith('.csv') else battery_name
    pred_path = os.path.join(kpd_root, dataset_name, batch_name, f"{clean_name}_pred_kpd.npy")
    
    if os.path.exists(pred_path):
        try:
            kpd_arr = np.load(pred_path).astype(np.float32)
            return torch.from_numpy(kpd_arr).to(device)
        except Exception:
            pass
    
    # Fallback to dataset sample knee distance
    return sample['knee_distance'].to(device)


def train_single_deepopinn_batch(
    dataset_name: str,
    batch_name: str,
    args: argparse.Namespace,
    config: Optional[DeepOPINNConfig] = None
) -> Dict[str, Any]:
    """
    Trains and evaluates a DeepOPINN model for one dataset batch (e.g. XJTU 2C).
    Applies knee-aware PDE residual loss: (f_residual^2) * max(0, kpd) + monotonicity loss.
    """
    cfg = config or get_deepopinn_config(dataset_name)
    epochs = args.epochs if args.epochs is not None else cfg.epochs
    lr = args.lr if args.lr is not None else cfg.lr
    early_stop_patience = args.early_stop if args.early_stop is not None else cfg.early_stop

    # Setup directories
    batch_save_dir = os.path.join(args.save_dir, dataset_name, batch_name)
    os.makedirs(batch_save_dir, exist_ok=True)
    logger = get_logger(batch_save_dir, filename="train_deepopinn.log")

    logger.info("==================================================")
    logger.info(f"Starting DeepOPINN Training for [{dataset_name}] - [{batch_name}]")
    logger.info(f"Device: {device} | Epochs: {epochs} | LR: {lr:.6f} | Alpha (PDE): {cfg.alpha:.5f} | Beta (Mono): {cfg.beta:.5f}")
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

    # 2. Build DeepOPINN Model
    model = DeepOPINN(config=cfg).to(device)

    # Optimizers & Adaptive LR Scheduler
    warmup_epochs = cfg.warmup_epochs
    warmup_lr = cfg.warmup_lr
    final_lr = cfg.final_lr
    lr_F = cfg.lr_F

    optimizer_solution = Adam(model.solution_u.parameters(), lr=warmup_lr)
    optimizer_extractor = Adam(model.extractor.parameters(), lr=warmup_lr * 0.1)
    optimizer_F = Adam(model.dynamical_F.parameters(), lr=lr_F)

    scheduler = LR_Scheduler(
        optimizer=optimizer_solution,
        warmup_epochs=warmup_epochs,
        warmup_lr=warmup_lr,
        num_epochs=epochs,
        base_lr=lr,
        final_lr=final_lr
    )

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_ckpt_path = os.path.join(batch_save_dir, "best_deepopinn.pth")

    train_losses = []
    val_losses = []
    start_time = time.time()

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        loss_data_meter = AverageMeter()
        loss_pde_meter = AverageMeter()
        loss_mono_meter = AverageMeter()
        loss_total_meter = AverageMeter()

        # Step LR Scheduler
        curr_lr = scheduler.step(epoch)
        for pg in optimizer_extractor.param_groups:
            pg['lr'] = curr_lr * 0.1

        for sample in train_loader:
            x1 = sample['x1'].to(device)
            x2 = sample['x2'].to(device)
            y1 = sample['y1'].to(device)
            y2 = sample['y2'].to(device)
            b_name = sample['battery_name']

            # Load Knee Point Distance (KPD) for knee-aware PDE weighting
            kpd = load_kpd_for_battery(dataset_name, batch_name, b_name, sample, args.kpd_root)
            kpd1 = kpd[:-1] if kpd.shape[0] > x1.shape[0] else kpd

            # Forward & Compute Knee-Aware Loss
            loss_dict = model.compute_loss(
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                kpd1=kpd1
            )
            total_loss = loss_dict['total_loss']

            optimizer_solution.zero_grad()
            optimizer_extractor.zero_grad()
            optimizer_F.zero_grad()

            total_loss.backward()

            optimizer_solution.step()
            optimizer_extractor.step()
            optimizer_F.step()

            loss_data_meter.update(loss_dict['loss_data'].item(), x1.size(0))
            loss_pde_meter.update(loss_dict['loss_pde'].item(), x1.size(0))
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
                _, u_pred = model(x)

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
                'extractor_state_dict': model.extractor.state_dict(),
                'solution_u_state_dict': model.solution_u.state_dict(),
                'dynamical_F_state_dict': model.dynamical_F.state_dict(),
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
                f"Train Loss: {loss_total_meter.avg:.6f} (Data: {loss_data_meter.avg:.5f}, PDE*kpd: {loss_pde_meter.avg:.5f}, Mono: {loss_mono_meter.avg:.5f}) | "
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
            _, u_pred = model(x)

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
    parser = argparse.ArgumentParser(description="Train Knee-Aware DeepOPINN operator network across 13 benchmark batches.")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "XJTU", "MIT", "TJU", "HUST"],
                        help="Dataset to train ('all' for all datasets, or specific: 'XJTU', 'MIT', 'TJU', 'HUST').")
    parser.add_argument("--batch", type=str, default="all",
                        help="Batch to train ('all' for all batches in dataset, or specific batch).")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--early_stop", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument("--data_root", type=str, default="../DOPLAX-SOH-main 4/data/Processed", help="Root directory of processed data.")
    parser.add_argument("--kpd_root", type=str, default="predictions/TCN_KPD", help="Root directory containing predicted KPD files.")
    parser.add_argument("--save_dir", type=str, default="checkpoints/DeepOPINN", help="Directory to save DeepOPINN checkpoints.")

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
    print(f"  Knee-Aware DeepOPINN Training Pipeline: {len(tasks)} Batches Scheduled")
    print("=" * 115)
    for idx, (d, b) in enumerate(tasks, 1):
        print(f"  {idx:2d}. Dataset: {d:<6} | Batch: {b}")
    print("=" * 115)

    all_results = []
    for idx, (d, b) in enumerate(tasks, 1):
        print(f"\n>>> Training Batch {idx}/{len(tasks)}: [{d}] - [{b}] ...")
        try:
            res = train_single_deepopinn_batch(d, b, args)
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
    summary_path = os.path.join(args.save_dir, "all_batches_deepopinn_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"Overall DeepOPINN summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
