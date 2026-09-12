import os
import sys
import time
import json
import argparse
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader import load_battery_dataloader
from configs import get_tcn_config, TCNConfig
from Model import TemporalConvNet
from Model.utils.util import eval_metrix, get_logger, AverageMeter
from Model.utils.lr_schedulers import LR_Scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All 13 canonical batches across the 4 datasets
ALL_13_BATCHES = [
    ("HUST", "default"),
    ("MIT", "2017-05-12"),
    ("MIT", "2017-06-30"),
    ("MIT", "2018-04-12"),
    ("TJU", "Dataset_1_NCA_battery"),
    ("TJU", "Dataset_2_NCM_battery"),
    ("TJU", "Dataset_3_NCM_NCA_battery"),
    ("XJTU", "2C"),
    ("XJTU", "3C"),
    ("XJTU", "R2.5"),
    ("XJTU", "R3"),
    ("XJTU", "RW"),
    ("XJTU", "Sim_satellite"),
]


def train_single_battery_batch(
    dataset_name: str,
    batch_name: str,
    args: argparse.Namespace,
    config: Optional[TCNConfig] = None
) -> Dict[str, Any]:
    """
    Trains and evaluates a TCN model for one dataset batch (e.g. XJTU 2C).
    """
    cfg = config or get_tcn_config(dataset_name)
    epochs = args.epochs if args.epochs is not None else cfg.epochs
    lr = args.lr if args.lr is not None else cfg.lr
    early_stop_patience = args.early_stop if args.early_stop is not None else cfg.early_stop

    # Setup directories
    batch_save_dir = os.path.join(args.save_dir, dataset_name, batch_name)
    os.makedirs(batch_save_dir, exist_ok=True)
    logger = get_logger(batch_save_dir, filename="train_tcn.log")

    logger.info("==================================================")
    logger.info(f"Starting TCN Training for [{dataset_name}] - [{batch_name}]")
    logger.info(f"Device: {device} | Epochs: {epochs} | LR: {lr} | EarlyStop: {early_stop_patience}")
    logger.info("==================================================")

    # 1. Load Data
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

    # 2. Build TCN Model
    model = TemporalConvNet(
        num_inputs=cfg.num_inputs,
        num_channels=cfg.num_channels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
        output_dim=cfg.output_dim
    ).to(device)

    warmup_epochs = getattr(cfg, 'warmup_epochs', 20)
    warmup_lr = getattr(cfg, 'warmup_lr', lr * 0.1)
    final_lr = getattr(cfg, 'final_lr', 1e-6)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = LR_Scheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        warmup_lr=warmup_lr,
        num_epochs=epochs,
        base_lr=lr,
        final_lr=final_lr
    )

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []

    # 3. Training Loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_meter = AverageMeter()

        for sample in train_loader:
            x = sample['x'].to(device)                            # [L, 17]
            target_knee_dist = sample['knee_distance'].to(device)  # [L, 1]

            optimizer.zero_grad()
            pred_knee_dist = model(x)                              # [L, 1]
            loss = criterion(pred_knee_dist, target_knee_dist)
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item(), n=x.size(0))

        # Adaptive LR step with Warmup and Cosine Annealing (from last project)
        current_lr = scheduler.step(epoch)
        train_losses.append(train_loss_meter.avg)

        # 4. Validation (Evaluate on Validation Dataset)
        model.eval()
        val_loss_meter = AverageMeter()
        with torch.no_grad():
            for sample in valid_loader:
                x = sample['x'].to(device)
                target_knee_dist = sample['knee_distance'].to(device)
                pred_knee_dist = model(x)
                v_loss = criterion(pred_knee_dist, target_knee_dist)
                val_loss_meter.update(v_loss.item(), n=x.size(0))

        current_val_loss = val_loss_meter.avg
        val_losses.append(current_val_loss)

        # Strictly select the best model by the lowest loss on validation dataset
        if current_val_loss < best_val_loss:
            prev_best = best_val_loss
            best_val_loss = current_val_loss
            best_epoch = epoch
            # Store deep copy of model state at minimum validation loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0

            # Save best checkpoint to disk
            ckpt_path = os.path.join(batch_save_dir, "best_tcn.pth")
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'val_loss': best_val_loss,
                'dataset': dataset_name,
                'batch': batch_name,
                'config': cfg.__dict__
            }, ckpt_path)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch [{epoch:4d}/{epochs}] - LR: {current_lr:.6f} | New best validation loss: {best_val_loss:.6f} (improved from {prev_best:.6f}) -> Saved model")
        else:
            patience_counter += 1

        if epoch % 20 == 0 and epoch != 1:
            logger.info(f"Epoch [{epoch:4d}/{epochs}] - LR: {current_lr:.6f} | Train Loss: {train_loss_meter.avg:.6f} | Val Loss: {current_val_loss:.6f} (Best Val Loss: {best_val_loss:.6f} @ Ep {best_epoch})")

        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (Validation loss did not improve for {early_stop_patience} epochs).")
            break

    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.2f}s. Best Model selected from Epoch {best_epoch} with lowest Validation Loss: {best_val_loss:.6f}")

    # 5. Testing with Best Model
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()

    test_preds, test_targets = [], []
    battery_kpis = {}

    with torch.no_grad():
        for sample in test_loader:
            b_name = sample['battery_name']
            x = sample['x'].to(device)
            target = sample['knee_distance'].cpu().numpy()
            pred = model(x).cpu().numpy()

            test_preds.append(pred)
            test_targets.append(target)

            # Per battery KPI
            [mae, mape, mse, rmse] = eval_metrix(pred, target)
            battery_kpis[b_name] = {'MAE': mae, 'MAPE': mape, 'MSE': mse, 'RMSE': rmse}

    all_preds = np.concatenate(test_preds, axis=0)
    all_targets = np.concatenate(test_targets, axis=0)
    [test_mae, test_mape, test_mse, test_rmse] = eval_metrix(all_preds, all_targets)

    logger.info("--------------------------------------------------")
    logger.info(f"Test Evaluation Results for [{dataset_name}] - [{batch_name}]:")
    logger.info(f"  MSE:  {test_mse:.6f}")
    logger.info(f"  RMSE: {test_rmse:.6f}")
    logger.info(f"  MAE:  {test_mae:.6f}")
    logger.info(f"  MAPE: {test_mape:.6f}%")
    logger.info("--------------------------------------------------")

    # Save summary metrics
    results_summary = {
        'dataset': dataset_name,
        'batch': batch_name,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_MSE': test_mse,
        'test_RMSE': test_rmse,
        'test_MAE': test_mae,
        'test_MAPE': test_mape,
        'battery_kpis': battery_kpis,
        'train_loss_history': train_losses,
        'valid_loss_history': val_losses,
        'elapsed_seconds': elapsed_time
    }

    with open(os.path.join(batch_save_dir, "test_kpis.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4)

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Train TCN Model for Knee Distance Prediction across all 13 batches.")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "XJTU", "MIT", "TJU", "HUST"],
                        help="Dataset name ('all' to train all datasets, or specific dataset: 'XJTU', 'MIT', 'TJU', 'HUST').")
    parser.add_argument("--batch", type=str, default="all",
                        help="Batch name ('all' for all batches in dataset, or specific batch: '2C', '2017-05-12', etc.).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: from config).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: from config).")
    parser.add_argument("--early_stop", type=int, default=None, help="Early stopping patience (default: from config).")
    parser.add_argument("--data_root", type=str, default="../DOPLAX-SOH-main 4/data/Processed", help="Root directory for processed data.")
    parser.add_argument("--save_dir", type=str, default="checkpoints/TCN", help="Output directory to save checkpoints and logs.")

    args = parser.parse_args()

    # Determine which batches to run
    tasks_to_run: List[Tuple[str, str]] = []
    
    if args.dataset == "all":
        tasks_to_run = ALL_13_BATCHES
    else:
        target_d = args.dataset.upper()
        if args.batch == "all":
            tasks_to_run = [(d, b) for (d, b) in ALL_13_BATCHES if d.upper() == target_d]
        else:
            tasks_to_run = [(target_d, args.batch)]

    print("=" * 95)
    print(f"  TCN Knee Distance Training Plan: {len(tasks_to_run)} Batches Scheduled to Train")
    print("=" * 95)
    for idx, (d, b) in enumerate(tasks_to_run, 1):
        print(f"  {idx:2d}. Dataset: {d:<6} | Batch: {b}")
    print("=" * 95)

    all_results = []
    
    for idx, (d, b) in enumerate(tasks_to_run, 1):
        print(f"\n>>> Running Batch {idx}/{len(tasks_to_run)}: [{d}] - [{b}] ...")
        try:
            res = train_single_battery_batch(d, b, args)
            all_results.append(res)
            print(f"    [COMPLETED] Best Ep: {res['best_epoch']} | Test RMSE: {res['test_RMSE']:.6f} | Test MAE: {res['test_MAE']:.6f}")
        except Exception as e:
            print(f"    [FAILED] Error training [{d}] - [{b}]: {str(e)}")

    # Print Final Summary Table
    print("\n" + "=" * 95)
    print(f"{'#':<3} | {'Dataset':<6} | {'Batch Name':<28} | {'Best Ep':<8} | {'Test MSE':<10} | {'Test RMSE':<10} | {'Test MAE':<10}")
    print("=" * 95)
    for idx, r in enumerate(all_results, 1):
        print(f"{idx:<3} | {r['dataset']:<6} | {r['batch']:<28} | {r['best_epoch']:<8} | {r['test_MSE']:<10.6f} | {r['test_RMSE']:<10.6f} | {r['test_MAE']:<10.6f}")
    print("=" * 95)

    # Save overall summary
    summary_path = os.path.join(args.save_dir, "all_batches_tcn_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll TCN training results saved to: {summary_path}")


if __name__ == "__main__":
    main()
