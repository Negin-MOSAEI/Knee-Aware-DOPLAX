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
from configs import get_fusion_config, get_deepopinn_config, get_lax_config, FusionMLPConfig
from Model import DeepOPINN, OptimizationNetwork, FusionMLP
from Model.utils.util import eval_metrix, get_logger, AverageMeter
from Model.utils.lr_schedulers import LR_Scheduler
from train_tcn import ALL_13_BATCHES
from train_deepopinn import train_single_deepopinn_batch, load_kpd_for_battery
from train_lax import train_single_lax_batch, compute_dataset_feature_stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained_deepopinn(dataset_name: str, batch_name: str, ckpt_dir: str, args: argparse.Namespace) -> DeepOPINN:
    """Loads pre-trained DeepOPINN or trains if missing."""
    dop_cfg = get_deepopinn_config(dataset_name)
    model = DeepOPINN(config=dop_cfg).to(device)
    
    ckpt_path = os.path.join(ckpt_dir, dataset_name, batch_name, "best_deepopinn.pth")
    if not os.path.exists(ckpt_path) or args.force_train:
        print(f"[*] DeepOPINN checkpoint not found at {ckpt_path}. Training first...")
        train_args = argparse.Namespace(
            epochs=args.epochs,
            lr=None,
            early_stop=args.early_stop,
            data_root=args.data_root,
            kpd_root=args.kpd_root,
            save_dir=ckpt_dir
        )
        train_single_deepopinn_batch(dataset_name, batch_name, train_args, config=dop_cfg)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_pretrained_lax(dataset_name: str, batch_name: str, ckpt_dir: str, train_loader, args: argparse.Namespace) -> OptimizationNetwork:
    """Loads pre-trained LAX solver or trains if missing."""
    lax_cfg = get_lax_config(dataset_name)
    x_mean, x_std = compute_dataset_feature_stats(train_loader)
    model = OptimizationNetwork(config=lax_cfg, x_sts=(x_mean, x_std)).to(device)

    ckpt_path = os.path.join(ckpt_dir, dataset_name, batch_name, "best_lax.pth")
    if not os.path.exists(ckpt_path) or args.force_train:
        print(f"[*] LAX checkpoint not found at {ckpt_path}. Training first...")
        train_args = argparse.Namespace(
            epochs=args.epochs,
            lr=None,
            early_stop=args.early_stop,
            data_root=args.data_root,
            save_dir=ckpt_dir
        )
        train_single_lax_batch(dataset_name, batch_name, train_args, config=lax_cfg)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def train_single_fusion_batch(
    dataset_name: str,
    batch_name: str,
    args: argparse.Namespace,
    config: Optional[FusionMLPConfig] = None
) -> Dict[str, Any]:
    """
    Trains and evaluates the FusionMLP network combining DeepOPINN, LAX, and TCN KPD predictions.
    """
    cfg = config or get_fusion_config(dataset_name)
    epochs = args.epochs if args.epochs is not None else cfg.epochs
    lr = args.lr if args.lr is not None else cfg.bagging_lr
    early_stop_patience = args.early_stop if args.early_stop is not None else cfg.early_stop

    # Setup directories
    batch_save_dir = os.path.join(args.save_dir, dataset_name, batch_name)
    os.makedirs(batch_save_dir, exist_ok=True)
    logger = get_logger(batch_save_dir, filename="train_fusion.log")

    logger.info("==================================================")
    logger.info(f"Starting FusionMLP Training for [{dataset_name}] - [{batch_name}]")
    logger.info(f"Device: {device} | Epochs: {epochs} | LR: {lr:.6f} | Mono Weight: {cfg.mono_bag:.4f} | Knee Feature: {cfg.include_knee_feature}")
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

    # 2. Load Pretrained Backbone Models (DeepOPINN & LAX)
    dop_model = load_pretrained_deepopinn(dataset_name, batch_name, args.deepopinn_dir, args)
    lax_model = load_pretrained_lax(dataset_name, batch_name, args.lax_dir, train_loader, args)

    # 3. Build FusionMLP Model
    fusion_model = FusionMLP(config=cfg).to(device)

    # Optimizer & Adaptive LR Scheduler
    warmup_epochs = cfg.warmup_epochs
    warmup_lr = cfg.warmup_lr
    final_lr = cfg.final_lr

    optimizer = Adam(fusion_model.parameters(), lr=lr, weight_decay=1e-5)
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
    best_ckpt_path = os.path.join(batch_save_dir, "best_fusion.pth")

    train_losses = []
    val_losses = []
    start_time = time.time()

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        fusion_model.train()
        loss_data_meter = AverageMeter()
        loss_mono_meter = AverageMeter()
        loss_total_meter = AverageMeter()

        # Step LR Scheduler
        curr_lr = scheduler.step(epoch)

        for sample in train_loader:
            x1 = sample['x1'].to(device)
            x2 = sample['x2'].to(device)
            y1 = sample['y1'].to(device)
            y2 = sample['y2'].to(device)
            b_name = sample['battery_name']

            # Load Knee Point Distance
            kpd = load_kpd_for_battery(dataset_name, batch_name, b_name, sample, args.kpd_root)
            kpd_1 = kpd[:-1] if kpd.shape[0] > x1.shape[0] else kpd
            kpd_2 = kpd[1:] if kpd.shape[0] > x2.shape[0] else kpd

            # Forward pre-trained backbones with no gradients
            with torch.no_grad():
                _, u_dop_1 = dop_model(x1)
                _, u_dop_2 = dop_model(x2)
                u_lax_1 = lax_model(x1)
                u_lax_2 = lax_model(x2)

            # Compute Fusion Loss
            loss_dict = fusion_model.compute_loss(
                u_dop_1=u_dop_1,
                u_lax_1=u_lax_1,
                y1=y1,
                u_dop_2=u_dop_2,
                u_lax_2=u_lax_2,
                y2=y2,
                kpd_1=kpd_1 if cfg.include_knee_feature else None,
                kpd_2=kpd_2 if cfg.include_knee_feature else None
            )
            total_loss = loss_dict['total_loss']

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_data_meter.update(loss_dict['loss_data'].item(), x1.size(0))
            loss_mono_meter.update(loss_dict['loss_mono'].item(), x1.size(0))
            loss_total_meter.update(total_loss.item(), x1.size(0))

        train_losses.append(loss_total_meter.avg)

        # 5. Validation Loop
        fusion_model.eval()
        val_loss_meter = AverageMeter()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for sample in valid_loader:
                x = sample['x'].to(device)
                y = sample['y'].to(device)
                b_name = sample['battery_name']
                kpd = load_kpd_for_battery(dataset_name, batch_name, b_name, sample, args.kpd_root)

                _, u_dop = dop_model(x)
                u_lax = lax_model(x)

                u_fused = fusion_model(
                    u_deepopinn=u_dop,
                    u_lax=u_lax,
                    knee_distance=kpd if cfg.include_knee_feature else None
                )

                mse_val = nn.functional.mse_loss(u_fused, y)
                val_loss_meter.update(mse_val.item(), x.size(0))

                val_preds.append(u_fused.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        val_losses.append(val_loss_meter.avg)

        if val_preds:
            val_p_cat = np.concatenate(val_preds, axis=0)
            val_t_cat = np.concatenate(val_targets, axis=0)
            [v_mae, v_mape, v_mse, v_rmse] = eval_metrix(val_p_cat, val_t_cat)
        else:
            v_mae, v_mape, v_mse, v_rmse = 0.0, 0.0, val_loss_meter.avg, np.sqrt(val_loss_meter.avg)

        # 6. Checkpoint & Best Model Selection strictly on lowest validation loss
        is_best = val_loss_meter.avg < best_val_loss
        if is_best:
            best_val_loss = val_loss_meter.avg
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': fusion_model.state_dict(),
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
                f"Train Loss: {loss_total_meter.avg:.6f} (Data: {loss_data_meter.avg:.5f}, Mono: {loss_mono_meter.avg:.5f}) | "
                f"Val MSE: {val_loss_meter.avg:.6f} (RMSE: {v_rmse:.4f}) "
                f"{'[BEST]' if is_best else ''}"
            )

        # Early Stopping
        if patience_counter >= early_stop_patience:
            logger.info(f"[*] Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val MSE: {best_val_loss:.6f}")
            break

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f}s. Loading best checkpoint from epoch {best_epoch} for final test evaluation...")

    # 7. Final Test Evaluation on Best Model Checkpoint
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    fusion_model.eval()

    test_preds, test_targets = [], []
    battery_test_metrics = {}

    with torch.no_grad():
        for sample in test_loader:
            b_name = sample['battery_name']
            x = sample['x'].to(device)
            y = sample['y'].to(device)
            kpd = load_kpd_for_battery(dataset_name, batch_name, b_name, sample, args.kpd_root)

            _, u_dop = dop_model(x)
            u_lax = lax_model(x)

            u_fused = fusion_model(
                u_deepopinn=u_dop,
                u_lax=u_lax,
                knee_distance=kpd if cfg.include_knee_feature else None
            )

            pred_np = u_fused.cpu().numpy()
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
    parser = argparse.ArgumentParser(description="Train Multi-Modal FusionMLP (DeepOPINN + LAX + KPD) across 13 benchmark batches.")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "XJTU", "MIT", "TJU", "HUST"],
                        help="Dataset to train ('all' for all datasets, or specific: 'XJTU', 'MIT', 'TJU', 'HUST').")
    parser.add_argument("--batch", type=str, default="all",
                        help="Batch to train ('all' for all batches in dataset, or specific batch).")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--early_stop", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument("--data_root", type=str, default="../DOPLAX-SOH-main 4/data/Processed", help="Root directory of processed data.")
    parser.add_argument("--kpd_root", type=str, default="predictions/TCN_KPD", help="Root directory containing predicted KPD files.")
    parser.add_argument("--deepopinn_dir", type=str, default="checkpoints/DeepOPINN", help="Directory containing DeepOPINN checkpoints.")
    parser.add_argument("--lax_dir", type=str, default="checkpoints/LAX", help="Directory containing LAX checkpoints.")
    parser.add_argument("--save_dir", type=str, default="checkpoints/FusionMLP", help="Directory to save FusionMLP checkpoints.")
    parser.add_argument("--force_train", action="store_true", help="Force retraining of upstream models if requested.")

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
    print(f"  Multi-Modal FusionMLP Training Pipeline: {len(tasks)} Batches Scheduled")
    print("=" * 115)
    for idx, (d, b) in enumerate(tasks, 1):
        print(f"  {idx:2d}. Dataset: {d:<6} | Batch: {b}")
    print("=" * 115)

    all_results = []
    for idx, (d, b) in enumerate(tasks, 1):
        print(f"\n>>> Training Batch {idx}/{len(tasks)}: [{d}] - [{b}] ...")
        try:
            res = train_single_fusion_batch(d, b, args)
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
    summary_path = os.path.join(args.save_dir, "all_batches_fusion_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"Overall FusionMLP summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
