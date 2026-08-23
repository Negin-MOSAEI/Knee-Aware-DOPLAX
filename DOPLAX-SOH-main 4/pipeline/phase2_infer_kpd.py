import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from Model.Backbones.tcn import KneeTCN
from utils.math_utils import tst_cold_start_kpd, calculate_kpd
from dataloader.dataloader import BatteryCycleDataset
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2,
    "figure.autolayout": True,
})

def infer_kpd_for_battery(model: torch.nn.Module, battery_id: str, dataset_name: str, project_root: str, device: torch.device, initial_knee_points: dict, window_size: int = 40, knee_point_cycle: int = 100) -> tuple:
    """
    Infers the full KPD sequence for a single battery.
    Uses cold start logic for cycles 1 to 39.
    Uses the trained KneeTCN model for cycles >= 40.
    """
    dataset = BatteryCycleDataset(
        battery_ids=[battery_id],
        initial_knee_points=initial_knee_points,
        dataset_name=dataset_name,
        data_root=os.path.join(project_root, 'data', 'Processed'),
        window_size=window_size
    )
    
    kpd_sequence = []
    real_kpd_sequence = []
    soh_sequence = []
    model.eval()
    with torch.no_grad():
        for cycle_idx in range(len(dataset)):
            cycle = cycle_idx + 1
            # Correctly unpack 3 elements
            features, target_kpd, target_soh = dataset[cycle_idx]
            
            if cycle < window_size:
                kpd = tst_cold_start_kpd(cycle, knee_point_cycle)
            else:
                window_tensor = features.unsqueeze(0).to(device) 
                pred_kpd = model(window_tensor)
                kpd = pred_kpd.item()
                
            kpd_sequence.append(kpd)
            real_kpd_sequence.append(target_kpd.item())
            soh_sequence.append(target_soh.item())
            
    return np.array(kpd_sequence, dtype=np.float32), np.array(real_kpd_sequence, dtype=np.float32), np.array(soh_sequence, dtype=np.float32)

def run_phase2(project_root: str):
    """Executes Phase 2: KPD Inference for all batteries."""
    print("Starting Phase 2: KneeTCN KPD Inference...")
    
    config_dir = os.path.join(project_root, 'config')
    models_dir = os.path.join(project_root, 'outputs', 'models')
    kpd_out_dir = os.path.join(project_root, 'outputs', 'kpd_predictions')
    figures_out_dir = os.path.join(project_root, 'outputs', 'figures', 'kpd_predictions')
    os.makedirs(kpd_out_dir, exist_ok=True)
    os.makedirs(figures_out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    with open(os.path.join(config_dir, 'initial_knee_points.json'), 'r') as f:
        initial_knee_points = json.load(f)
        
    for dataset, batches in train_test_split.items():
        for batch in batches:
            print(f"Inferring KPD for {dataset} - {batch}...")
            
            model_path = os.path.join(models_dir, f'tcn_{dataset}_{batch}.pt')
            if not os.path.exists(model_path):
                print(f"Warning: KneeTCN model {model_path} not found. Skipping.")
                continue
                
            train_bats = train_test_split[dataset][batch]['train']
            test_bats = train_test_split[dataset][batch]['test']
            all_bats = train_bats + test_bats
            
            if not all_bats:
                continue
                
            # Create a dataset for train_bats to calculate standardization statistics
            train_dataset = BatteryCycleDataset(battery_ids=train_bats, initial_knee_points=initial_knee_points, dataset_name=dataset, data_root=os.path.join(project_root, 'data', 'Processed'))
            if len(train_dataset) == 0:
                print(f"Warning: No training data to compute standardization stats for {dataset}-{batch}. Skipping.")
                continue
            
            all_x_tst = np.concatenate([s[0] for s in train_dataset.samples], axis=0)
            X_mean_tst = torch.tensor(np.mean(all_x_tst, axis=0), dtype=torch.float32).to(device)
            X_std_tst = torch.tensor(np.std(all_x_tst, axis=0), dtype=torch.float32).to(device)
            x_sts_tst = [X_mean_tst, X_std_tst]

            features_sample, _, _ = train_dataset[0]
            actual_num_features = features_sample.shape[-1]
            
            from utils.hpo_utils import get_model_params
            best_params = get_model_params(project_root, dataset, batch, "KneeTCN")
            if best_params:
                model = KneeTCN(num_features=actual_num_features, x_sts=x_sts_tst, **best_params).to(device)
            else:
                model = KneeTCN(num_features=actual_num_features, x_sts=x_sts_tst).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            
            batch_kpd_dir = os.path.join(kpd_out_dir, dataset, batch)
            batch_figures_dir = os.path.join(figures_out_dir, dataset, batch)
            os.makedirs(batch_kpd_dir, exist_ok=True)
            os.makedirs(batch_figures_dir, exist_ok=True)
            
            for bat_id in all_bats:
                bat_key = bat_id
                if dataset in initial_knee_points and batch in initial_knee_points[dataset] and bat_key in initial_knee_points[dataset][batch]:
                    true_knee = initial_knee_points[dataset][batch][bat_key]
                else:
                    true_knee = 100
                    
                kpd_seq, real_kpd_seq, soh_seq = infer_kpd_for_battery(model, bat_id, dataset, project_root, device, initial_knee_points, knee_point_cycle=true_knee)
                
                safe_bat_id = bat_id.replace('/', '_')
                save_path = os.path.join(batch_kpd_dir, f"{safe_bat_id}_kpd.npy")
                np.save(save_path, kpd_seq)
                
                # Plot KPD Prediction, Real KPD, and SOH
                cycles = np.arange(1, len(kpd_seq) + 1)
                
                bat_key = bat_id
                if dataset in initial_knee_points and batch in initial_knee_points[dataset] and bat_key in initial_knee_points[dataset][batch]:
                    true_knee = initial_knee_points[dataset][batch][bat_key]
                else:
                    true_knee = 100
                    
                true_kpd_seq = []
                window_size = 40
                for c in cycles:
                    if c < window_size:
                        true_kpd_seq.append(tst_cold_start_kpd(c)) 
                    else:
                        true_kpd_seq.append(calculate_kpd(c, true_knee))
                        
                fig_kpd, ax_kpd = plt.subplots(figsize=(12, 8))
                
                # Plot ground truth SOH on primary y-axis
                line1 = ax_kpd.plot(cycles, soh_seq, color='black', linewidth=2, linestyle='-', label='Ground Truth SOH')
                ax_kpd.set_xlabel('Cycle', fontsize=12, fontweight='bold')
                ax_kpd.set_ylabel('SOH (Capacity / Nominal Capacity)', fontsize=12, fontweight='bold', color='black')
                ax_kpd.tick_params(axis='y', labelcolor='black')
                
                # Create secondary y-axis for KPD
                ax_kpd2 = ax_kpd.twinx()
                
                # 1. Physics-Informed True KPD
                # line2 = ax_kpd2.plot(cycles, true_kpd_seq, color='green', linewidth=2, label='True KPD (Physics-Informed)')
                
                # 2. Predicted KPD from Phase 2
                line3 = ax_kpd2.plot(cycles, kpd_seq, color='blue', linewidth=2, linestyle='-', label='Predicted KPD (Phase 2)')
                
                # 3. TST (Cold Start) KPD
                tst_kpd = [tst_cold_start_kpd(c, true_knee) if c > 40 else tst_cold_start_kpd(c) for c in cycles]
                line4 = ax_kpd2.plot(cycles, tst_kpd, color='orange', linewidth=2, linestyle='-', label='TST (Cold Start) Baseline')
                
                # True Knee Point vertical line
                line5 = ax_kpd.axvline(x=true_knee, color='red', linestyle='-', linewidth=2, label='True Knee Point')
                
                ax_kpd2.set_ylabel('Knee Point Distance (KPD)', fontsize=12, fontweight='bold', color='blue')
                ax_kpd2.tick_params(axis='y', labelcolor='blue')
                # ax_kpd2.set_ylim(-0.1, 1.1) # KPD goes from 1 to 0
                
                split_label = "[TRAIN]" if bat_id in train_bats else "[TEST]"
                plt.title(f"{split_label} [{dataset}-{batch}] Battery {bat_key} - KPD Analysis", fontsize=14, fontweight='bold')
                    
                lines = line1  + line3 + line4 + [line5] #+ line2
                labels = [l.get_label() for l in lines]
                ax_kpd.legend(lines, labels, loc='best', fontsize=10)
                
                ax_kpd.grid(True, linestyle=':', alpha=0.6)
                
                plot_path = os.path.join(batch_figures_dir, f"{safe_bat_id}_Plot_A_KPD.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig_kpd)
                
    print("Phase 2 completed successfully. KPD sequences and plots saved.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_phase2(project_root)
