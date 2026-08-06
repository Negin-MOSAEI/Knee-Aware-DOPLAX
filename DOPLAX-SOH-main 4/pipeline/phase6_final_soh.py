import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy.signal import savgol_filter

from Model.Backbones.tst import TimeSeriesTransformer
from Model.PI_nets.DeepOPINN import Model as DeepOpinn
from Model.Backbones.bagging_mlp import BaggingMLP
from dataloader.dataloader import BatteryCycleDataset

DATASETS = {
    'HUST': ['default'],
    'MIT': ['default'],
    'TJU': ['NCA', 'NCM', 'NCA_NCM'],
    'XJTU': ['2C', '3C', 'R2.5', 'R3', 'RW', 'sim_satellite']
}

def apply_kadoplax_fusion(deepopinn_soh: np.ndarray, bagging_mlp_soh: np.ndarray, kpd: np.ndarray) -> np.ndarray:
    """
    Knee-aware DOPLAX fusion.
    Uses KPD to weight the predictions: Before knee, lean towards Bagging MLP. After knee, lean towards DeepOpinn.
    Since KPD is now positive before the knee and negative after, we invert it for the sine weight.
    """
    weight = 0.5 * (np.sin(-kpd) + 1.0) # roughly 0 before knee, 1 after knee
    return weight * deepopinn_soh + (1 - weight) * bagging_mlp_soh

def run_phase6(project_root: str):
    """Executes Phase 6: Final SOH Inference & Plotting."""
    print("Starting Phase 6: Final SOH Estimation & Plotting...")
    
    config_dir = os.path.join(project_root, 'config')
    models_dir = os.path.join(project_root, 'outputs', 'models')
    kpd_out_dir = os.path.join(project_root, 'outputs', 'kpd_predictions')
    figures_dir = os.path.join(project_root, 'outputs', 'figures', 'final_soh')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    for dataset, batches in train_test_split.items():
        for batch in batches:
            print(f"Generating SOH plots for {dataset} - {batch}...")
            batch_fig_dir = os.path.join(figures_dir, dataset, batch)
            os.makedirs(batch_fig_dir, exist_ok=True)
            
            all_bats = train_test_split[dataset][batch]['train'] + train_test_split[dataset][batch]['test']
            train_set = set(train_test_split[dataset][batch]['train'])
            
            if not all_bats:
                continue
                
            peek_dataset = BatteryCycleDataset([all_bats[0]], {all_bats[0]: 0}, dataset, project_root)
            if len(peek_dataset) == 0:
                print(f"Warning: No data to peek num_features for {dataset}-{batch}. Skipping.")
                continue
            features_sample, _, _ = peek_dataset[0]
            actual_num_features = features_sample.shape[-1]
            
            # Load Models
            try:
                deepopinn = DeepOpinn(input_dim=actual_num_features).to(device)
                deepopinn.load_state_dict(torch.load(os.path.join(models_dir, f'deepopinn_{dataset}_{batch}.pt'), map_location=device, weights_only=True))
                deepopinn.eval()
                
                bagging_mlp = BaggingMLP(num_estimators=5, input_dim=actual_num_features).to(device)
                bagging_mlp.load_state_dict(torch.load(os.path.join(models_dir, f'bagging_mlp_{dataset}_{batch}.pt'), map_location=device, weights_only=True))
                bagging_mlp.eval()
            except Exception as e:
                print(f"Skipping {dataset}-{batch} due to missing models: {e}")
                continue
            
            with torch.no_grad():
                for bat_id in all_bats:
                    is_train = bat_id in train_set
                    kpd_path = os.path.join(kpd_out_dir, dataset, batch, f"{bat_id.replace('/', '_')}_kpd.npy")
                    
                    if not os.path.exists(kpd_path):
                        continue
                        
                    kpd_seq = np.load(kpd_path)
                    
                    # Correctly load the real dataset for inference to get features and true_soh
                    dataset_obj = BatteryCycleDataset(
                        battery_ids=[bat_id], 
                        initial_knee_points={}, 
                        dataset_name=dataset, 
                        project_root=project_root
                    )
                    
                    # Extract full sequence
                    features_list = []
                    true_soh_list = []
                    for i in range(len(dataset_obj)):
                        f, _, s = dataset_obj[i]
                        features_list.append(f)
                        true_soh_list.append(s.item())
                        
                    true_soh = np.array(true_soh_list, dtype=np.float32)
                    features = torch.stack(features_list).to(device)
                    num_cycles = len(true_soh)
                    
                    # Inference
                    current_features = features[:, -1, :]
                    deepopinn_soh = deepopinn(current_features).cpu().numpy().flatten()
                    bagging_soh = bagging_mlp(current_features).cpu().numpy().flatten()
                    
                    # Fusion
                    # Use min(num_cycles, len(kpd_seq)) to avoid shape mismatches
                    valid_len = min(num_cycles, len(kpd_seq))
                    kadoplax_soh = apply_kadoplax_fusion(deepopinn_soh[:valid_len], bagging_soh[:valid_len], kpd_seq[:valid_len])
                    
                    # Post-processing (Savitzky-Golay filter)
                    window_length = min(51, valid_len if valid_len % 2 != 0 else valid_len - 1)
                    if window_length > 3:
                        post_kadoplax_soh = savgol_filter(kadoplax_soh, window_length, 3)
                    else:
                        post_kadoplax_soh = kadoplax_soh
                    
                    # Plotting
                    fig, ax = plt.subplots(figsize=(12, 7))
                    bg_color = '#eaffea' if is_train else '#ffeaea'
                    ax.set_facecolor(bg_color)
                    
                    cycles = np.arange(1, valid_len + 1)
                    ax.plot(cycles, true_soh[:valid_len], label='Ground Truth', color='black', linewidth=2, linestyle='--')
                    ax.plot(cycles, bagging_soh[:valid_len], label='Lax (Bagging MLP)', color='orange', alpha=0.7)
                    ax.plot(cycles, deepopinn_soh[:valid_len], label='DeepOpinn', color='blue', alpha=0.7)
                    ax.plot(cycles, kadoplax_soh, label='KaDOPLAX', color='purple', alpha=0.8)
                    ax.plot(cycles, post_kadoplax_soh, label='Post-processed KaDOPLAX', color='green', linewidth=2)
                    
                    split_label = "[TRAIN]" if is_train else "[TEST]"
                    plt.title(f'Final SOH Estimation {split_label} - {dataset} ({batch}) - {bat_id}', fontsize=16, fontweight='bold')
                    plt.xlabel('Cycle', fontsize=14)
                    plt.ylabel('State of Health (SOH)', fontsize=14)
                    plt.grid(True, linestyle=':', alpha=0.6)
                    plt.legend(loc='best', fontsize=12)
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(batch_fig_dir, f"battery_{bat_id.replace('/', '_')}.png"), dpi=300)
                    plt.close()
                    
                    # Save the sequences for phase 7
                    out_preds_dir = os.path.join(project_root, 'outputs', 'predictions', dataset, batch)
                    os.makedirs(out_preds_dir, exist_ok=True)
                    np.savez(os.path.join(out_preds_dir, f"{bat_id.replace('/', '_')}_preds.npz"),
                             true_soh=true_soh[:valid_len], lax=bagging_soh[:valid_len], deepopinn=deepopinn_soh[:valid_len], 
                             kadoplax=kadoplax_soh, post_kadoplax=post_kadoplax_soh, kpd=kpd_seq[:valid_len])

    print("Phase 6 completed successfully.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_phase6(project_root)
