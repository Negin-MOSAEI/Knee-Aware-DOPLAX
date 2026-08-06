import os
import json
import torch
import numpy as np
from typing import Dict, List, Any

from Model.Backbones.tst import TimeSeriesTransformer
from utils.math_utils import tst_cold_start_kpd
from dataloader.dataloader import BatteryCycleDataset

DATASETS = {
    'HUST': ['default'],
    'MIT': ['default'],
    'TJU': ['NCA', 'NCM', 'NCA_NCM'],
    'XJTU': ['2C', '3C', 'R2.5', 'R3', 'RW', 'sim_satellite']
}

def infer_kpd_for_battery(model: torch.nn.Module, battery_id: str, dataset_name: str, project_root: str, device: torch.device, window_size: int = 40) -> np.ndarray:
    """
    Infers the full KPD sequence for a single battery.
    Uses cold start logic for cycles 1 to 39.
    Uses the trained TST model for cycles >= 40.
    """
    dataset = BatteryCycleDataset(
        battery_ids=[battery_id],
        initial_knee_points={battery_id: 0},
        dataset_name=dataset_name,
        data_root=os.path.join(project_root, 'data', 'Processed'),
        window_size=window_size
    )
    
    kpd_sequence = []
    model.eval()
    with torch.no_grad():
        for cycle_idx in range(len(dataset)):
            cycle = cycle_idx + 1
            # Correctly unpack 3 elements
            features, _, _ = dataset[cycle_idx]
            
            if cycle < window_size:
                kpd = tst_cold_start_kpd(cycle)
            else:
                window_tensor = features.unsqueeze(0).to(device) 
                pred_kpd = model(window_tensor)
                kpd = pred_kpd.item()
                
            kpd_sequence.append(kpd)
            
    return np.array(kpd_sequence, dtype=np.float32)

def run_phase2(project_root: str):
    """Executes Phase 2: KPD Inference for all batteries."""
    print("Starting Phase 2: KPD Inference...")
    
    config_dir = os.path.join(project_root, 'config')
    models_dir = os.path.join(project_root, 'outputs', 'models')
    kpd_out_dir = os.path.join(project_root, 'outputs', 'kpd_predictions')
    os.makedirs(kpd_out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    for dataset, batches in train_test_split.items():
        for batch in batches:
            print(f"Inferring KPD for {dataset} - {batch}...")
            
            model_path = os.path.join(models_dir, f'tst_{dataset}_{batch}.pt')
            if not os.path.exists(model_path):
                print(f"Warning: Model {model_path} not found. Skipping.")
                continue
                
            train_bats = train_test_split[dataset][batch]['train']
            test_bats = train_test_split[dataset][batch]['test']
            all_bats = train_bats + test_bats
            
            if not all_bats:
                continue
                
            # Peek into the first battery's dataset to dynamically find num_features
            peek_dataset = BatteryCycleDataset([all_bats[0]], {all_bats[0]: 0}, dataset, project_root)
            if len(peek_dataset) == 0:
                print(f"Warning: No data to peek num_features for {dataset}-{batch}. Skipping.")
                continue
            features_sample, _, _ = peek_dataset[0]
            actual_num_features = features_sample.shape[-1]
            
            model = TimeSeriesTransformer(num_features=actual_num_features).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            
            batch_kpd_dir = os.path.join(kpd_out_dir, dataset, batch)
            os.makedirs(batch_kpd_dir, exist_ok=True)
            
            for bat_id in all_bats:
                kpd_seq = infer_kpd_for_battery(model, bat_id, dataset, project_root, device)
                
                safe_bat_id = bat_id.replace('/', '_')
                save_path = os.path.join(batch_kpd_dir, f"{safe_bat_id}_kpd.npy")
                np.save(save_path, kpd_seq)
                
    print("Phase 2 completed successfully. KPD sequences saved.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_phase2(project_root)
