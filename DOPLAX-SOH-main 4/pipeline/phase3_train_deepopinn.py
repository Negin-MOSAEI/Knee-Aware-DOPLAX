import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List

from Model.PI_nets.DeepOPINN import Model as DeepOpinn
from utils.kpi_tracker import KPITracker
from dataloader.dataloader import BatteryCycleDataset

DATASETS = {
    'HUST': ['default'],
    'MIT': ['default'],
    'TJU': ['NCA', 'NCM', 'NCA_NCM'],
    'XJTU': ['2C', '3C', 'R2.5', 'R3', 'RW', 'sim_satellite']
}

class ArgsMock:
    def __init__(self):
        self.save_folder = None
        self.log_dir = "logs"
        self.data = "XJTU"
        
        # Base attributes
        self.lr = 1e-3
        self.warmup_lr = 1e-3
        self.lr_F = 1e-3
        self.final_lr = 1e-4
        self.F_hidden_dim = 64
        self.F_layers_num = 3
        self.dropout = 0.1
        self.alpha = 0.5
        self.beta = 0.5
        self.warmup_epochs = 1
        
        # XJTU
        self.lr_XJTU = 1e-3
        self.warmup_lr_XJTU = 1e-3
        self.lr_F_XJTU = 1e-3
        self.final_lr_XJTU = 1e-4
        self.F_hidden_dim_XJTU = 64
        self.F_layers_num_XJTU = 3
        self.dropout_XJTU = 0.1
        self.alpha_XJTU = 0.5
        self.beta_XJTU = 0.5
        self.warmup_epochs_XJTU = 1
        
        # TJU
        self.lr_TJU = 1e-3
        self.warmup_lr_TJU = 1e-3
        self.lr_F_TJU = 1e-3
        self.final_lr_TJU = 1e-4
        self.F_hidden_dim_TJU = 64
        self.F_layers_num_TJU = 3
        self.dropout_TJU = 0.1
        self.alpha_TJU = 0.5
        self.beta_TJU = 0.5
        self.warmup_epochs_TJU = 1
        
        # MIT
        self.lr_MIT = 1e-3
        self.warmup_lr_MIT = 1e-3
        self.lr_F_MIT = 1e-3
        self.final_lr_MIT = 1e-4
        self.F_hidden_dim_MIT = 64
        self.F_layers_num_MIT = 3
        self.dropout_MIT = 0.1
        self.alpha_MIT = 0.5
        self.beta_MIT = 0.5
        self.warmup_epochs_MIT = 1
        
        # HUST
        self.lr_HUST = 1e-3
        self.warmup_lr_HUST = 1e-3
        self.lr_F_HUST = 1e-3
        self.final_lr_HUST = 1e-4
        self.F_hidden_dim_HUST = 64
        self.F_layers_num_HUST = 3
        self.dropout_HUST = 0.1
        self.alpha_HUST = 0.5
        self.beta_HUST = 0.5
        self.warmup_epochs_HUST = 1
        
        self.dim_x = 1
        self.epochs = 5
        self.batch_size = 64

class DeepOpinnDataset(Dataset):
    """
    Dataset for DeepOpinn training. Yields features, inferred kpd, and true SOH.
    """
    def __init__(self, battery_ids: List[str], kpd_dir: str, dataset_name: str, project_root: str):
        self.base_dataset = BatteryCycleDataset(
            data_root=os.path.join(project_root, 'data', 'Processed'),
            dataset_name=dataset_name, 
            battery_ids=battery_ids,
            capacity_column_index=0
        )
        
        all_kpd = []
        for bat_id in battery_ids:
            kpd_path = os.path.join(kpd_dir, f"{bat_id.replace('/', '_')}_kpd.npy")
            if os.path.exists(kpd_path):
                kpd_seq = np.load(kpd_path)
            else:
                kpd_seq = np.zeros(800, dtype=np.float32)
            all_kpd.extend(kpd_seq)
            
        self.inferred_kpds = all_kpd

    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        # 3-element unpacking from BatteryCycleDataset: features, target_kpd, target_soh
        features, _, true_soh = self.base_dataset[idx]
        inferred_kpd = torch.tensor([self.inferred_kpds[idx]], dtype=torch.float32)
        # return in format expected by DeepOPINN.py: (x1, target_kpd, target_soh)
        # DeepOPINN expects features to be 2D matrix or flattened
        # But we pass features as-is
        features_flat = features.reshape(-1)
        return features_flat, inferred_kpd, true_soh

def run_phase3(project_root: str, num_epochs: int = 5, batch_size: int = 64):
    """Executes Phase 3: Train DeepOpinn model on Train batteries."""
    print("Starting Phase 3: Training DeepOpinn...")
    
    config_dir = os.path.join(project_root, 'config')
    kpd_out_dir = os.path.join(project_root, 'outputs', 'kpd_predictions')
    models_dir = os.path.join(project_root, 'outputs', 'models')
    reports_dir = os.path.join(project_root, 'outputs', 'reports')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    kpi_report = {}
    
    args = ArgsMock()
    args.epochs = num_epochs
    args.batch_size = batch_size
    
    for dataset, batches in train_test_split.items():
        kpi_report[dataset] = {}
        for batch in batches:
            print(f"\n--- Training DeepOpinn for {dataset} - {batch} ---")
            
            train_bats = train_test_split[dataset][batch]['train']
            batch_kpd_dir = os.path.join(kpd_out_dir, dataset, batch)
            
            dataset_obj = DeepOpinnDataset(train_bats, batch_kpd_dir, dataset, project_root)
            
            if len(dataset_obj) == 0:
                print("No data found for this batch. Skipping.")
                continue
                
            generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
            dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True, generator=generator)
            
            kpi_tracker = KPITracker()
            kpi_tracker.start(num_batteries=len(train_bats), total_cycles=len(dataset_obj))
            
            # Initialize Model
            args = ArgsMock()
            from utils.hpo_utils import get_model_params
            best_params = get_model_params(project_root, dataset, batch, "DeepOPINN")
            if best_params:
                print(f"Using optimized DeepOPINN architecture: {best_params}")
                model = DeepOpinn(args, save_args=False, **best_params).to(device)
            else:
                model = DeepOpinn(args, save_args=False).to(device)
            
            for epoch in range(num_epochs):
                model.train_one_epoch(epoch, dataloader)
                kpi_tracker.update_ram()
                
            metrics = kpi_tracker.stop()
            kpi_report[dataset][batch] = metrics
            
            model_save_path = os.path.join(models_dir, f'deepopinn_{dataset}_{batch}.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            
    with open(os.path.join(reports_dir, 'deepopinn_kpis.json'), 'w') as f:
        json.dump(kpi_report, f, indent=4)
        
    print("\nPhase 3 completed successfully.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_phase3(project_root)
