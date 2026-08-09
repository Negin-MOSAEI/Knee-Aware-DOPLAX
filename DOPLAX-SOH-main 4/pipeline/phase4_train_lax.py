import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List
from Model.PI_nets.LAX import OptimizationNetwork, run_epoch
from dataloader.dataloader import BatteryCycleDataset
from utils.kpi_tracker import KPITracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LaxDataset(Dataset):
    """
    Dataset for LAX training. Yields pairs of consecutive cycles: (x1, x2, y1, y2)
    where x1 and x2 have the time index appended to the features.
    """
    def __init__(self, battery_ids: List[str], dataset_name: str, project_root: str):
        # We use a custom loading approach here because we need consecutive cycles for LAX
        self.samples = []
        data_dir = os.path.join(project_root, 'data', 'Processed', f"{dataset_name} data")
        
        for bat_id in battery_ids:
            file_path = os.path.join(data_dir, f"{bat_id}.csv")
            if not os.path.exists(file_path):
                continue
            
            import pandas as pd
            df = pd.read_csv(file_path).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
            raw_data_matrix = df.values
            
            # SOH
            capacity = raw_data_matrix[:, 0]
            initial_capacity = capacity[0] if capacity[0] != 0 else 1e-6
            soh = capacity / initial_capacity
            
            # For LAX, x requires features and cycle time t
            num_cycles = raw_data_matrix.shape[0]
            features_with_t = np.zeros((num_cycles, 3 + 1))
            features_with_t[:, :3] = raw_data_matrix[:, :3]
            features_with_t[:, -1] = np.arange(1, num_cycles + 1)
            
            for i in range(num_cycles - 1):
                x1 = features_with_t[i]
                x2 = features_with_t[i+1]
                y1 = np.array([soh[i]])
                y2 = np.array([soh[i+1]])
                self.samples.append((x1, x2, y1, y2))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.samples[idx]
        return (torch.tensor(x1, dtype=torch.float32), 
                torch.tensor(x2, dtype=torch.float32), 
                torch.tensor(y1, dtype=torch.float32), 
                torch.tensor(y2, dtype=torch.float32))

class ArgsMockLax:
    def __init__(self, project_root):
        self.results_path = os.path.join(project_root, 'outputs')
        self.log_dir = "logs"
        self.beta_LAX = 1.0
        self.time_block_LAX = "theta"
        self.dim_in_LAX = 3 # Only using the 3 basic features
        self.dim_hidden_LAX = 64
        self.center_block_LAX = "H*"
        self.dim_output_LAX = 1
        self.inside_S_MLP_layers = [64, 64, 64]
        self.inside_betan_layers = [64, 64]
        self.n_steps_LAX = 100
        self.run_for_LAX = True
        self.epochs = 5
        self.epoch_th_LAX = 0.5
        self.batch_size = 64
        self.theta_LAX = 1.0
        self.zeta_LAX = 1.0
        self.kata_LAX = 1.0
        self.betha_LAX = 1.0
        self.dual_LAX = 1.0
        self.distance_block_LAX = "MLP"
        self.h_dim_LAX = 64
        self.epoch_y_LAX = 5
        self.s_LAX = "MLP"
        self.inside_distance_block_MLP_layers = [64]
        self.inside_theta_layers = [64]
        self.inside_multivar_theta_layers = [64]
        self.inside_h_star_layers = [64, 64]
        self.inside_phi_layers = [64, 64]
        self.inside_g_layers = [64]
        self.h_out_LAX = 1
        self.phi_out_LAX = 1
        self.g_dim = 3
        self.g_out_LAX = 1
        self.norm = 2
        self.schedule_beta = 0
        self.dual_LAX = 0

def run_phase4(project_root: str, num_epochs: int = 5, batch_size: int = 64):
    """Executes Phase 4: Train LAX model on Train batteries."""
    print("Starting Phase 4: Training LAX...")
    
    config_dir = os.path.join(project_root, 'config')
    models_dir = os.path.join(project_root, 'outputs', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    args = ArgsMockLax(project_root)
    
    for dataset, batches in train_test_split.items():
        for batch in batches:
            print(f"\n--- Training LAX for {dataset} - {batch} ---")
            
            train_bats = train_test_split[dataset][batch]['train']
            
            dataset_obj = LaxDataset(train_bats, dataset, project_root)
            
            if len(dataset_obj) == 0:
                print("No data found for this batch. Skipping.")
                continue
                
            generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
            dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True, generator=generator)
            
            kpi_tracker = KPITracker()
            kpi_tracker.start(num_batteries=len(train_bats), total_cycles=len(dataset_obj))
            
            # Provide sample x_sts based on data
            all_x = np.array([s[0] for s in dataset_obj.samples])
            X_mean = torch.tensor(np.mean(all_x[:, :-1], axis=0), dtype=torch.float32).to(device)
            X_std = torch.tensor(np.std(all_x[:, :-1], axis=0), dtype=torch.float32).to(device)
            x_sts = [X_mean, X_std]
            x_dim = all_x.shape[1] - 1
            y_dim = x_dim
            
            from utils.hpo_utils import get_model_params
            best_params = get_model_params(project_root, dataset, batch, "LAX")
            if best_params:
                print(f"Using optimized LAX architecture: {best_params}")
                model = OptimizationNetwork(x_sts, args, x_dim, y_dim, **best_params).to(device)
            else:
                model = OptimizationNetwork(x_sts, args, x_dim, y_dim).to(device)
            # Dummy optimizer for passing to run_epoch
            opt_net = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.current_lr_y = 1e-3
            
            for epoch in range(num_epochs):
                run_epoch(model, opt_net, opt_net, args, batch, dataset, epoch=epoch, phase="train", dataloader=dataloader)
                kpi_tracker.update_ram()
                
            kpi_tracker.stop()
            
            model_save_path = os.path.join(models_dir, f'lax_{dataset}_{batch}.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved LAX model to {model_save_path}")

if __name__ == "__main__":
    run_phase4(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
