import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List

from Model.Combination_nets.KaDOPLAX import KaDOPLAX
from Model.Backbones.bagging_mlp import BaggingMLP
from Model.PI_nets.DeepOPINN import Model as DeepOpinn
from Model.PI_nets.LAX import OptimizationNetwork as LAXModel
from pipeline.phase3_train_deepopinn import ArgsMock
from pipeline.phase4_train_lax import ArgsMockLax
from utils.kpi_tracker import KPITracker
from dataloader.dataloader import BatteryCycleDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FusionDataset(Dataset):
    """
    Dataset for Fusion training. Yields features, inferred kpd, and true SOH.
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
            
        self.inferred_kpds = np.array(all_kpd)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        features, _, true_soh = self.base_dataset[idx]
        inferred_kpd = torch.tensor([self.inferred_kpds[idx]], dtype=torch.float32)
        cycle_t = torch.tensor([idx + self.base_dataset.window_size], dtype=torch.float32)
        return torch.tensor(features, dtype=torch.float32), inferred_kpd, cycle_t, torch.tensor([true_soh], dtype=torch.float32)

def run_phase5(project_root: str, num_epochs: int = 5, batch_size: int = 64):
    """Executes Phase 5: Train Fusion MLP (BaggingMLP) model."""
    print("Starting Phase 5: Training Fusion (Bagging MLP)...")
    
    config_dir = os.path.join(project_root, 'config')
    kpd_out_dir = os.path.join(project_root, 'outputs', 'kpd_predictions')
    models_dir = os.path.join(project_root, 'outputs', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    for dataset, batches in train_test_split.items():
        for batch in batches:
            print(f"\n--- Training Fusion MLP for {dataset} - {batch} ---")
            
            train_bats = train_test_split[dataset][batch]['train']
            
            dataset_obj = FusionDataset(train_bats, os.path.join(kpd_out_dir, dataset, batch), dataset, project_root)
            
            if len(dataset_obj) == 0:
                print("No data found for this batch. Skipping.")
                continue
                
            generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
            dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True, generator=generator)
            
            kpi_tracker = KPITracker()
            kpi_tracker.start(num_batteries=len(train_bats), total_cycles=len(dataset_obj))
            
            # Load DeepOPINN
            args_do = ArgsMock()
            from utils.hpo_utils import get_model_params
            best_params_do = get_model_params(project_root, dataset, batch, "DeepOPINN")
            if best_params_do:
                deepopinn = DeepOpinn(args_do, save_args=False, **best_params_do).to(device)
            else:
                deepopinn = DeepOpinn(args_do, save_args=False).to(device)
            
            # Need actual features size
            peek_features = dataset_obj[0][0]
            deepopinn_num_features = np.prod(peek_features.shape)
            y_dim = deepopinn_num_features
            deepopinn.initialize_networks(deepopinn_num_features)
            
            do_path = os.path.join(models_dir, f'deepopinn_{dataset}_{batch}.pt')
            if os.path.exists(do_path):
                deepopinn.load_state_dict(torch.load(do_path, map_location=device, weights_only=True))
            
            args_lax = ArgsMockLax(project_root)
            lax_num_features = dataset_obj.base_dataset.num_features
            X_mean = torch.zeros(lax_num_features, dtype=torch.float32).to(device)
            X_std = torch.ones(lax_num_features, dtype=torch.float32).to(device)
            x_sts = [X_mean, X_std]
            y_dim = lax_num_features
            
            best_params_lax = get_model_params(project_root, dataset, batch, "LAX")
            if best_params_lax:
                lax = LAXModel(x_sts, args_lax, lax_num_features, y_dim, **best_params_lax).to(device)
            else:
                lax = LAXModel(x_sts, args_lax, lax_num_features, y_dim).to(device)
                
            lax_path = os.path.join(models_dir, f'lax_{dataset}_{batch}.pt')
            if os.path.exists(lax_path):
                state_dict = torch.load(lax_path, map_location=device, weights_only=True)
                if 'y' in state_dict:
                    del state_dict['y']
                lax.load_state_dict(state_dict, strict=False)
            
            # Initialize Fusion MLP with input_dim=3 (u_1, u_2, KPD)
            from utils.hpo_utils import get_model_params
            best_params = get_model_params(project_root, dataset, batch, "FusionMLP")
            if best_params:
                print(f"Using optimized FusionMLP architecture: {best_params}")
                bagging_mlp = BaggingMLP(num_models=5, input_dim=3, **best_params).to(device)
            else:
                bagging_mlp = BaggingMLP(num_models=5, input_dim=3, hidden_dim=64, num_layers=3).to(device)
            
            # KaDOPLAX handles freezing DeepOPINN and LAX internally
            model = KaDOPLAX(deepopinn, lax, bagging_mlp).to(device)
            
            optimizer = optim.Adam(model.fusion_mlp.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(num_epochs):
                total_loss = 0.0
                for features, kpd, cycle_t, target_soh in dataloader:
                    features = features.to(device)
                    kpd = kpd.to(device)
                    cycle_t = cycle_t.to(device)
                    target_soh = target_soh.to(device)
                    
                    optimizer.zero_grad()
                    # Forward pass
                    # The fusion logic inside KaDOPLAX expects features, kpd, cycle_t, epoch
                    pred_soh = model(features, kpd, cycle_t, epoch=1000) # Use 1000 to enable optimization of LAX
                    
                    pred_soh = pred_soh.view(-1, 1)
                    target_soh = target_soh.view(-1, 1)
                    
                    loss = criterion(pred_soh, target_soh)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.6f}")
                kpi_tracker.update_ram()
                
            kpi_tracker.stop()
            
            model_save_path = os.path.join(models_dir, f'kadoplax_{dataset}_{batch}.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved KaDOPLAX model to {model_save_path}")

if __name__ == "__main__":
    run_phase5(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
