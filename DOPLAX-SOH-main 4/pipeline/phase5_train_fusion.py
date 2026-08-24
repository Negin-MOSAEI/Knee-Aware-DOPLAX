import os
import json
import torch
import numpy as np
import pandas as pd
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
from utils.plot_utils import plot_learning_curve, save_loss_history

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
        data_dir = os.path.join(project_root, 'data', 'Processed', f'{dataset_name} data')
        for bat_id in battery_ids:
            csv_path = os.path.join(data_dir, f'{bat_id}.csv')
            num_cycles = len(pd.read_csv(csv_path)) if os.path.exists(csv_path) else 800
            kpd_path = os.path.join(kpd_dir, f"{bat_id.replace('/', '_')}_kpd.npy")
            if os.path.exists(kpd_path):
                kpd_seq = np.load(kpd_path)
            else:
                kpd_seq = np.zeros(num_cycles, dtype=np.float32)
            if len(kpd_seq) < num_cycles:
                kpd_seq = np.pad(kpd_seq, (0, num_cycles - len(kpd_seq)), mode='edge')
            elif len(kpd_seq) > num_cycles:
                kpd_seq = kpd_seq[:num_cycles]
            all_kpd.extend(kpd_seq)
            
        self.inferred_kpds = np.array(all_kpd)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        features, _, true_soh = self.base_dataset[idx]
        inferred_kpd = torch.tensor([self.inferred_kpds[idx]], dtype=torch.float32)
        cycle_t = torch.tensor([idx + self.base_dataset.window_size], dtype=torch.float32)
        return torch.tensor(features, dtype=torch.float32), inferred_kpd, cycle_t, torch.tensor([true_soh], dtype=torch.float32)

def run_phase5(project_root: str, num_epochs: int = 5, batch_size: int = 64, resume: bool = True):
    """Executes Phase 5: Train MoE Fusion (KaDOPLAX gating head) model."""
    print("Starting Phase 5: Training Fusion (MoE adaptive gating)...")
    
    config_dir = os.path.join(project_root, 'config')
    kpd_out_dir = os.path.join(project_root, 'outputs', 'kpd_predictions')
    models_dir = os.path.join(project_root, 'outputs', 'models')
    os.makedirs(models_dir, exist_ok=True)
    kpi_report = {}
    
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    for dataset, batches in train_test_split.items():
        for batch in batches:
            # Resume support: skip fusion models that are already trained
            kadoplax_save_path = os.path.join(models_dir, f'kadoplax_{dataset}_{batch}.pt')
            if resume and os.path.exists(kadoplax_save_path):
                print(f"\n--- Skipping {dataset} - {batch} (already trained: {os.path.basename(kadoplax_save_path)}) ---")
                continue

            print(f"\n--- Training Fusion MLP for {dataset} - {batch} ---")
            
            train_bats = train_test_split[dataset][batch]['train']
            val_bats = train_test_split[dataset][batch].get('val', [])
            if not val_bats:
                val_bats = train_test_split[dataset][batch]['test']
            
            dataset_obj = FusionDataset(train_bats, os.path.join(kpd_out_dir, dataset, batch), dataset, project_root)
            val_dataset_obj = FusionDataset(val_bats, os.path.join(kpd_out_dir, dataset, batch), dataset, project_root)
            
            if len(dataset_obj) == 0:
                print("No data found for this batch. Skipping.")
                continue
                
            dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset_obj, batch_size=batch_size, shuffle=False)
            
            kpi_tracker = KPITracker()
            kpi_tracker.start(num_batteries=len(train_bats), total_cycles=len(dataset_obj))
            
            # Load DeepOPINN
            args_do = ArgsMock(project_root)
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
            lax_num_features = args_lax.dim_in_LAX
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
                try:
                    lax.load_state_dict(state_dict, strict=False)
                except RuntimeError as e:
                    print(f"  Warning: Cannot load old LAX weights (architecture mismatch): {e}")
                    print(f"  Training LAX from scratch.")
            
            # Initialize Fusion MLP with input_dim=3 (u_1, u_2, KPD)
            from utils.hpo_utils import get_model_params
            best_params = get_model_params(project_root, dataset, batch, "FusionMLP")
            if best_params:
                print(f"Using optimized FusionMLP architecture: {best_params}")
                bagging_mlp = BaggingMLP(input_dim=3, **best_params).to(device)
            else:
                bagging_mlp = BaggingMLP(input_dim=3, hidden_dim=64, num_layers=3).to(device)
            
            # KaDOPLAX handles freezing DeepOPINN and LAX internally
            model = KaDOPLAX(deepopinn, lax, bagging_mlp).to(device)
            
            # Train only the MoE fusion head (moe_gate); expert backbones are
            # frozen inside KaDOPLAX. A small LR with AdamW + weight decay
            # makes the gate gently learn to correct the base models instead
            # of overwriting them.
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=1e-2
            )
            criterion = nn.HuberLoss(delta=0.1)
            
            model.train()
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            best_model_state = None
            
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                for features, kpd, cycle_t, target_soh in dataloader:
                    features, kpd, cycle_t, target_soh = features.to(device), kpd.to(device), cycle_t.to(device), target_soh.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(features, kpd, cycle_t, epoch=1000)
                    
                    loss = criterion(outputs.view(-1), target_soh.view(-1))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    kpi_tracker.update_ram()
                    
                avg_loss = total_loss/len(dataloader)
                train_losses.append(avg_loss)
                
                # Validation loop
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for features, kpd, cycle_t, target_soh in val_dataloader:
                        features, kpd, cycle_t, target_soh = features.to(device), kpd.to(device), cycle_t.to(device), target_soh.to(device)
                        outputs = model(features, kpd, cycle_t, epoch=1000)
                        loss = criterion(outputs.view(-1), target_soh.view(-1))
                        val_loss += loss.item()
                avg_val_loss = val_loss / max(1, len(val_dataloader))
                val_losses.append(avg_val_loss)
                
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                
            metrics = kpi_tracker.stop()
            if dataset not in kpi_report:
                kpi_report[dataset] = {}
            kpi_report[dataset][batch] = metrics
            
            learning_curve_dir = os.path.join(project_root, 'experiments', 'fusion_experiments')
            plot_learning_curve(train_losses, val_losses, "FusionMLP", dataset, batch, os.path.join(project_root, 'outputs', 'figures', 'learning_curves'))
            save_loss_history(train_losses, val_losses, "FusionMLP", dataset, batch, learning_curve_dir)
            
            model_save_path = os.path.join(models_dir, f'kadoplax_{dataset}_{batch}.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved KaDOPLAX model to {model_save_path}")

    # Save KPI report
    reports_dir = os.path.join(project_root, 'outputs', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, 'fusion_training_kpis.json'), 'w') as f:
        json.dump(kpi_report, f, indent=4)
    print("Phase 5 completed successfully.")

if __name__ == "__main__":
    run_phase5(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
