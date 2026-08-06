import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

from Model.Backbones.bagging_mlp import BaggingMLP
from utils.kpi_tracker import KPITracker
from dataloader.dataloader import get_dataloader

DATASETS = {
    'HUST': ['default'],
    'MIT': ['default'],
    'TJU': ['NCA', 'NCM', 'NCA_NCM'],
    'XJTU': ['2C', '3C', 'R2.5', 'R3', 'RW', 'sim_satellite']
}

def run_phase4(project_root: str, num_epochs: int = 5, batch_size: int = 64):
    """Executes Phase 4: Train Bagging MLP on Train batteries."""
    print("Starting Phase 4: Training Bagging MLP...")
    
    config_dir = os.path.join(project_root, 'config')
    models_dir = os.path.join(project_root, 'outputs', 'models')
    reports_dir = os.path.join(project_root, 'outputs', 'reports')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    kpi_report = {}
    
    for dataset, batches in train_test_split.items():
        kpi_report[dataset] = {}
        for batch in batches:
            print(f"\n--- Training Bagging MLP for {dataset} - {batch} ---")
            
            train_bats = train_test_split[dataset][batch]['train']
            
            dataloader = get_dataloader(
                battery_ids=train_bats,
                initial_knee_points={},
                dataset_name=dataset,
                data_root=os.path.join(project_root, 'data', 'Processed'),
                batch_size=batch_size,
                shuffle=True
            )
            
            dataset_obj = dataloader.dataset
            if len(dataset_obj) == 0:
                print("No data found for this batch. Skipping.")
                continue
                
            features_sample, _, _ = dataset_obj[0]
            actual_num_features = features_sample.shape[-1]
            
            kpi_tracker = KPITracker()
            kpi_tracker.start(num_batteries=len(train_bats), total_cycles=len(dataset_obj))
            
            model = BaggingMLP(num_estimators=5, input_dim=actual_num_features).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                # Correctly unpack 3-element tuple and use target_soh
                for features, _, target_soh in dataloader:
                    features = features.to(device)
                    # SOH needs to be reshaped to [batch_size, 1] if not already
                    target_soh = target_soh.view(-1, 1).to(device)
                    
                    optimizer.zero_grad()
                    
                    # MLP expects [batch_size, num_features], extract latest cycle
                    current_features = features[:, -1, :]
                    pred_soh = model(current_features)
                    
                    loss = criterion(pred_soh, target_soh)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    kpi_tracker.update_ram()
                    
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
                
            kpi_metrics = kpi_tracker.stop()
            kpi_report[dataset][batch] = kpi_metrics
            
            model_save_path = os.path.join(models_dir, f'bagging_mlp_{dataset}_{batch}.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            
    with open(os.path.join(reports_dir, 'bagging_mlp_kpis.json'), 'w') as f:
        json.dump(kpi_report, f, indent=4)
        
    print("\nPhase 4 completed successfully.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_phase4(project_root)
