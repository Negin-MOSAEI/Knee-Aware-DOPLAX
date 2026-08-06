import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

from dataloader.dataloader import get_dataloader
from Model.Backbones.tst import TimeSeriesTransformer
from utils.kpi_tracker import KPITracker

# Datasets and Batches definition
DATASETS = {
    'HUST': ['default'],
    'MIT': ['default'],
    'TJU': ['NCA', 'NCM', 'NCA_NCM'],
    'XJTU': ['2C', '3C', 'R2.5', 'R3', 'RW', 'sim_satellite']
}

def run_phase1(project_root: str, num_epochs: int = 5, batch_size: int = 32):
    """
    Executes Phase 1: Train the Time-Series Transformer (TST) on Train batteries.
    
    Args:
        project_root (str): Root directory of the project.
        num_epochs (int): Number of training epochs per batch.
        batch_size (int): Batch size for DataLoader.
    """
    print("Starting Phase 1: Training the Time-Series Transformer (TST)...")
    
    config_dir = os.path.join(project_root, 'config')
    models_dir = os.path.join(project_root, 'outputs', 'models')
    reports_dir = os.path.join(project_root, 'outputs', 'reports')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configs
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    with open(os.path.join(config_dir, 'initial_knee_points.json'), 'r') as f:
        initial_knee_points = json.load(f)
        
    kpi_report = {}
        
    for dataset, batches in train_test_split.items():
        kpi_report[dataset] = {}
        for batch in batches:
            print(f"\n--- Training TST for {dataset} - {batch} ---")
            
            # Retrieve train batteries and knee points for this specific batch
            train_bats = train_test_split[dataset][batch]['train']
            knee_points_batch = initial_knee_points[dataset][batch]
            
            # Initialize KPI Tracker
            kpi_tracker = KPITracker()
            
            # Initialize DataLoader (window_size=40, num_features=3)
            dataloader = get_dataloader(
                battery_ids=train_bats,
                initial_knee_points=knee_points_batch,
                dataset_name=dataset,
                data_root=os.path.join(project_root, 'data', 'Processed'),
                batch_size=batch_size,
                shuffle=True,
                window_size=40
            )
            
            # Calculate total cycles for KPI
            dataset_obj = dataloader.dataset
            if len(dataset_obj) == 0:
                print("No data found for this batch. Skipping.")
                continue
                
            features_sample, _, _ = dataset_obj[0]
            actual_num_features = features_sample.shape[-1]
            
            kpi_tracker = KPITracker()
            kpi_tracker.start(num_batteries=len(train_bats), total_cycles=len(dataset_obj))
            
            # Initialize Model
            model = TimeSeriesTransformer(num_features=actual_num_features).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # Training loop
            model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for features, target_kpd, _ in dataloader:
                    features = features.to(device)
                    target_kpd = target_kpd.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(features)
                    
                    loss = criterion(outputs, target_kpd)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    # Update RAM usage tracking periodically
                    kpi_tracker.update_ram()
                    
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
            
            # Stop KPI tracking
            kpi_metrics = kpi_tracker.stop()
            kpi_report[dataset][batch] = kpi_metrics
            
            # Save Model Weights
            model_save_path = os.path.join(models_dir, f'tst_{dataset}_{batch}.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            print(f"KPIs: {kpi_metrics}")
            
    # Save KPI report for Phase 1
    with open(os.path.join(reports_dir, 'phase1_kpi_report.json'), 'w') as f:
        json.dump(kpi_report, f, indent=4)
        
    print("\nPhase 1 completed successfully.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_phase1(project_root)
