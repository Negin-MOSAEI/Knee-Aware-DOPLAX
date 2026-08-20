import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

from dataloader.dataloader import get_dataloader
from Model.Backbones.tst import TimeSeriesTransformer
from Model.utils.lr_schedulers import cosine_annealing
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
            val_bats = train_test_split[dataset][batch].get('val', [])
            if not val_bats:
                val_bats = train_test_split[dataset][batch]['test']
            knee_points_batch = initial_knee_points[dataset][batch]
            
            # Initialize KPI Tracker
            kpi_tracker = KPITracker()
            
            # Check for HPO parameters
            from utils.hpo_utils import get_model_params
            best_params = get_model_params(project_root, dataset, batch, "TST")
            
            current_batch_size = batch_size
            current_lr = 1e-4
            model_params = {}
            if best_params:
                print(f"Using optimized TST hyperparameters: {best_params}")
                current_batch_size = best_params.get('batch_size', current_batch_size)
                current_lr = best_params.get('lr', current_lr)
                model_params = {k: v for k, v in best_params.items() if k not in ['batch_size', 'lr']}
            
            # Initialize DataLoader (window_size=40, num_features=3)
            dataloader = get_dataloader(
                battery_ids=train_bats,
                initial_knee_points=knee_points_batch,
                dataset_name=dataset,
                data_root=os.path.join(project_root, 'data', 'Processed'),
                batch_size=current_batch_size,
                shuffle=True,
                window_size=40
            )
            
            val_dataloader = get_dataloader(
                battery_ids=val_bats,
                initial_knee_points=knee_points_batch,
                dataset_name=dataset,
                data_root=os.path.join(project_root, 'data', 'Processed'),
                batch_size=current_batch_size,
                shuffle=False,
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
            all_x_tst = np.concatenate([s[0] for s in dataloader.dataset.samples], axis=0)
            X_mean_tst = torch.tensor(np.mean(all_x_tst, axis=0), dtype=torch.float32).to(device)
            X_std_tst = torch.tensor(np.std(all_x_tst, axis=0), dtype=torch.float32).to(device)
            x_sts_tst = [X_mean_tst, X_std_tst]

            model = TimeSeriesTransformer(num_features=actual_num_features, x_sts=x_sts_tst, **model_params).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            
            from utils.plot_utils import plot_learning_curve, save_loss_history
            
            # Training loop
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            best_model_state = None
            
            for epoch in range(num_epochs):
                # Update learning rate dynamically
                curr_epoch_lr = cosine_annealing(
                    epoch=epoch,
                    warmup_epochs=max(1, num_epochs // 5),
                    restart_period=max(1, num_epochs),
                    min_lr=1e-6,
                    initial_lr=current_lr
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = curr_epoch_lr

                model.train()
                epoch_loss = 0.0
                for batch_idx, (features, target_kpd, _) in enumerate(dataloader):
                    features = features.to(device)
                    target_kpd = target_kpd.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(features)
                    
                    outputs = outputs.view(-1, 1)
                    target_kpd = target_kpd.view(-1, 1)
                    loss = criterion(outputs, target_kpd)
                    loss.backward()
                    optimizer.step()
                    
                    if batch_idx == 0:
                        print(f"Batch 0 predictions (first 5): {outputs.squeeze()[:5].detach().cpu().numpy()}")
                        print(f"Batch 0 targets (first 5): {target_kpd.squeeze()[:5].detach().cpu().numpy()}")
                    
                    epoch_loss += loss.item()
                    
                    # Update RAM usage tracking periodically
                    kpi_tracker.update_ram()
                    
                avg_loss = epoch_loss/len(dataloader)
                train_losses.append(avg_loss)
                
                # Validation loop
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for features, target_kpd, _ in val_dataloader:
                        features = features.to(device)
                        target_kpd = target_kpd.to(device).view(-1, 1)
                        outputs = model(features).view(-1, 1)
                        loss = criterion(outputs, target_kpd)
                        val_loss += loss.item()
                        
                avg_val_loss = val_loss / max(1, len(val_dataloader))
                val_losses.append(avg_val_loss)
                
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            # Stop KPI tracking
            kpi_metrics = kpi_tracker.stop()
            kpi_report[dataset][batch] = kpi_metrics
            
            # Save learning curves
            learning_curve_dir = os.path.join(project_root, 'experiments', 'tst_experiments')
            plot_learning_curve(train_losses, val_losses, "TST", dataset, batch, os.path.join(project_root, 'outputs', 'figures', 'learning_curves'))
            save_loss_history(train_losses, val_losses, "TST", dataset, batch, learning_curve_dir)
            
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
