import os
import json
import optuna
import torch
import torch.nn as nn
from utils.hpo_utils import save_optimized_architecture
from dataloader.dataloader import get_dataloader
from Model.Backbones.tst import TimeSeriesTransformer
from Model.PI_nets.DeepOPINN import Model as DeepOPINN
from Model.PI_nets.LAX import OptimizationNetwork as LAXModel
from pipeline.phase3_train_deepopinn import ArgsMock
from pipeline.phase4_train_lax import ArgsMockLax
import numpy as np

# A very fast objective function wrapper to evaluate architectures
def optimize_tst(trial, dataloader, device):
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    model = TimeSeriesTransformer(num_features=3, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Train for a few batches to evaluate
    model.train()
    for batch_idx, (features, target_kpd, _) in enumerate(dataloader):
        features = features.to(device)
        target_kpd = target_kpd.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), target_kpd.squeeze())
        loss.backward()
        optimizer.step()
        if batch_idx > 5:
            break
            
    return loss.item()

def optimize_deepopinn(trial, dataloader, device):
    F_hidden_dim = trial.suggest_categorical('F_hidden_dim', [20, 40, 60])
    F_layers_num = trial.suggest_int('F_layers_num', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    args = ArgsMock()
    model = DeepOPINN(args, save_args=False, F_hidden_dim=F_hidden_dim, F_layers_num=F_layers_num, dropout=dropout).to(device)
    
    model.train()
    total_loss = 0
    for iter, (x1, target_kpd, target_soh) in enumerate(dataloader):
        x1, target_kpd, target_soh = x1.to(device), target_kpd.to(device), target_soh.to(device)

        if model.extractor_deepopinn is None:
            # Flatten x1 to match DeepOPINN dataset structure
            x1_flat = x1.view(x1.shape[0], -1)
            model.initialize_networks(x1_flat.shape[1])
            
        x1_flat = x1.view(x1.shape[0], -1)
        u1, f1 = model.forward_deepopinn(x1_flat)
        weight = model.relu(-target_kpd)
        loss1 = (torch.pow(u1 - target_soh, 2) * weight).mean()
        f_target = torch.zeros_like(f1)
        loss2 = (torch.pow(f1 - f_target, 2) * weight).mean()
        loss = loss1 + model.args.alpha*loss2
        
        model.optimizer_extractor.zero_grad()
        model.optimizer_solution.zero_grad()
        model.optimizer_F.zero_grad()
        loss.backward()
        model.optimizer_extractor.step()
        model.optimizer_solution.step()
        model.optimizer_F.step()
        
        total_loss += loss.item()
        if iter > 2:
            break
            
    return total_loss

def run_hpo_pipeline(project_root):
    print("="*50)
    print("BATTERY PROGNOSTICS BAYESIAN OPTIMIZATION")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_dir = os.path.join(project_root, 'config')
    
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    for dataset, batches in train_test_split.items():
        for batch in batches:
            print(f"\nRunning HPO for Dataset: {dataset}, Batch: {batch}")
            train_bats = train_test_split[dataset][batch]['train']
            
            with open(os.path.join(config_dir, 'initial_knee_points.json'), 'r') as f:
                initial_knee_points = json.load(f)
            knee_points_batch = initial_knee_points[dataset][batch]
            data_root = os.path.join(project_root, 'data', 'Processed')
            # Load dummy dataloader for TST & DeepOPINN
            dataloader = get_dataloader(data_root, dataset, train_bats, batch_size=64, shuffle=True, initial_knee_points=knee_points_batch)
            
            # 1. Optimize TST
            study_tst = optuna.create_study(direction='minimize')
            study_tst.optimize(lambda trial: optimize_tst(trial, dataloader, device), n_trials=5)
            save_optimized_architecture(project_root, dataset, batch, "TST", study_tst.best_params)
            
            # 2. Optimize DeepOPINN
            study_deepopinn = optuna.create_study(direction='minimize')
            study_deepopinn.optimize(lambda trial: optimize_deepopinn(trial, dataloader, device), n_trials=5)
            save_optimized_architecture(project_root, dataset, batch, "DeepOPINN", study_deepopinn.best_params)
            
            # NOTE: Similar functions can be easily added for LAX and KaDOPLAX
            
    print("HPO pipeline completed successfully!")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.dirname(__file__))
    run_hpo_pipeline(project_root)
