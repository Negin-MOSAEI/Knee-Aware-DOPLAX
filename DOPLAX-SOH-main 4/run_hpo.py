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
from pipeline.phase4_train_lax import ArgsMockLax, LaxDataset
from Model.Backbones.bagging_mlp import BaggingMLP
import numpy as np
import torch.utils.data as data

# A very fast objective function wrapper to evaluate architectures
def optimize_tst(trial, dataloader, device):
    d_model = trial.suggest_categorical('d_model', [8, 16, 32, 64, 128, 256, 512, 1024])
    num_layers = trial.suggest_int('num_layers', 1, 12)
    nhead = trial.suggest_categorical('nhead', [1, 2, 4, 8, 16, 32])
    
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned()
        
    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    
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

def optimize_deepopinn(trial, dataloader, device, project_root):
    F_hidden_dim = trial.suggest_categorical('F_hidden_dim', [8, 16, 32, 64, 128, 256, 512])
    F_layers_num = trial.suggest_int('F_layers_num', 2, 12)
    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    
    args = ArgsMock(project_root)
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

def optimize_fusion(trial, dataloader, device):
    hidden_dim = trial.suggest_categorical('hidden_dim', [8, 16, 32, 64, 128, 256, 512, 1024])
    num_layers = trial.suggest_int('num_layers', 1, 12)
    num_models = trial.suggest_int('num_models', 2, 30)
    
    model = BaggingMLP(num_models=num_models, input_dim=3, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for batch_idx, (features, target_kpd, target_soh) in enumerate(dataloader):
        features, target_kpd, target_soh = features.to(device), target_kpd.to(device), target_soh.to(device)
        
        optimizer.zero_grad()
        dummy_input = torch.randn(features.size(0), 3).to(device)
        outputs = model(dummy_input)
        loss = criterion(outputs.view(-1), target_soh.view(-1))
        loss.backward()
        optimizer.step()
        if batch_idx > 2:
            break
            
    return loss.item()

def optimize_lax(trial, dataloader, device, project_root, x_sts, x_dim, y_dim):
    h_dim_LAX = trial.suggest_categorical('h_dim_LAX', [8, 16, 32, 64, 128, 256, 512])
    
    layer_options = [[16], [32], [64], [128], [256],
                     [16, 16], [32, 32], [64, 64], [128, 128], [256, 256],
                     [16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128], [256, 256, 256],
                     [32, 64, 32], [64, 128, 64], [128, 256, 128]]
    inside_S_MLP_layers = trial.suggest_categorical('inside_S_MLP_layers', layer_options)
    
    args = ArgsMockLax(project_root)
    model = LAXModel(x_sts, args, x_dim, y_dim, 
                     h_dim_LAX=h_dim_LAX, 
                     inside_S_MLP_layers=inside_S_MLP_layers).to(device)
                     
    opt_net = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_F = torch.optim.Adam(model.dynamical_F.parameters(), lr=1e-3)
    model.current_lr_y = 1e-3
    
    criterion = nn.MSELoss()
    total_loss = 0
    for iter, (x1, x2, y1, y2) in enumerate(dataloader):
        x1, y1 = x1.to(device), y1.to(device)
        x2, y2 = x2.to(device), y2.to(device)
        
        x1, x2 = x1[:, :-1], x2[:, :-1]
        t1, t2 = x1[:, -1],  x2[:, -1]
        
        out_1 = model(x=x1, t=t1, epoch=0, return_f=False)
        out_2 = model(x=x2, t=t2, epoch=0, return_f=False)
        
        loss = 0.5 * criterion(out_1.view(-1), y1.view(-1)) + 0.5 * criterion(out_2.view(-1), y2.view(-1))
        
        opt_net.zero_grad()
        opt_F.zero_grad()
        loss.backward()
        opt_net.step()
        opt_F.step()
        
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
            if torch.cuda.is_available():
                torch.set_default_tensor_type(torch.FloatTensor)
                if hasattr(torch, 'set_default_device'):
                    torch.set_default_device('cpu')
            print(f"\nRunning HPO for Dataset: {dataset}, Batch: {batch}")
            train_bats = train_test_split[dataset][batch]['train']
            
            with open(os.path.join(config_dir, 'initial_knee_points.json'), 'r') as f:
                initial_knee_points = json.load(f)
            knee_points_batch = initial_knee_points[dataset][batch]
            data_root = os.path.join(project_root, 'data', 'Processed')
            # Load dummy dataloader for TST & DeepOPINN
            dataloader = get_dataloader(data_root, dataset, train_bats, batch_size=64, shuffle=True, initial_knee_points=knee_points_batch)
            # Recreate dataloader without shuffle to prevent PyTorch generator device mismatch errors
            dataloader = data.DataLoader(dataloader.dataset, batch_size=64, shuffle=False)
            
            # 1. Optimize TST
            study_tst = optuna.create_study(direction='minimize')
            study_tst.optimize(lambda trial: optimize_tst(trial, dataloader, device), n_trials=200)
            save_optimized_architecture(project_root, dataset, batch, "TST", study_tst.best_params)
            
            # 2. Optimize DeepOPINN
            study_deepopinn = optuna.create_study(direction='minimize')
            study_deepopinn.optimize(lambda trial: optimize_deepopinn(trial, dataloader, device, project_root), n_trials=200)
            save_optimized_architecture(project_root, dataset, batch, "DeepOPINN", study_deepopinn.best_params)
            
            # 3. Optimize Fusion MLP
            study_fusion = optuna.create_study(direction='minimize')
            study_fusion.optimize(lambda trial: optimize_fusion(trial, dataloader, device), n_trials=200)
            save_optimized_architecture(project_root, dataset, batch, "FusionMLP", study_fusion.best_params)
            
            # 4. Optimize LAX
            lax_dataset_obj = LaxDataset(train_bats, dataset, project_root)
            if len(lax_dataset_obj) > 0:
                lax_dataloader = data.DataLoader(lax_dataset_obj, batch_size=64, shuffle=False)
                
                all_x = np.array([s[0] for s in lax_dataset_obj.samples])
                X_mean = torch.tensor(np.mean(all_x[:, :-1], axis=0), dtype=torch.float32).to(device)
                X_std = torch.tensor(np.std(all_x[:, :-1], axis=0), dtype=torch.float32).to(device)
                x_sts = [X_mean, X_std]
                x_dim = all_x.shape[1] - 1
                y_dim = x_dim

                study_lax = optuna.create_study(direction='minimize')
                study_lax.optimize(lambda trial: optimize_lax(trial, lax_dataloader, device, project_root, x_sts, x_dim, y_dim), n_trials=200)
                save_optimized_architecture(project_root, dataset, batch, "LAX", study_lax.best_params)

            
    print("HPO pipeline completed successfully!")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.dirname(__file__))
    run_hpo_pipeline(project_root)
