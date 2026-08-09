import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from Model.Combination_nets.KaDOPLAX import KaDOPLAX
from Model.Backbones.bagging_mlp import BaggingMLP
from Model.PI_nets.DeepOPINN import Model as DeepOpinn
from Model.PI_nets.LAX import OptimizationNetwork as LAXModel
from dataloader.dataloader import BatteryCycleDataset
from pipeline.phase3_train_deepopinn import ArgsMock
from pipeline.phase4_train_lax import ArgsMockLax

def run_phase6(project_root: str):
    """Executes Phase 6: Final SOH Inference & Plotting using KaDOPLAX Fusion."""
    print("Starting Phase 6: Final SOH Estimation & Plotting...")
    
    config_dir = os.path.join(project_root, 'config')
    models_dir = os.path.join(project_root, 'outputs', 'models')
    kpd_out_dir = os.path.join(project_root, 'outputs', 'kpd_predictions')
    figures_dir = os.path.join(project_root, 'outputs', 'figures', 'final_soh')
    results_out_dir = os.path.join(project_root, 'outputs', 'final_results')
    os.makedirs(results_out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    with open(os.path.join(config_dir, 'initial_knee_points.json'), 'r') as f:
        initial_knee_points = json.load(f)
        
    for dataset, batches in train_test_split.items():
        for batch in batches:
            print(f"Generating SOH plots for {dataset} - {batch}...")
            batch_fig_dir = os.path.join(figures_dir, dataset, batch)
            os.makedirs(batch_fig_dir, exist_ok=True)
            batch_results_dir = os.path.join(results_out_dir, dataset, batch)
            os.makedirs(batch_results_dir, exist_ok=True)
            
            all_bats = train_test_split[dataset][batch]['train'] + train_test_split[dataset][batch]['test']
            train_set = set(train_test_split[dataset][batch]['train'])
            
            if not all_bats:
                continue
                
            # Load KaDOPLAX
            try:
                # 1. DeepOPINN
                args_do = ArgsMock()
                from utils.hpo_utils import get_model_params
                best_params_do = get_model_params(project_root, dataset, batch, "DeepOPINN")
                if best_params_do:
                    deepopinn = DeepOpinn(args_do, save_args=False, **best_params_do).to(device)
                else:
                    deepopinn = DeepOpinn(args_do, save_args=False).to(device)
                
                # Let's peek dataset to be safe
                peek_dataset = BatteryCycleDataset(battery_ids=[all_bats[0]], dataset_name=dataset, data_root=os.path.join(project_root, 'data', 'Processed'))
                f_sample, _, _ = peek_dataset[0]
                deepopinn_num_features = np.prod(f_sample.shape)
                deepopinn.initialize_networks(deepopinn_num_features)
                deepopinn.load_state_dict(torch.load(os.path.join(models_dir, f'deepopinn_{dataset}_{batch}.pt'), map_location=device, weights_only=True))
                deepopinn.eval()
                
                # 2. LAX
                args_lax = ArgsMockLax(project_root)
                lax_num_features = 3 # Hardcoding it to 3 since dataset_obj is created later in phase6
                X_mean = torch.zeros(lax_num_features, dtype=torch.float32).to(device)
                X_std = torch.ones(lax_num_features, dtype=torch.float32).to(device)
                x_sts = [X_mean, X_std]
                y_dim = lax_num_features
                best_params_lax = get_model_params(project_root, dataset, batch, "LAX")
                if best_params_lax:
                    lax = LAXModel(x_sts, args_lax, lax_num_features, y_dim, **best_params_lax).to(device)
                else:
                    lax = LAXModel(x_sts, args_lax, lax_num_features, y_dim).to(device)
                state_dict = torch.load(os.path.join(models_dir, f'lax_{dataset}_{batch}.pt'), map_location=device, weights_only=True)
                if 'y' in state_dict:
                    del state_dict['y']
                lax.load_state_dict(state_dict, strict=False)
                lax.eval()
                
                # 3. Fusion MLP
                best_params_mlp = get_model_params(project_root, dataset, batch, "FusionMLP")
                if best_params_mlp:
                    bagging_mlp = BaggingMLP(num_models=5, input_dim=3, **best_params_mlp).to(device)
                else:
                    bagging_mlp = BaggingMLP(num_models=5, input_dim=3, hidden_dim=64, num_layers=3).to(device)
                
                model = KaDOPLAX(deepopinn, lax, bagging_mlp).to(device)
                kadoplax_state_dict = torch.load(os.path.join(models_dir, f'kadoplax_{dataset}_{batch}.pt'), map_location=device, weights_only=True)
                if 'lax.y' in kadoplax_state_dict:
                    del kadoplax_state_dict['lax.y']
                model.load_state_dict(kadoplax_state_dict, strict=False)
                model.eval()
                
            except Exception as e:
                print(f"Skipping {dataset}-{batch} due to missing models: {e}")
                continue
            
            with torch.no_grad():
                for bat_id in all_bats:
                    is_train = bat_id in train_set
                    kpd_path = os.path.join(kpd_out_dir, dataset, batch, f"{bat_id.replace('/', '_')}_kpd.npy")
                    
                    if not os.path.exists(kpd_path):
                        continue
                        
                    kpd_seq = np.load(kpd_path)
                    
                    dataset_obj = BatteryCycleDataset(
                        battery_ids=[bat_id], 
                        dataset_name=dataset, 
                        data_root=os.path.join(project_root, 'data', 'Processed')
                    )
                    
                    # Extract full sequence
                    features_list = []
                    true_soh_list = []
                    for i in range(len(dataset_obj)):
                        f, _, s = dataset_obj[i]
                        features_list.append(f)
                        true_soh_list.append(s.item())
                        
                    true_soh = np.array(true_soh_list, dtype=np.float32)
                    features = torch.stack([torch.tensor(f, dtype=torch.float32) for f in features_list]).to(device)
                    num_cycles = len(true_soh)
                    valid_len = min(num_cycles, len(kpd_seq))
                    
                    kpd_tensor = torch.tensor(kpd_seq[:valid_len], dtype=torch.float32).unsqueeze(-1).to(device)
                    cycle_t = torch.tensor(np.arange(dataset_obj.window_size, dataset_obj.window_size + valid_len), dtype=torch.float32).unsqueeze(-1).to(device)
                    features = features[:valid_len]
                    
                    # Fusion Inference
                    kadoplax_soh_tensor, u_1_tensor, u_2_tensor, std_pred_tensor = model(features, kpd_tensor, cycle_t, epoch=1000, return_all=True)
                    kadoplax_soh = kadoplax_soh_tensor.cpu().detach().numpy().flatten()
                    u_1_out = u_1_tensor.cpu().detach().numpy().flatten()
                    u_2_out = u_2_tensor.cpu().detach().numpy().flatten()
                    std_pred = std_pred_tensor.cpu().detach().numpy().flatten()
                    
                    # Post-processing (Savitzky-Golay filter)
                    window_length = min(51, valid_len if valid_len % 2 != 0 else valid_len - 1)
                    if window_length > 3:
                        post_kadoplax_soh = savgol_filter(kadoplax_soh, window_length, 3)
                    else:
                        post_kadoplax_soh = kadoplax_soh
                        
                    # Save explicitly as requested
                    save_path_npz = os.path.join(batch_results_dir, f"{bat_id.replace('/', '_')}_results.npz")
                    np.savez(save_path_npz, true_soh=true_soh[:valid_len], predicted_soh=kadoplax_soh, post_processed_soh=post_kadoplax_soh, predicted_kpd=kpd_seq[:valid_len], u_1=u_1_out, u_2=u_2_out, std_pred=std_pred)
                    
                    # Plotting
                    split_label = "[TRAIN]" if is_train else "[TEST]"
                    cycles = np.arange(1, valid_len + 1)
                    bg_color = '#eaffea' if is_train else '#ffeaea'

                    # Plot A: Comprehensive KPD vs SOH Trajectory
                    bat_key = bat_id.split('/')[-1]
                    if dataset in initial_knee_points and batch in initial_knee_points[dataset] and bat_key in initial_knee_points[dataset][batch]:
                        true_knee = initial_knee_points[dataset][batch][bat_key]
                    else:
                        true_knee = 100
                        
                    true_kpd_seq = []
                    for c in cycles:
                        if c < dataset_obj.window_size:
                            true_kpd_seq.append(1.5) # cold start
                        else:
                            true_kpd_seq.append(-np.arctan(c - true_knee))
                            
                    fig_kpd, ax_kpd = plt.subplots(figsize=(10, 5))
                    ax_kpd.set_facecolor(bg_color)
                    
                    ax2 = ax_kpd.twinx()
                    
                    line1 = ax_kpd.plot(cycles, true_soh[:valid_len], color='black', linewidth=2, linestyle='--', label='Ground Truth SOH')
                    line2 = ax2.plot(cycles, true_kpd_seq[:valid_len], color='green', linewidth=2, label='True KPD')
                    line3 = ax2.plot(cycles, kpd_seq[:valid_len], color='red', linewidth=1.5, linestyle=':', label='Predicted KPD (TST)')
                    
                    ax_kpd.set_xlabel('Cycle', fontsize=12)
                    ax_kpd.set_ylabel('State of Health (SOH)', color='black', fontsize=12)
                    ax2.set_ylabel('Knee Point Distance (KPD)', color='red', fontsize=12)
                    
                    plt.title(f'Plot A: KPD vs SOH Trajectory {split_label} - {dataset} ({batch}) - {bat_id}', fontsize=14, fontweight='bold')
                    
                    lines = line1 + line2 + line3
                    labels = [l.get_label() for l in lines]
                    ax_kpd.legend(lines, labels, loc='best', fontsize=10)
                    
                    ax_kpd.grid(True, linestyle=':', alpha=0.6)
                    plt.tight_layout()
                    plt.savefig(os.path.join(batch_fig_dir, f"{bat_id.replace('/', '_')}_Plot_A_KPD.png"), dpi=150, bbox_inches='tight')
                    plt.close(fig_kpd)

                    # Plot B: Comprehensive SOH Comparison
                    fig_soh, ax_soh = plt.subplots(figsize=(14, 8))
                    ax_soh.set_facecolor(bg_color)
                    
                    # Ground Truth
                    ax_soh.plot(cycles, true_soh[:valid_len], label='Ground Truth SOH', color='black', linewidth=2.5, linestyle='--')
                    
                    # DeepOPINN (u_1) and LAX (u_2)
                    ax_soh.plot(cycles, u_1_out, label='DeepOPINN Pipeline ($u_1$)', color='darkorange', linewidth=1.5, alpha=0.7)
                    ax_soh.plot(cycles, u_2_out, label='LAX Pipeline ($u_2$)', color='cyan', linewidth=1.5, alpha=0.7)
                    
                    # Bagging MLP Confidence Intervals
                    ax_soh.fill_between(cycles, kadoplax_soh - std_pred, kadoplax_soh + std_pred, color='purple', alpha=0.2, label='Bagging MLP $\pm 1 \sigma$ Confidence')
                    
                    # Raw KaDOPLAX and Filtered
                    ax_soh.plot(cycles, kadoplax_soh, label='Raw KaDOPLAX Fusion', color='purple', alpha=0.8, linewidth=1.5)
                    ax_soh.plot(cycles, post_kadoplax_soh, label='Post-Processed KaDOPLAX', color='green', linewidth=2.5)
                    
                    plt.title(f'Plot B: Comprehensive SOH Comparison {split_label}\n{dataset} ({batch}) - {bat_id}', fontsize=16, fontweight='bold')
                    plt.xlabel('Cycle', fontsize=14)
                    plt.ylabel('State of Health (SOH)', fontsize=14)
                    plt.grid(True, linestyle=':', alpha=0.6)
                    plt.legend(loc='best', fontsize=12)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(batch_fig_dir, f"{bat_id.replace('/', '_')}_Plot_B_SOH.png"), dpi=150, bbox_inches='tight')
                    plt.close(fig_soh)

if __name__ == "__main__":
    run_phase6(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
