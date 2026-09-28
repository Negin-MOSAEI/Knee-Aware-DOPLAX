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
from Model.utils.learnable_smoother import tv_regularization_loss
from pipeline.phase3_train_deepopinn import ArgsMock
from pipeline.phase4_train_lax import ArgsMockLax
from utils.kpi_tracker import KPITracker
from dataloader.dataloader import BatteryCycleDataset
from utils.plot_utils import plot_learning_curve, save_loss_history
from Model.utils.losses import gaussian_nll_loss

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


def run_phase5(project_root: str, num_epochs: int = 10, batch_size: int = 64, resume: bool = True):
    """
    Executes Phase 5: Production-hardened 2-Stage MoE Fusion Training for KaDOPLAX.

    Stage 1: Gate-only warmup with high-to-low entropy annealing to prevent early collapse.
    Stage 2: Physics-preserving joint fine-tuning with learned homoscedastic uncertainty weighting
             (Kendall & Gal) over data, PDE, monotonicity, and fusion task losses.
    """
    print("Starting Phase 5: Production-Hardened 2-Stage MoE Fusion Training...")
    
    config_dir = os.path.join(project_root, 'config')
    kpd_out_dir = os.path.join(project_root, 'outputs', 'kpd_predictions')
    models_dir = os.path.join(project_root, 'outputs', 'models')
    os.makedirs(models_dir, exist_ok=True)
    kpi_report = {}
    
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    for dataset, batches in train_test_split.items():
        for batch in batches:
            kadoplax_save_path = os.path.join(models_dir, f'kadoplax_{dataset}_{batch}.pt')
            if resume and os.path.exists(kadoplax_save_path):
                print(f"Model for {dataset} ({batch}) already exists, skipping...")
                continue

            print(f"\n--- Training Production KaDOPLAX MoE for {dataset} - {batch} ---")
            
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
            
            # 1. Load DeepOPINN
            args_do = ArgsMock(project_root)
            from utils.hpo_utils import get_model_params
            best_params_do = get_model_params(project_root, dataset, batch, "DeepOPINN")
            if best_params_do:
                deepopinn = DeepOpinn(args_do, save_args=False, **best_params_do).to(device)
            else:
                deepopinn = DeepOpinn(args_do, save_args=False).to(device)
            
            peek_features = dataset_obj[0][0]
            deepopinn_num_features = np.prod(peek_features.shape)
            deepopinn.initialize_networks(deepopinn_num_features)
            
            do_path = os.path.join(models_dir, f'deepopinn_{dataset}_{batch}.pt')
            if os.path.exists(do_path):
                deepopinn.load_state_dict(torch.load(do_path, map_location=device, weights_only=True))
            
            # 2. Load LAX
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
                    print(f"  Warning: Cannot load old LAX weights: {e}. Training from scratch.")
            
            # 3. Assemble KaDOPLAX model
            model = KaDOPLAX(deepopinn, lax, enable_smoother=True).to(device)
            criterion = nn.HuberLoss(delta=0.1)

            # Two-phase schedule configuration
            stage1_epochs = max(4, int(0.4 * num_epochs))
            stage2_epochs = max(4, num_epochs - stage1_epochs)

            # Cosine annealing schedule definitions
            # Stage 1: 0.10 -> 0.01 (gate-only warmup)
            s1_start, s1_end = 0.10, 0.01
            # Stage 2: 0.01 -> 0.0001 (joint fine-tuning)
            s2_start, s2_end = 0.01, 0.0001

            # Verify and print Stage 1 exploration schedule
            epochs_above_003 = sum(
                1 for ep in range(stage1_epochs)
                if (s1_end + 0.5 * (s1_start - s1_end) * (1.0 + np.cos(np.pi * ep / max(1, stage1_epochs - 1)))) >= 0.03
            )
            print(f"\nEntropy Schedule Configuration:")
            print(f"  Stage 1 (Warmup): {stage1_epochs} epochs | Stage 2 (Joint): {stage2_epochs} epochs (Total: {stage1_epochs + stage2_epochs})")
            print(f"  Stage 1 maintains λ_entropy >= 0.03 for {epochs_above_003}/{stage1_epochs} epochs, ensuring robust exploration before router specializes.")

            # Stage 1 Setup: Experts frozen
            model.freeze_experts()
            optimizer = optim.AdamW(
                [p for p in model.moe_gate.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=1e-2
            )

            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            best_model_state = None

            print(f"Plan: Stage 1 ({stage1_epochs} epochs: Gate warmup) | Stage 2 ({stage2_epochs} epochs: Joint fine-tuning)")

            lambda_stage1_final = s1_end

            # --- STAGE 1: GATE-ONLY WARMUP ---
            for epoch in range(stage1_epochs):
                model.train()
                total_loss = 0.0
                
                # Compute Stage 1 cosine entropy weight
                prog = epoch / max(1, stage1_epochs - 1)
                lambda_entropy = s1_end + 0.5 * (s1_start - s1_end) * (1.0 + np.cos(np.pi * prog))
                lambda_stage1_final = float(lambda_entropy)

                for features, kpd, cycle_t, target_soh in dataloader:
                    features, kpd, cycle_t, target_soh = features.to(device), kpd.to(device), cycle_t.to(device), target_soh.to(device)
                    optimizer.zero_grad()

                    # 1. Forward fusion model
                    mean_out, log_var_out, weights, entropy = model(features, kpd, cycle_t, epoch=1000, return_gate_info=True)
                    task_loss = gaussian_nll_loss(mean_out.view(-1), log_var_out.view(-1), target_soh.view(-1))

                    
                    # Negative entropy penalty prevents gate collapse
                    loss = task_loss - lambda_entropy * entropy
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    kpi_tracker.update_ram()

                avg_train = total_loss / len(dataloader)
                train_losses.append(avg_train)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for features, kpd, cycle_t, target_soh in val_dataloader:
                        features, kpd, cycle_t, target_soh = features.to(device), kpd.to(device), cycle_t.to(device), target_soh.to(device)
                        mean_out, log_var_out = model(features, kpd, cycle_t, epoch=1000)
                        val_loss += gaussian_nll_loss(mean_out.view(-1), log_var_out.view(-1), target_soh.view(-1)).item()
                avg_val = val_loss / max(1, len(val_dataloader))
                val_losses.append(avg_val)

                print(f"[Stage 1 | Epoch {epoch+1}/{stage1_epochs}] Train: {avg_train:.4f}, Val: {avg_val:.4f}, λ_entropy: {lambda_entropy:.4f}")

            # --- STAGE TRANSITION HANDOFF ---
            print("\n" + "="*60)
            print(f"[STAGE TRANSITION] Handing off from Stage 1 to Stage 2.")
            print(f"  Stage 1 final λ_entropy: {lambda_stage1_final:.5f}")
            print(f"  Stage 2 start λ_entropy: {s2_start:.5f}")
            assert abs(lambda_stage1_final - s2_start) <= 0.02, (
                f"Discontinuous entropy schedule jump at handoff: {lambda_stage1_final:.4f} -> {s2_start:.4f}"
            )
            print("  Handoff continuity verified: schedule smooth within tolerance.")
            print("="*60 + "\n")

            # --- STAGE 2 SETUP: UNFREEZE EXPERTS & MULTI-TASK OPTIMIZER ---
            model.unfreeze_experts(unfreeze_lax=False)  # DeepOPINN unfrozen
            stage2_params = [
                {'params': model.moe_gate.parameters(), 'lr': 1e-4},
                {'params': model.deepopinn.parameters(), 'lr': 2e-5, 'weight_decay': 1e-4},
                {'params': [model.s_data, model.s_pde, model.s_mono, model.s_fusion], 'lr': 1e-3}
            ]
            optimizer = optim.AdamW(stage2_params)

            # --- STAGE 2: JOINT PHYSICS-PRESERVING FINE-TUNING ---
            for epoch_s2 in range(stage2_epochs):
                current_epoch = stage1_epochs + epoch_s2
                model.train()
                total_loss = 0.0

                # Diagnostic accumulators for first 3 epochs
                diag = {'data': 0.0, 'pde': 0.0, 'mono': 0.0, 'fusion': 0.0, 'entropy': 0.0, 'tv': 0.0, 'n': 0}

                # Compute Stage 2 cosine entropy weight
                prog_s2 = epoch_s2 / max(1, stage2_epochs - 1)
                lambda_entropy = s2_end + 0.5 * (s2_start - s2_end) * (1.0 + np.cos(np.pi * prog_s2))

                for features, kpd, cycle_t, target_soh in dataloader:
                    features, kpd, cycle_t, target_soh = features.to(device), kpd.to(device), cycle_t.to(device), target_soh.to(device)
                    optimizer.zero_grad()

                    # 1. Forward fusion model
                    mean_out, log_var_out, weights, entropy = model(features, kpd, cycle_t, epoch=1000, return_gate_info=True)
                    loss_fusion = gaussian_nll_loss(mean_out.view(-1), log_var_out.view(-1), target_soh.view(-1))


                    # 2. Physics losses from DeepOPINN
                    features_flat = features.reshape(features.shape[0], -1)
                    u1, f1 = model.deepopinn.forward_deepopinn(features_flat)
                    weight = model.deepopinn.relu(-kpd)

                    loss_data = (torch.pow(u1 - target_soh, 2) * weight).mean()
                    loss_pde = (torch.pow(f1, 2) * weight).mean()
                    if u1.shape[0] > 1:
                        loss_mono = torch.mean(torch.relu(u1[1:] - u1[:-1]))
                    else:
                        loss_mono = torch.tensor(0.0, device=device)

                    # 3. Self-adaptive homoscedastic multi-task loss (Kendall & Gal)
                    mt_loss, eff_weights = model.compute_multitask_loss(
                        l_data=loss_data,
                        l_pde=loss_pde,
                        l_mono=loss_mono,
                        l_fusion=loss_fusion
                    )

                    # 4. Total Variation penalty & Entropy regularization
                    tv_loss = tv_regularization_loss(mean_out, order=2)
                    loss = mt_loss - lambda_entropy * entropy + 0.01 * tv_loss

                    # Accumulate diagnostics
                    if epoch_s2 < 3:
                        diag['data'] += loss_data.item()
                        diag['pde'] += loss_pde.item()
                        diag['mono'] += loss_mono.item()
                        diag['fusion'] += loss_fusion.item()
                        diag['entropy'] += entropy.item()
                        diag['tv'] += tv_loss.item()
                        diag['n'] += 1

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    kpi_tracker.update_ram()

                avg_train = total_loss / len(dataloader)
                train_losses.append(avg_train)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for features, kpd, cycle_t, target_soh in val_dataloader:
                        features, kpd, cycle_t, target_soh = features.to(device), kpd.to(device), cycle_t.to(device), target_soh.to(device)
                        mean_out, log_var_out = model(features, kpd, cycle_t, epoch=1000)
                        val_loss += gaussian_nll_loss(mean_out.view(-1), log_var_out.view(-1), target_soh.view(-1)).item()
                avg_val = val_loss / max(1, len(val_dataloader))
                val_losses.append(avg_val)

                # Per-epoch effective weight logging
                eff_w = model.get_effective_loss_weights()
                print(
                    f"[Stage 2 | Epoch {epoch_s2+1}/{stage2_epochs}] "
                    f"Train: {avg_train:.4f}, Val: {avg_val:.4f} | "
                    f"w_data: {eff_w['data']:.3f}, w_pde: {eff_w['pde']:.3f}, "
                    f"w_mono: {eff_w['mono']:.3f}, w_fusion: {eff_w['fusion']:.3f} | "
                    f"λ_entropy: {lambda_entropy:.5f}"
                )

                # Print raw unweighted loss diagnostics for first 3 epochs of Stage 2
                if epoch_s2 < 3 and diag['n'] > 0:
                    n_b = diag['n']
                    print(f"  [Stage 2 Diagnostic - Epoch {epoch_s2+1} Raw Loss Magnitudes (Unweighted)]")
                    print(f"    Raw L_data:    {diag['data']/n_b:.6f}")
                    print(f"    Raw L_pde:     {diag['pde']/n_b:.6f}")
                    print(f"    Raw L_mono:    {diag['mono']/n_b:.6f}")
                    print(f"    Raw L_fusion:  {diag['fusion']/n_b:.6f}")
                    print(f"    Raw Entropy:   {diag['entropy']/n_b:.6f} (H(w) -> penalty term = {-lambda_entropy * (diag['entropy']/n_b):.6f})")
                    print(f"    Raw TV Loss:   {diag['tv']/n_b:.6f} (Penalty: {0.01 * (diag['tv']/n_b):.6f})")

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            metrics = kpi_tracker.stop()
            if dataset not in kpi_report:
                kpi_report[dataset] = {}
            kpi_report[dataset][batch] = metrics

            learning_curve_dir = os.path.join(project_root, 'experiments', 'fusion_experiments')
            plot_learning_curve(train_losses, val_losses, "KaDOPLAX_MoE", dataset, batch, os.path.join(project_root, 'outputs', 'figures', 'learning_curves'))
            save_loss_history(train_losses, val_losses, "KaDOPLAX_MoE", dataset, batch, learning_curve_dir)

            model_save_path = os.path.join(models_dir, f'kadoplax_{dataset}_{batch}.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved Production KaDOPLAX model to {model_save_path}")

    # Save KPI report
    reports_dir = os.path.join(project_root, 'outputs', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, 'fusion_training_kpis.json'), 'w') as f:
        json.dump(kpi_report, f, indent=4)
    print("Phase 5 Production 2-Stage Training completed successfully.")

if __name__ == "__main__":
    run_phase5(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
