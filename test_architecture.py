import os
import sys
import torch

# Append the DOPLAX-SOH-main 4 directory to path to import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DOPLAX-SOH-main 4'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Model.PI_nets.DeepOPINN import Model as DeepOPINN
from Model.PI_nets.LAX import OptimizationNetwork as LAXModel
from Model.Combination_nets.KaDOPLAX import KaDOPLAX
from Model.PI_nets.FastSurrogate import FastSurrogateMLP
from Model.utils.losses import gaussian_nll_loss
from pipeline.phase3_train_deepopinn import ArgsMock as ArgsMockDeepOPINN
from pipeline.phase4_train_lax import ArgsMockLax

def run_dry_run():
    print("🚀 Starting Architectural Dry Run...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Initialization ---
    print("[1/5] Initializing Models...")
    # DeepOPINN
    args_do = ArgsMockDeepOPINN(project_root)
    deepopinn = DeepOPINN(args_do, save_args=False).to(device)
    # Initialize networks based on flattened feature size. 
    # Let's say window=40, features=3 -> 120
    deepopinn.initialize_networks(120)

    # LAX
    args_lax = ArgsMockLax(project_root)
    lax_num_features = 3
    X_mean = torch.zeros(lax_num_features, dtype=torch.float32).to(device)
    X_std = torch.ones(lax_num_features, dtype=torch.float32).to(device)
    x_sts = [X_mean, X_std]
    y_dim = lax_num_features
    lax = LAXModel(x_sts, args_lax, lax_num_features, y_dim).to(device)

    # KaDOPLAX Fusion
    kadoplax = KaDOPLAX(deepopinn, lax, enable_smoother=True).to(device)

    # Fast Surrogate
    surrogate = FastSurrogateMLP(window_size=40, num_features=3, hidden_dim=64).to(device)

    print("✅ Models initialized successfully.\n")

    # --- 2. Dummy Data Creation ---
    print("[2/5] Creating Dummy Tensors...")
    B = 4
    W = 40
    F = 3
    features = torch.randn(B, W, F).to(device)
    kpd = torch.randn(B, 1).to(device)
    cycle_t = torch.randn(B, 1).to(device)
    target_soh = torch.rand(B, 1).to(device)
    print("✅ Dummy tensors created successfully.\n")

    # --- 3. Forward Pass KaDOPLAX ---
    print("[3/5] Testing KaDOPLAX Forward Pass...")
    try:
        # Disable inference mode in LAX just in case, though KaDOPLAX sets it to True in init
        mean_out, log_var_out, weights, entropy = kadoplax(
            features, kpd, cycle_t, epoch=1000, return_gate_info=True
        )
        assert mean_out.shape == (B, 1), f"Expected mean shape (4,1), got {mean_out.shape}"
        assert log_var_out.shape == (B, 1), f"Expected log_var shape (4,1), got {log_var_out.shape}"
        print(f"  Mean Output Shape: {mean_out.shape}")
        print(f"  Log-Var Output Shape: {log_var_out.shape}")
        print("✅ KaDOPLAX forward pass successful.\n")
    except Exception as e:
        print(f"❌ KaDOPLAX forward pass failed: {e}")
        raise e

    # --- 4. Testing UQ & Losses ---
    print("[4/5] Testing UQ (predict_with_uq) & Gaussian NLL Loss...")
    try:
        # Gaussian NLL
        loss = gaussian_nll_loss(mean_out.view(-1), log_var_out.view(-1), target_soh.view(-1))
        assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf"
        print(f"  Gaussian NLL Loss Computed: {loss.item():.4f}")

        # predict_with_uq
        final_mean, total_std = kadoplax.predict_with_uq(features, kpd, cycle_t, n_samples=5, epoch=1000)
        assert final_mean.shape == (B, 1), f"Expected final_mean shape (4,1), got {final_mean.shape}"
        assert total_std.shape == (B, 1), f"Expected total_std shape (4,1), got {total_std.shape}"
        print(f"  predict_with_uq Mean Shape: {final_mean.shape}")
        print(f"  predict_with_uq Std Shape: {total_std.shape}")
        print("✅ UQ and Loss computation successful.\n")
    except Exception as e:
        print(f"❌ UQ / Loss testing failed: {e}")
        raise e

    # --- 5. Forward Pass Fast Surrogate ---
    print("[5/5] Testing FastSurrogateMLP Forward Pass...")
    try:
        surrogate_out = surrogate(features)
        assert surrogate_out.shape == (B, 1), f"Expected surrogate shape (4,1), got {surrogate_out.shape}"
        print(f"  Surrogate Output Shape: {surrogate_out.shape}")
        print("✅ FastSurrogateMLP forward pass successful.\n")
    except Exception as e:
        print(f"❌ FastSurrogateMLP forward pass failed: {e}")
        raise e

    print("=" * 60)
    print("✅ Dry Run Successful: All architectural changes and tensor shapes are perfectly aligned!")
    print("=" * 60)

if __name__ == "__main__":
    run_dry_run()
