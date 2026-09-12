import torch
from dataloader import load_battery_dataloader
from Model import (
    TemporalConvNet,
    DeepOPINN,
    OptimizationNetwork,
    FusionMLP,
    MLP_Bagging_NN
)

def test():
    print("Testing models with real battery batch from DataLoader...")
    loader = load_battery_dataloader('XJTU', '2C', data_root='../DOPLAX-SOH-main 4/data/Processed')
    
    batch = None
    for b in loader['train']:
        batch = b
        break
        
    x = batch['x']                   # [L_i, 17]
    y = batch['y']                   # [L_i, 1]
    x1 = batch['x1']                 # [L_i - 1, 17]
    x2 = batch['x2']                 # [L_i - 1, 17]
    y1 = batch['y1']                 # [L_i - 1, 1]
    y2 = batch['y2']                 # [L_i - 1, 1]
    knee_dist = batch['knee_distance'] # [L_i, 1]
    
    print(f"Loaded Battery: {batch['battery_name']}, Length: {batch['length']} cycles, Features: {x.shape}")
    
    # 1. Test TCN
    print("\n1. Testing TemporalConvNet (TCN for Knee Distance)...")
    tcn = TemporalConvNet(num_inputs=17, num_channels=[32, 64, 32], output_dim=1)
    pred_knee_dist = tcn(x)
    print(f"  TCN input: {x.shape} -> TCN output: {pred_knee_dist.shape}")
    assert pred_knee_dist.shape == knee_dist.shape, "TCN output shape mismatch!"
    
    # 2. Test DeepOPINN
    print("\n2. Testing DeepOPINN...")
    deepopinn = DeepOPINN(input_dim=17, hidden_dim=32, layers_num=3)
    _, u_deepopinn = deepopinn(x)
    print(f"  DeepOPINN forward output: {u_deepopinn.shape}")
    loss_dict = deepopinn.compute_loss(x1, x2, y1, y2)
    print(f"  DeepOPINN loss: total={loss_dict['total_loss'].item():.6f}, data={loss_dict['loss_data'].item():.6f}, pde={loss_dict['loss_pde'].item():.6f}")
    
    # 3. Test LAX Network
    print("\n3. Testing LAX OptimizationNetwork...")
    lax = OptimizationNetwork(x_dim=16, y_dim=16, h_dim=16)
    u_lax = lax(x=x[:, :-1], t=x[:, -1:])
    print(f"  LAX output: {u_lax.shape}")
    
    # 4. Test FusionMLP
    print("\n4. Testing FusionMLP (DeepOPINN + LAX + TCN Knee Feature)...")
    fusion = FusionMLP(input_dim=2, hidden_dims=[32, 16], include_knee_feature=True)
    u_final = fusion(u_deepopinn=u_deepopinn, u_lax=u_lax, knee_distance=pred_knee_dist)
    print(f"  FusionMLP final SOH output: {u_final.shape}")
    
    # 5. Test Bagging_u
    print("\n5. Testing Bagging MLP...")
    bagging = MLP_Bagging_NN(input_dim=2, hidden_dim=[32])
    u_bag = bagging(torch.cat([u_deepopinn, u_lax], dim=-1))
    print(f"  Bagging output: {u_bag.shape}")
    
    print("\n========================================")
    print("SUCCESS: All 4 models (TCN, DeepOPINN, LAX, Fusion MLP) verified successfully!")

if __name__ == "__main__":
    test()
