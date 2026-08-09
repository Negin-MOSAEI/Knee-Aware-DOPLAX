import torch
import torch.nn as nn
from Model.Backbones.bagging_mlp import BaggingMLP
from Model.PI_nets.DeepOPINN import Model as DeepOPINN
from Model.PI_nets.LAX import OptimizationNetwork as LAXModel

class KaDOPLAX(nn.Module):
    def __init__(self, deepopinn_model, lax_model, bagging_mlp):
        super(KaDOPLAX, self).__init__()
        self.deepopinn = deepopinn_model
        self.lax = lax_model
        self.fusion_mlp = bagging_mlp
        
        # Freeze the pre-trained DeepOPINN and LAX models
        for param in self.deepopinn.parameters():
            param.requires_grad = False
        for param in self.lax.parameters():
            param.requires_grad = False

    def forward(self, features, kpd, cycle_t, epoch=None, return_all=False):
        """
        features: [batch_size, window_size, num_features]
        kpd: [batch_size, 1] or similar
        cycle_t: [batch_size, 1]
        """
        # We need to flatten the features for DeepOPINN if it expects 2D inputs
        batch_size = features.shape[0]
        actual_num_features = features.shape[-1]
        
        features_flat = features.reshape(batch_size, -1)
        
        # Inference from DeepOPINN
        # deepopinn might return (final_out, f) or just out based on how predict is structured
        with torch.no_grad():
            try:
                # DeepOPINN
                # DeepOPINN predict returns (u_encoded, u)
                u_1_tuple = self.deepopinn.predict(features_flat)
                u_1 = u_1_tuple[1] if isinstance(u_1_tuple, tuple) else u_1_tuple
            except Exception as e:
                # Fallback to forward_deepopinn which returns (u, f)
                print(f"DeepOPINN predict error: {e}")
                u_1_tuple = self.deepopinn.forward_deepopinn(features_flat)
                u_1 = u_1_tuple[0] if isinstance(u_1_tuple, tuple) else u_1_tuple
            
            # LAX
            last_x = features[:, -1, :] # The last cycle features in the window (batch_size, 3)
            try:
                if epoch is not None:
                    u_2 = self.lax(x=last_x, t=cycle_t, epoch=epoch, return_f=False)
                else:
                    u_2 = self.lax(x=last_x, t=cycle_t, epoch=1000, return_f=False) # Enable optimization
            except Exception as e:
                print(f"LAX Error: {e}")
                u_2 = self.lax(x=last_x, t=cycle_t, epoch=1000)
                
            if isinstance(u_2, tuple):
                u_2 = u_2[0]
                
        # Ensure correct shapes
        if u_1.dim() > 2:
            u_1 = u_1.view(batch_size, -1)
        if u_2.dim() > 2:
            u_2 = u_2.view(batch_size, -1)
        if kpd.dim() > 2:
            kpd = kpd.view(batch_size, -1)
            
        fusion_input = torch.cat([u_1, u_2, kpd], dim=1)
        
        if return_all:
            mean_pred, std_pred = self.fusion_mlp(fusion_input, return_std=True)
            return mean_pred, u_1, u_2, std_pred
        else:
            return self.fusion_mlp(fusion_input, return_std=False)
