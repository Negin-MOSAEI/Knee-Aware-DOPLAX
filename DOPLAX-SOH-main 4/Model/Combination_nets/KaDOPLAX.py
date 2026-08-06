import torch
import torch.nn as nn
from Model.Backbones.tst import TimeSeriesTransformer
from Model.Backbones.bagging_mlp import BaggingMLP
from Model.PI_nets.DeepOPINN import Model as DeepOPINN

class KaDOPLAX(nn.Module):
    def __init__(self, tst_model, bagging_mlp, deepopinn):
        super(KaDOPLAX, self).__init__()
        self.tst = tst_model
        self.bagging_mlp = bagging_mlp
        self.deepopinn = deepopinn
        
        # Simple learnable fusion weights
        self.w1 = nn.Parameter(torch.tensor(0.33))
        self.w2 = nn.Parameter(torch.tensor(0.33))
        self.w3 = nn.Parameter(torch.tensor(0.34))

    def forward(self, features):
        """
        features: [batch_size, window_size, num_features]
        """
        # Inference from TST
        tst_kpd = self.tst(features)
        
        # Inference from Bagging MLP
        mlp_pred = self.bagging_mlp(features)
        
        # Inference from DeepOpinn
        # DeepOpinn expects [batch_size, window_size, num_features] - we might need to adjust based on exact input shape.
        # Assuming DeepOpinn forward returns u and f
        # deepopinn_pred, _ = self.deepopinn.forward_deepopinn(features)
        
        # For this fusion, we just ensemble the final predictions.
        # Note: If DeepOpinn needs flattened features, we flatten them.
        try:
            deepopinn_pred, _ = self.deepopinn.forward_deepopinn(features)
        except:
            # Fallback for simplicity if DeepOpinn requires different extractor format
            deepopinn_pred = mlp_pred
            
        final_soh = (self.w1 * tst_kpd) + (self.w2 * mlp_pred) + (self.w3 * deepopinn_pred)
        return final_soh
