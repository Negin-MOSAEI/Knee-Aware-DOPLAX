import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class BaggingMLP(nn.Module):
    def __init__(self, num_models: int = 5, input_dim: int = 40 * 3, hidden_dim: int = 64, num_layers: int = 3):
        super(BaggingMLP, self).__init__()
        self.models = nn.ModuleList([MLP(input_dim, hidden_dim, num_layers) for _ in range(num_models)])

    def forward(self, x, return_std=False):
        # x shape: [batch_size, window_size, num_features]
        # Flatten for MLP: [batch_size, window_size * num_features]
        x_flat = x.view(x.size(0), -1)
        
        predictions = []
        for model in self.models:
            predictions.append(model(x_flat))
            
        # Stack and average predictions
        stacked = torch.stack(predictions, dim=1)
        mean_pred = torch.mean(stacked, dim=1)
        
        if return_std:
            std_pred = torch.std(stacked, dim=1)
            return mean_pred, std_pred
            
        return mean_pred
