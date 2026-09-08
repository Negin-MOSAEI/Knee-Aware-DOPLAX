import torch
import torch.nn as nn
import deepxde as dde
import torch
torch.set_default_device('cpu')
torch.set_default_tensor_type(torch.FloatTensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model(nn.Module):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation="relu", kernel_initializer="Glorot normal"):
        super().__init__()
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()
        # Branch network processes function space
        self.branch = self._build_network(layer_sizes_branch)
        # Trunk network processes coordinate space
        self.trunk = self._build_network(layer_sizes_trunk)
        self._init_weights(kernel_initializer)

    def _build_network(self, layer_sizes):
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(self.activation)
        return nn.Sequential(*layers)

    def _init_weights(self, initializer):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if initializer == "Glorot normal":
                    nn.init.xavier_normal_(m.weight)
                elif initializer == "He normal":
                    nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        func, spatial = inputs
        
        # Process through networks
        branch_out = self.branch(func)  # output: [batch_size, branch_out_features]
        trunk_out = self.trunk(spatial)  # output: [num_coordinates, trunk_out_features]

        # Compute dot product for each (sample, coordinate) pair
        # Changed from bi,bi->b to bi,ci->bc to handle different batch sizes
        output = torch.einsum('bi,ci->bc', branch_out, trunk_out)  # [batch_size, num_coordinates]
        
        return output



if __name__ == "__main__":
    # 	# Assuming:
# 	# X: (389, 2845) - 389 samples, 2845 features each
# 	# coordinate: (2845, 1) - 2845 spatial coordinates
# 	import pandas as pd
# 	import numpy as np

# 	df = pd.read_csv("data/XJTU data/current_capacity/charge/Batch-1/2C_battery-1.csv").set_index(['cycle', 'dtype'])
# 	X, Y = df.loc[(slice(None), 'charge_current'), :], df.loc[(slice(None), 'charge_capacity'), :]
# 	# coordinate = np.linspace(0, 1, X.shape[1]).reshape(-1, 1) # 2845 coordinates
# 	coordinate = np.linspace(0, 1, int(X.shape[1]/2)).reshape(-1, 1) # 1422 coordinates

# 	net = DeepONet(
# 	    layer_sizes_branch=[2845, 64, 64],  # Input features must match X.shape[1]
# 	    layer_sizes_trunk=[1, 64, 64],      # Input dim must match coordinate.shape[1]
# 	    activation="relu",
# 	    kernel_initializer="Glorot normal"
# 	).to(device)

# 	# Convert inputs
# 	X_tensor = torch.as_tensor(X.values, dtype=torch.float32)  # (389, 2845)
# 	coord_tensor = torch.as_tensor(coordinate, dtype=torch.float32)  # (2845, 1)

# 	# Forward pass
# 	inputs = (X_tensor, coord_tensor).to(device)
# 	y_pred = net(inputs)  # Output shape: (389, 2845)
	# y_pred.shape
    
    net = dde.nn.DeepONetCartesianProd(
        layer_sizes_branch=[2845, 64, 64],
        layer_sizes_trunk=[1, 64, 64],
        activation="relu",
        kernel_initializer="Glorot normal",
    ).to(device)
    
    X_tensor = torch.randn(size=(389, 2845)).to(device)
    coord_tensor = torch.randn(size=(1, 1)).to(device)
    res = net((X_tensor, coord_tensor))
    print('result shape:', res.shape)

    
