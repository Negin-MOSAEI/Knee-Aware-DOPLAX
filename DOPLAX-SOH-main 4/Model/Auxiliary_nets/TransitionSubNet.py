import torch
import torch.nn as nn
from Model.utils.activation_functions import Sin



class TransitionSubNet(nn.Module): 
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super(TransitionSubNet, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            Sin(),
            nn.Linear(input_dim, output_dim)
        )
        self._init_()

    def forward(self,x):
        return self.net(x)

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)

                

class TrunkNet(nn.Module): 
    def __init__(self, trunk_hidden):
        super(TrunkNet, self).__init__()
        trunk_layers = []
        in_features = 1
        for out_features in trunk_hidden:
            trunk_layers.append(nn.Linear(in_features, out_features))
            trunk_layers.append(nn.LayerNorm(out_features))
            trunk_layers.append(nn.ReLU())
            in_features = out_features
        self.trunk = nn.Sequential(*trunk_layers)
        self._init_()

    def forward(self,x):
        return self.trunk(x)

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)


