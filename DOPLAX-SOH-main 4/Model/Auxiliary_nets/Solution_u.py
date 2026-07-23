import torch
import torch.nn as nn
from Model.utils.activation_functions import Sin
from Model.Auxiliary_nets.MLP import MLP



class Predictor(nn.Module):
    def __init__(self, input_dim=40, dropout=0.2):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, input_dim),
            Sin(),
            nn.Linear(input_dim,1)
        )
        self.input_dim = input_dim


    def forward(self,x):
        return self.net(x)



class Solution_u(nn.Module): 
    def __init__(self, input_dim=17, output_dim=32, layers_num=3, hidden_dim=60, dropout=0.2, LAX_model=None):
        super(Solution_u, self).__init__()
        self.LAX_model = LAX_model
        self.encoder = MLP(input_dim=input_dim, output_dim=output_dim, layers_num=layers_num, hidden_dim=hidden_dim, dropout=dropout) 
        if self.LAX_model is None:
            self.predictor = Predictor(input_dim=output_dim, dropout=dropout)
        else:
            self.predictor = self.LAX_model
        self._init_()


    def get_embedding(self,x):
        return self.encoder(x)


    def forward(self, x, epoch=None):
        u_encoded = self.encoder(x)
        if self.LAX_model is None:
            # Used MLP as a predictor here (MLP Predictor)
            u = self.predictor(u_encoded)
        else:
            t = x[:, -1]
            # Used LAX as a predictor here (LAX Predictor)
            u = self.predictor(x=u_encoded, t=t, epoch=epoch)
        return u_encoded, u


    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
