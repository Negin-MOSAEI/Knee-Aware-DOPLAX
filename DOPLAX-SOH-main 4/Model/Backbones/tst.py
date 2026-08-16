import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1, x_sts=None):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        
        if x_sts is not None:
            self.register_buffer('X_mean', x_sts[0].to(torch.float32))
            self.register_buffer('X_std', x_sts[1].to(torch.float32))
        else:
            self.register_buffer('X_mean', torch.zeros(num_features, dtype=torch.float32))
            self.register_buffer('X_std', torch.ones(num_features, dtype=torch.float32))
            
        self.input_linear = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, src):
        # src shape: [batch_size, window_size, num_features]
        # Standardize inputs robustly
        src = (src - self.X_mean) / torch.clamp(self.X_std, min=1e-5)
        
        # Transformer expects [sequence_length, batch_size, num_features]
        src = src.transpose(0, 1)
        
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src)
        
        # Take the output of the last time step
        output = output[-1, :, :]
        
        prediction = self.decoder(output)
        return prediction
