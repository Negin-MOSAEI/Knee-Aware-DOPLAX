import torch
import torch.nn as nn
from Model.utils.util import get_clones
from Model.utils.activation_functions import WaveAct



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()
        self.dropout = nn.Dropout(dropout)
        # Added LayerNorm for stability
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.dropout(self.attn(x2, x2, x2)[0])
        x = self.norm1(x)  # normalize after attention
        x2 = self.act2(x)
        x = x + self.ff(x2)
        x = self.norm2(x)  # normalize after feed-forward
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super(DecoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()
        self.dropout = nn.Dropout(dropout)
        # Added LayerNorm for stability
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, e_outputs):
        x2 = self.act1(x)
        x = x + self.dropout(self.attn(x2, e_outputs, e_outputs)[0])
        x = self.norm1(x)  # normalize after cross-attention
        x2 = self.act2(x)
        x = x + self.ff(x2)
        x = self.norm2(x)  # normalize after feed-forward
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.act = WaveAct()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.act = WaveAct()

    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)


class Model(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads, dropout):
        super(Model, self).__init__()

        self.linear_emb = nn.Linear(17, d_model)

        self.encoder = Encoder(d_model, N, heads, dropout)
        self.decoder = Decoder(d_model, N, heads, dropout)
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Dropout(p=dropout),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Dropout(p=dropout),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, xt):
        src = self.linear_emb(xt)
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)
        return output
