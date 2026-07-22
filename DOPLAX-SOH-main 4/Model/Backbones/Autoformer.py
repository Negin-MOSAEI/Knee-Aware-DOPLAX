import torch
import torch.nn as nn
from Model.Backbones.layers.Embed import DataEmbedding_wo_pos
from Model.Backbones.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from Model.Backbones.layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm
from Model.utils.util import get_feature_extractor_training_info
device = 'cuda' if torch.cuda.is_available() else 'cpu'


        
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.d_model = configs.d_model

        # Ensure d_model is even for positional embeddings
        if self.d_model % 2 != 0:
            self.d_model += 1
            self.is_even = False
        else:
            self.is_even = True

        # 1. Project input to d_model channels
        self.input_proj = nn.Conv1d(1, self.d_model, kernel_size=1).to(device)
        
        # 2. Autoformer components with proper temporal embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            c_in=self.d_model,  
            d_model=self.d_model,
            embed_type=self.configs.embed,  
            freq=self.configs.freq,       
            dropout=self.configs.dropout
        ).to(device)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.configs.factor, 
                                       attention_dropout=self.configs.dropout),
                        self.d_model, self.configs.n_heads
                    ),
                    self.d_model,
                    self.configs.d_ff,
                    moving_avg=self.configs.moving_avg,
                    dropout=self.configs.dropout,
                    activation=self.configs.activation
                ) for _ in range(self.configs.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        ).to(device)
        
        # 3. Feature aggregation
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1).to(device)

        
    def forward(self, x):
        # x shape: [B, L] where last column is cycle index
        
        # # Separate features and time information
        features = x[:, :-1]  # All columns except last
        time_info = x[:, -1:]  # Last column (cycle index)
        
        # 1. Project input features
        # x_proj = x.unsqueeze(1) # [B, 1, L-1]
        x_proj = features.unsqueeze(1)  # [B, 1, L-1]
        x_proj = self.input_proj(x_proj)  # [B, d_model, L-1]
        
        # 2. Autoformer processing with temporal embedding
        x_proj = x_proj.transpose(1, 2)  # [B, L-1, d_model]
        enc_out = self.enc_embedding(x_proj, None) # [B, L-1, d_model]
        enc_out, _ = self.encoder(enc_out) # [B, L-1, d_model]
        
        # 3. Aggregate features
        features = self.adaptive_pool(enc_out.transpose(1, 2))  # [B, d_model, 1]  
        features = features.squeeze(-1)[:, :self.configs.d_model] # [B, d_model]

        return features



def get_FE_args():
    import argparse
    parser = argparse.ArgumentParser('Hyper Parameters for XJTU dataset')

    # basic config
    parser.add_argument('--charge_length', type=int, default=2000, help='The resampled length for charge curves')
    parser.add_argument('--discharge_length', type=int, default=4000, help='The resampled length for discharge curves')
    parser.add_argument('--seq_len', type=int, default=1, help='input sequence length')

    # Datal Loader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch',type=str,default='2C',choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")
    parser.add_argument('--process_batches', type=bool, default=True, help='train features extractor based on one batch or all batches')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.01, help='base lr')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
    parser.add_argument('--lr_F', type=float, default=0.001, help='lr of F')
    parser.add_argument('--early_stop', type=int, default=40, help='Stop training when a monitored quantity (training or testing loss) has stopped improving')

    # Model related
    parser.add_argument('--freq', type=str, default='s', help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='relu', help='activation')

    # optimization
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights, [MSE, MAE, MAPE, RMSE]')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results of reviewer/', help='A folder to save the results')

    args = parser.parse_args()
    return args

    
# if __name__ == "__main__":
#     args = get_FE_args()
#     tensor = torch.randint(low=0, high=60, size=(256, 2501)).float().to(device)
#     model = Model(args).to(device)
#     res = model(tensor)
#     print(res.shape)
    
