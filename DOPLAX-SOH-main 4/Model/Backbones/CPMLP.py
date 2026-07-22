import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'



class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        out = self.ln(self.dropout(out) + x)
        return out



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_var = configs.num_var
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.charge_discharge_length = configs.charge_discharge_length
        self.drop_rate = configs.dropout
        self.e_layers = configs.e_layers
        self.intra_flatten = nn.Flatten(start_dim=2)
        self.intra_embed = nn.Linear(self.charge_discharge_length*self.num_var, self.d_model)
        self.intra_MLP = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate) for _ in range(configs.e_layers)])

        if self.d_model < self.charge_discharge_length:
            # calculates dimensions dynamically
            self.inter_flatten = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LazyLinear(self.charge_discharge_length*self.d_model),  # Infers input size
                nn.Linear(self.charge_discharge_length*self.d_model, self.charge_discharge_length),
                nn.Linear(self.charge_discharge_length, self.d_model)
            )
        else:
            self.inter_flatten = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LazyLinear(self.charge_discharge_length*self.d_model),  # Infers input size
                nn.Linear(self.charge_discharge_length*self.d_model, self.d_model),
            )


    def forward(self, cycle_curve_data): 
        '''
        cycle_curve_data: [B, L, charge_discharge_length, num_var]
        curve_attn_mask: [B, L]
        '''
        cycle_curve_data = self.intra_flatten(cycle_curve_data) # [B, L, charge_discharge_length * num_var]
        cycle_curve_data = self.intra_embed(cycle_curve_data)
        for i in range(self.e_layers):
            cycle_curve_data = self.intra_MLP[i](cycle_curve_data) # [B, L, d_model]

        # Calculate flattened dimension dynamically
        batch_size = cycle_curve_data.size(0)
        flattened_size = cycle_curve_data.size(1) * cycle_curve_data.size(2)  # L * d_model
        
        # Replace the first Linear layer's weight if using LazyLinear
        if hasattr(self.inter_flatten[1], 'reset_parameters'):
            self.inter_flatten[1] = nn.Linear(flattened_size, self.charge_discharge_length*self.d_model).to(cycle_curve_data.device)
        
        cycle_curve_data = self.inter_flatten(cycle_curve_data)
        return cycle_curve_data # [B, L]


        
# def get_FE_args():
#     import argparse
#     parser = argparse.ArgumentParser('Hyper Parameters for XJTU dataset')

#     # basic config
#     parser.add_argument('--charge_discharge_length', type=int, default=100, help='The resampled length for charge and discharge curves')
#     parser.add_argument('--seq_len', type=int, default=1, help='input sequence length')

#     # Datal Loader related
#     parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
#     parser.add_argument('--batch',type=str,default='2C',choices=['2C','3C','R2.5','R3','RW','satellite'])
#     parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
#     parser.add_argument('--batch_size', type=int, default=256, help='batch size')
#     parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
#     parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")
#     parser.add_argument('--process_batches', type=bool, default=True, help='train features extractor based on one batch or all batches')

#     # scheduler related
#     parser.add_argument('--epochs', type=int, default=200, help='epoch')
#     parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
#     parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
#     parser.add_argument('--lr', type=float, default=0.01, help='base lr')
#     parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
#     parser.add_argument('--lr_F', type=float, default=0.001, help='lr of F')
#     parser.add_argument('--early_stop', type=int, default=40, help='Stop training when a monitored quantity (training or testing loss) has stopped improving')

#     # Model related
#     parser.add_argument('--freq', type=str, default='s', help='freq for time features encoding, '
#                          'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
#                          'you can also use more detailed freq like 15min or 3h')
#     parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
#     parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
#     parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
#     parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
#     parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
#     parser.add_argument('--factor', type=int, default=1, help='attn factor')
#     parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
#     parser.add_argument('--embed', type=str, default='fixed',
#                         help='time features encoding, options:[timeF, fixed, learned]')
#     parser.add_argument('--activation', type=str, default='relu', help='activation')

#     # optimization
#     parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights, [MSE, MAE, MAPE, RMSE]')

#     # Directory related
#     parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
#     parser.add_argument('--save_folder', type=str, default='results of reviewer/', help='A folder to save the results')

#     args = parser.parse_args()
#     return args


    
# if __name__ == "__main__":
#     args = get_FE_args()

#     # cycle_curve_data: [B, early_cycle, fixed_len, num_var]
#     B = args.batch_size
#     L = 1800
#     fixed_len = args.charge_discharge_length
#     variables = args.dtypes
#     variables.remove('capacity')
#     num_var = len(variables)
#     args.num_var = num_var
#     # cycle_curve_data: [B, early_cycle, fixed_len, num_var]
#     t = torch.randn(size=(B, L))
#     t = t.repeat(1, fixed_len * num_var) # Repeat along the second dimension
#     cycle_curve_data = t.view(B, L, fixed_len, num_var).to(device)
    
#     model = Model(args).to(device)
#     res = model(cycle_curve_data)
#     print(res.shape)
    
