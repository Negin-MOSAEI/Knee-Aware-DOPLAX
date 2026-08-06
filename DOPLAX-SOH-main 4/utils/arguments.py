import argparse



def get_MLP_args(): # pinn set
    parser = argparse.ArgumentParser('Hyper Parameters for training MLP')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default= 256, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=512, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='The layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='The hidden dim of F')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch') # 200
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.01, help='base lr')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
    parser.add_argument('--lr_F', type=float, default=0.001, help='lr of F')


    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.7, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.2, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.05, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.5, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results of reviewer/XJTU results', help='A folder to save the results')

    args = parser.parse_args()
    return args

    
def get_PINN_args(): # pinn set
    parser = argparse.ArgumentParser('Hyper Parameters for training PINN')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default= 256, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=512, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='The layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='The hidden dim of F')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch') # 200
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.01, help='base lr')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
    parser.add_argument('--lr_F', type=float, default=0.001, help='lr of F')


    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.7, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.2, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.05, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.5, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results of reviewer/XJTU results', help='A folder to save the results')

    args = parser.parse_args()
    return args


def get_DeepONet_args(): # optuna2 set
    parser = argparse.ArgumentParser('Hyper Parameters for training DeepONet')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F') 
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=128, help='The hidden dim of F') # 256->in best trial, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=80, help='The hidden dim of F') # 128->in best trial, 80
    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')
    
    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=80, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.006268229580119017, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.006456520169381107, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.006158883939912325, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.005237775841792184, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.006252096490545448, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.001221299541334721, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.00011720460436102193, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0007215089581483758, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.5973286128658127e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.1099224334559236e-05, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=1.447250132592353e-06, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.747286485446226e-05, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00881767573907948, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004916804735455082, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=2.181510135329394e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=4.8990846543190585e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.11809194837918663, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.015956048434866418, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.8843241457826115, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02840429623105723, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.5999950741786956, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.01804601857570287, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.2795906470184894, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.1308092300047794, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='/RUL Group/PINN4SOH/results of reviewer/XJTU results/', help='A folder to save the results')

    args = parser.parse_args()
    return args


def get_PINNsFormer_args(): # optuna2 set
    parser = argparse.ArgumentParser('Hyper Parameters for training PINNsFormer')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F')
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=256, help='The hidden dim of F') # 256, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')
    parser.add_argument('--d_out', type=int, default=1, help='the dimension of the model output')
    parser.add_argument('--d_model_XJTU', type=int, default=32, help='dimension of model')
    parser.add_argument('--d_model_TJU', type=int, default=64, help='dimension of model')
    parser.add_argument('--d_model_MIT', type=int, default=32, help='dimension of model')
    parser.add_argument('--d_model_HUST', type=int, default=128, help='dimension of model')
    parser.add_argument('--d_hidden_XJTU', type=int, default=32, help='dimension of model')
    parser.add_argument('--d_hidden_TJU', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_hidden_MIT', type=int, default=64, help='dimension of model')
    parser.add_argument('--d_hidden_HUST', type=int, default=256, help='dimension of model')
    parser.add_argument('--N_XJTU', type=int, default=3, help='the number of encoder and decoder blocks')
    parser.add_argument('--N_TJU', type=int, default=5, help='the number of encoder and decoder blocks')
    parser.add_argument('--N_MIT', type=int, default=1, help='the number of encoder and decoder blocks')
    parser.add_argument('--N_HUST', type=int, default=3, help='the number of encoder and decoder blocks')
    parser.add_argument('--heads_XJTU', type=int, default=2, help='the number of attention heads')
    parser.add_argument('--heads_TJU', type=int, default=4, help='the number of attention heads')
    parser.add_argument('--heads_MIT', type=int, default=2, help='the number of attention heads')
    parser.add_argument('--heads_HUST', type=int, default=2, help='the number of attention heads')
    parser.add_argument('--transformer_dropout_XJTU', type=float, default=0.1784448096682799, help='transformer_dropout')
    parser.add_argument('--transformer_dropout_TJU', type=float, default=0.2389404128864696, help='transformer_dropout')
    parser.add_argument('--transformer_dropout_MIT', type=float, default=0.15129809797862956, help='transformer_dropout')
    parser.add_argument('--transformer_dropout_HUST', type=float, default=0.2867026986359125, help='transformer_dropout')
    
    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=80, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.002934555763005439, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.0006182130774160487, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.00803896011172995, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.0010338093783875922, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.00015094796720626922, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.0007409057899817629, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.0014405097378976264, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0006627661895345457, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.1692515949240485e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.055699105475732e-06, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=9.764939561657171e-05, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.1379872064564874e-06, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00016323908889202802, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004052220886310363, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=1.233238529948589e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=1.662064130220059e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.5811991915960922, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.011545788022743975, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.5825806482497653, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.010090398659589862, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.3449460599994166, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.6455116881575989, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.21995781876948328, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.07151490190404086, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='/RUL Group/PINN4SOH/results of reviewer/XJTU results/', help='A folder to save the results')

    args = parser.parse_args()
    return args

    
def get_DONG_args(): # optuna2 set
    parser = argparse.ArgumentParser('Hyper Parameters for training DONG')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F') 
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=128, help='The hidden dim of F') # 256->in best trial, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=80, help='The hidden dim of F') # 128->in best trial, 80
    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')
    
    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=80, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.006268229580119017, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.006456520169381107, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.006158883939912325, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.005237775841792184, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.006252096490545448, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.001221299541334721, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.00011720460436102193, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0007215089581483758, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.5973286128658127e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.1099224334559236e-05, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=1.447250132592353e-06, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.747286485446226e-05, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00881767573907948, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004916804735455082, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=2.181510135329394e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=4.8990846543190585e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.11809194837918663, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.015956048434866418, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.8843241457826115, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02840429623105723, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.5999950741786956, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.01804601857570287, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.2795906470184894, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.1308092300047794, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='/RUL Group/PINN4SOH/results of reviewer/XJTU results/', help='A folder to save the results')

    args = parser.parse_args()
    return args
    

def get_FormerPINN_args(): # optuna2 set
    parser = argparse.ArgumentParser('Hyper Parameters for training FormerPINN')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default= 256, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=512, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='The layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='The hidden dim of F')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 200
    parser.add_argument('--early_stop', type=int, default=80, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.01, help='base lr')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
    parser.add_argument('--lr_F', type=float, default=0.001, help='lr of F')


    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.7, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.2, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.05, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.5, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results of reviewer/XJTU results', help='A folder to save the results')

    args = parser.parse_args()
    return args

    
def get_DeepOPINN_args(): # optuna2 set
    parser = argparse.ArgumentParser('Hyper Parameters for training DeepOPINN')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F') 
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=128, help='The hidden dim of F') # 256->in best trial, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=80, help='The hidden dim of F') # 128->in best trial, 80
    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')
    
    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=80, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.006268229580119017, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.006456520169381107, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.006158883939912325, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.005237775841792184, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.006252096490545448, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.001221299541334721, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.00011720460436102193, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0007215089581483758, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.5973286128658127e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.1099224334559236e-05, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=1.447250132592353e-06, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.747286485446226e-05, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00881767573907948, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004916804735455082, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=2.181510135329394e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=4.8990846543190585e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.11809194837918663, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.015956048434866418, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.8843241457826115, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02840429623105723, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.5999950741786956, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.01804601857570287, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.2795906470184894, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.1308092300047794, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='/RUL Group/PINN4SOH/results of reviewer/XJTU results/', help='A folder to save the results')

    args = parser.parse_args()
    return args


def get_DOPFormer_args(): # optuna2 set
    parser = argparse.ArgumentParser('Hyper Parameters for training DOPFormer')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F')
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=256, help='The hidden dim of F') # 256, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')
    parser.add_argument('--d_out', type=int, default=1, help='the dimension of the model output')
    parser.add_argument('--d_model_XJTU', type=int, default=32, help='dimension of model')
    parser.add_argument('--d_model_TJU', type=int, default=64, help='dimension of model')
    parser.add_argument('--d_model_MIT', type=int, default=32, help='dimension of model')
    parser.add_argument('--d_model_HUST', type=int, default=128, help='dimension of model')
    parser.add_argument('--d_hidden_XJTU', type=int, default=32, help='dimension of model')
    parser.add_argument('--d_hidden_TJU', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_hidden_MIT', type=int, default=64, help='dimension of model')
    parser.add_argument('--d_hidden_HUST', type=int, default=256, help='dimension of model')
    parser.add_argument('--N_XJTU', type=int, default=3, help='the number of encoder and decoder blocks')
    parser.add_argument('--N_TJU', type=int, default=5, help='the number of encoder and decoder blocks')
    parser.add_argument('--N_MIT', type=int, default=1, help='the number of encoder and decoder blocks')
    parser.add_argument('--N_HUST', type=int, default=3, help='the number of encoder and decoder blocks')
    parser.add_argument('--heads_XJTU', type=int, default=2, help='the number of attention heads')
    parser.add_argument('--heads_TJU', type=int, default=4, help='the number of attention heads')
    parser.add_argument('--heads_MIT', type=int, default=2, help='the number of attention heads')
    parser.add_argument('--heads_HUST', type=int, default=2, help='the number of attention heads')
    parser.add_argument('--transformer_dropout_XJTU', type=float, default=0.1784448096682799, help='transformer_dropout')
    parser.add_argument('--transformer_dropout_TJU', type=float, default=0.2389404128864696, help='transformer_dropout')
    parser.add_argument('--transformer_dropout_MIT', type=float, default=0.15129809797862956, help='transformer_dropout')
    parser.add_argument('--transformer_dropout_HUST', type=float, default=0.2867026986359125, help='transformer_dropout')
    
    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=80, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.002934555763005439, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.0006182130774160487, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.00803896011172995, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.0010338093783875922, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.00015094796720626922, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.0007409057899817629, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.0014405097378976264, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0006627661895345457, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.1692515949240485e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.055699105475732e-06, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=9.764939561657171e-05, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.1379872064564874e-06, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00016323908889202802, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004052220886310363, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=1.233238529948589e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=1.662064130220059e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.5811991915960922, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.011545788022743975, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.5825806482497653, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.010090398659589862, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.3449460599994166, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.6455116881575989, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.21995781876948328, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.07151490190404086, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='/RUL Group/PINN4SOH/results of reviewer/XJTU results/', help='A folder to save the results')

    args = parser.parse_args()
    return args

    
def get_LAX_args():
    # Pre-parse only --data
    tmp = argparse.ArgumentParser(add_help=False)
    tmp.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    args, rest = tmp.parse_known_args()

    parser = argparse.ArgumentParser('Hyper Parameters for datasets')
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')

    ## TJU 
    if args.data == 'TJU':
        parser.add_argument('--train_batch', type=int, default=-1, choices=[-1,0,1,2],
                            help='(if -1, read all data and random split train and test sets;'
                                 'else, read the corresponding batch data)')
        parser.add_argument('--test_batch', type=int, default=-1, choices=[-1,0,1,2],
                            help='(if -1, read all data and random split train and test sets;'
                                 'else, read the corresponding batch data)')
        parser.add_argument('--batch',type=int,default=0,
                            choices=[0,1,2], help='TJU batch name')
    ## XJTU
    elif args.data == 'XJTU':
        parser.add_argument('--train_batch', type=int, default=0, choices=[-1,0,1,2,3,4,5],
                            help='(if -1, read all data and random split train and test sets;'
                                 'else, read the corresponding batch data)')
        parser.add_argument('--test_batch', type=int, default=1, choices=[-1,0,1,2,3,4,5],
                            help='(if -1, read all data and random split train and test sets;'
                                 'else, read the corresponding batch data)')
        parser.add_argument('--batch', type=str, default='2C',
                    choices=['2C','3C','R2.5','R3','RW','satellite'],
                    help='XJTU batch name')
    

    parser.add_argument('--batch_size', type=int, default=526, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    # loss related
    parser.add_argument('--betha_LAX_XJTU', type=float, default=.0256, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_TJU', type=float, default=0.7447313058846572, help='loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--betha_LAX_MIT', type=float, default=0.5927778785956674, help='loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--betha_LAX_HUST', type=float, default=0.846945564070922, help='loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')

    parser.add_argument('--beta_LAX_XJTU', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_TJU', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_MIT', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_HUST', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')

    parser.add_argument('--theta_LAX_XJTU', type=float, default=0.0, help='weight for mape loss')
    parser.add_argument('--theta_LAX_TJU', type=float, default=0.9290600824788182, help='weight for mape loss')
    parser.add_argument('--theta_LAX_MIT', type=float, default=0.4808822091755273, help='weight for mape loss')
    parser.add_argument('--theta_LAX_HUST', type=float, default=0.809772111877841, help='weight for mape loss')
    parser.add_argument('--F_hidden_dim_LAX', type=int, default=25, help='the hidden dim of F')

    parser.add_argument('--distance_block_LAX_XJTU', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_TJU', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_MIT', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_HUST', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    
    parser.add_argument('--center_block_LAX_XJTU', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_TJU', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_MIT', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_HUST', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')

    parser.add_argument('--time_block_LAX_XJTU', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_TJU', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_MIT', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_HUST', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    
    parser.add_argument('--s_LAX', type=str, default='MLP', choices=['ordinary_sum', 'MLP', 'Transformer'], 
                        help= 'ordinary Sum or an MLP block to train it.')

    # LAX architecture 
    
    parser.add_argument('--inside_h_star_layers_LAX_XJTU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_XJTU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_XJTU', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_XJTU', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_XJTU', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_XJTU', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_XJTU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_XJTU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_XJTU', type= int, default=39, help= 'g input dimension') # 16, 39:NASA
    parser.add_argument('--g_out_LAX_XJTU', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--H_out_LAX_XJTU', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_XJTU', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_XJTU', type = int, default = 1, help = 'dimension of whole LAX output')

    parser.add_argument('--inside_h_star_layers_LAX_TJU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_TJU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_TJU', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_TJU', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_TJU', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_TJU', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_TJU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_TJU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_TJU', type= int, default=39, help= 'g input dimension') # 16, 39:NASA
    parser.add_argument('--g_out_LAX_TJU', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--H_out_LAX_TJU', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_TJU', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_TJU', type = int, default = 1, help = 'dimension of whole LAX output')

    parser.add_argument('--inside_h_star_layers_LAX_MIT', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_MIT', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_MIT', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_MIT', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_MIT', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_MIT', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_MIT', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_MIT', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_MIT', type= int, default=39, help= 'g input dimension') # 16, 39:NASA
    parser.add_argument('--g_out_LAX_MIT', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--H_out_LAX_MIT', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_MIT', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_MIT', type = int, default = 1, help = 'dimension of whole LAX output')

    
    parser.add_argument('--inside_h_star_layers_LAX_HUST', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_HUST', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_HUST', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_HUST', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_HUST', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_HUST', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_HUST', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_HUST', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_HUST', type= int, default=39, help= 'g input dimension') # 16, 39:NASA
    parser.add_argument('--g_out_LAX_HUST', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--H_out_LAX_HUST', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_HUST', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_HUST', type = int, default = 1, help = 'dimension of whole LAX output')

    parser.add_argument('--zeta_LAX', type=float, default=1.0, help= 'wieght for mse loss')
    parser.add_argument('--kata_LAX', type=float, default=0.0, help = 'wieght for mae loss')
    parser.add_argument('--dynamical_F_LAX', type=bool, default=True, help='this boolean specifies existence of Dynamical_F in our network.')
    parser.add_argument('--dual_LAX_XJTU', type=float, default=.0, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_TJU', type=float, default=0.008827241117241124, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_MIT', type=float, default=0.0003537265723182956, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_HUST', type=float, default=0.570865721084567, help='weight for dual_H loss')

    parser.add_argument('--log_dir', type=str, default='text log.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results of reviewer/results', help='save folder')
    
    parser.add_argument('--lr_net_LAX_XJTU', type=float, default=0.04, help='net learning rate')
    parser.add_argument('--lr_net_LAX_TJU', type=float, default=0.024314095112883502, help='net learning rate')
    parser.add_argument('--lr_net_LAX_MIT', type=float, default=0.011935902068420634, help='net learning rate')
    parser.add_argument('--lr_net_LAX_HUST', type=float, default=0.014526149878088328, help='net learning rate')
    parser.add_argument('--lr_y', type=float, default=0.08, help='y learning rate')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs in the combination mode')
    parser.add_argument('--epoch_net', type=int, default=350, help='number of net epoch')
    parser.add_argument('--epoch_y_LAX', type=int, default=600, help='numbemr of y epoch')
    parser.add_argument('--epoch_th_LAX', type=int, default=3000, help='number of epochs without y optimization')

    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default= 1e-6, help='Early stopping min_delta')

    parser.add_argument('--results_path', type=str, default="./result_fixed_dataloader_ytest", help='result path folder')

    # lr scheduler arguments
    parser.add_argument('--warmup_epochs_net', type=int, default=100, help='Number of warmup epochs of net')
    parser.add_argument('--warmup_epochs_y', type= int, default=100,  help='Number of warmup epochs of y')
    parser.add_argument('--restart_period', type=int, default=150, help='Epochs between LR restarts')
    parser.add_argument('--min_lr_net', type=float, default=1e-5, help='Minimum learning rate for net')
    parser.add_argument('--min_lr_y', type=float, default=1e-4, help='Minimum learning rate for y')
    parser.add_argument('--in_same_batch', type=bool, default=True, help='ether train and test sets are in the same batch')
    
    # number of experiments per batchs 
    parser.add_argument('--num_experiments_per_batch', type = int, default= 10, help= 'Number of experiments per batch of battery')
    parser.add_argument('--num_experiments_optuna', type = int, default= 200, help= 'Number of trials per batch of optuna')
    
    # threshold for the epoch which we want to show our plots of losses
    parser.add_argument('--plot_threshold', type= int, default= 50, help = 'A threshold which after that we show the losses plots during training model')
    parser.add_argument('--plot_update_period', type = int, default= 40, help = 'A threshold which after that, the losses plots will be updated during training')
    parser.add_argument('--h_dim_LAX_XJTU', type=int, default=49, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_TJU', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_MIT', type=int, default=30, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_HUST', type=int, default=58, help='number of inputs of H module.')

    #  ----- Added by Amir -----
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00881767573907948, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004916804735455082, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=2.181510135329394e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=4.8990846543190585e-05, help='lr of F')
    parser.add_argument('--schedule_beta', type=int, default=0, help='schedule_beta used by the PDE network')
    #  -------------------------
    
    parser.add_argument('--run_for_LAX', type=bool, default=False, help='run_for_LAX')
    parser.add_argument('--flagsoori', type=int, default=526, help='??')
    parser.add_argument('--trained_by_y_opt_LAX', default= False, type=bool, help='Whether the loaded model was trained with y_opt in LAX')
    args = parser.parse_args(rest)
    return args   


def get_DOPDeepOLAX_args(): 
    parser = argparse.ArgumentParser('Hyper Parameters for training DOPLAX using a bagging net')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F') 
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=128, help='The hidden dim of F') # 256->in best trial, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=80, help='The hidden dim of F') # 128->in best trial, 80
    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')

    # Combination related
    parser.add_argument('--bagging_NN_lr_XJTU', type=float, default=0.02, help='bagging_NN_lr')
    parser.add_argument('--bagging_NN_lr_TJU', type=float, default=0.006456520169381107, help='bagging_NN_lr')
    parser.add_argument('--bagging_NN_lr_MIT', type=float, default=0.006663648646643286, help='bagging_NN_lr')
    parser.add_argument('--bagging_NN_lr_HUST', type=float, default=0.00022334184499247844, help='bagging_NN_lr')
    parser.add_argument('--bag_hidden_dim_XJTU', type=list, default=[50, 50], help='The hidden dim of Bagging Neural Network')
    parser.add_argument('--bag_hidden_dim_TJU', type=list, default=[50, 50], help='The hidden dim of Bagging Neural Network')
    parser.add_argument('--bag_hidden_dim_MIT', type=list, default=[50, 50], help='The hidden dim of Bagging Neural Network')
    parser.add_argument('--bag_hidden_dim_HUST', type=list, default=[100, 100], help='The hidden dim of Bagging Neural Network')
    parser.add_argument('--mono_bag_XJTU', type=float, default=0.4, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--mono_bag_TJU', type=float, default=0.01804601857570287, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--mono_bag_MIT', type=float, default=0.014599210344919067, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--mono_bag_HUST', type=float, default=0.9933428738041119, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')

    # Lax Model related
    parser.add_argument('--lr_net_LAX_XJTU', type=float, default=0.04, help='net learning rate')
    parser.add_argument('--lr_net_LAX_TJU', type=float, default=0.024314095112883502, help='net learning rate')
    parser.add_argument('--lr_net_LAX_MIT', type=float, default=0.011935902068420634, help='net learning rate')
    parser.add_argument('--lr_net_LAX_HUST', type=float, default=0.014526149878088328, help='net learning rate')
    parser.add_argument('--lr_y_LAX', type=float, default=0.08, help='y learning rate')

    parser.add_argument('--epoch_y_LAX', type=int, default=600, help='numbemr of y epoch')
    parser.add_argument('--epoch_th_LAX', type=int, default=3000, help='number of epochs without y optimization')

    parser.add_argument('--warmup_epochs_net_LAX', type=int, default=100, help='Number of warmup epochs of net')
    parser.add_argument('--warmup_epochs_y_LAX', type=int, default=100, help='Number of warmup epochs of y')
    parser.add_argument('--restart_period_LAX', type=int, default=100, help='Epochs between LR restarts')
    parser.add_argument('--min_lr_net_LAX', type=float, default=1e-5, help='Minimum learning rate for net')
    parser.add_argument('--min_lr_y_LAX', type=float, default=1e-4, help='Minimum learning rate for y')
    parser.add_argument('--F_hidden_dim_LAX', type=int, default=25, help='the hidden dim of F')

    parser.add_argument('--theta_LAX_XJTU', type=float, default=0.0, help='weight for mape loss')
    parser.add_argument('--theta_LAX_TJU', type=float, default=0.9290600824788182, help='weight for mape loss')
    parser.add_argument('--theta_LAX_MIT', type=float, default=0.4808822091755273, help='weight for mape loss')
    parser.add_argument('--theta_LAX_HUST', type=float, default=0.809772111877841, help='weight for mape loss')
    parser.add_argument('--zeta_LAX', type=float, default=1.0, help='wieght for mse loss')
    parser.add_argument('--kata_LAX', type=float, default=0.0, help='wieght for mae loss')
    parser.add_argument('--betha_LAX_XJTU', type=float, default=.0256, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_TJU', type=float, default=0.7447313058846572, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_MIT', type=float, default=0.5927778785956674, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_HUST', type=float, default=0.846945564070922, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--dual_LAX_XJTU', type=float, default=.0, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_TJU', type=float, default=0.008827241117241124, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_MIT', type=float, default=0.0003537265723182956, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_HUST', type=float, default=0.570865721084567, help='weight for dual_H loss')
    parser.add_argument('--h_dim_LAX_XJTU', type=int, default=49, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_TJU', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_MIT', type=int, default=30, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_HUST', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--run_for_LAX', type=bool, help='run_for_LAX')
    parser.add_argument('--trained_by_y_opt_LAX', default= False, type=bool, help='Whether the loaded model was trained with y_opt in LAX')
    parser.add_argument('--flagsoori', type=bool, help='??')
    parser.add_argument('--beta_LAX_XJTU', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_TJU', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_MIT', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_HUST', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--dynamical_F_LAX', type=bool, default=True, help='this boolean specifies existence of Dynamical_F in our network.')

    parser.add_argument('--distance_block_LAX_XJTU', type= str, default="SumProductNetwork",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_TJU', type= str, default="SumProductNetwork",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_MIT', type= str, default="SumProductNetwork",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_HUST', type= str, default="SumProductNetwork",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    
    parser.add_argument('--center_block_LAX_XJTU', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_TJU', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_MIT', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_HUST', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    
    parser.add_argument('--time_block_LAX_XJTU', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_TJU', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_MIT', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_HUST', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    
    parser.add_argument('--s_LAX', type=str, default='MLP', choices=['ordinary_sum', 'MLP', 'Transformer'], 
                        help= 'ordinary Sum or an MLP block to train it.')
    # LAX architecture 
    
    parser.add_argument('--inside_h_star_layers_LAX_XJTU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_XJTU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_XJTU', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_XJTU', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_XJTU', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_XJTU', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_XJTU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_XJTU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_XJTU', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_LAX_XJTU', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--h_out_LAX_XJTU', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_XJTU', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_XJTU', type = int, default = 1, help = 'dimension of whole LAX output')

    parser.add_argument('--inside_h_star_layers_LAX_TJU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_TJU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_TJU', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_TJU', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_TJU', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_TJU', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_TJU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_TJU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_TJU', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_LAX_TJU', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--h_out_LAX_TJU', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_TJU', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_TJU', type = int, default = 1, help = 'dimension of whole LAX output')
                        
    parser.add_argument('--inside_h_star_layers_LAX_MIT', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_MIT', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_MIT', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_MIT', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_MIT', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_MIT', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_MIT', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_MIT', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_MIT', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_LAX_MIT', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--h_out_LAX_MIT', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_MIT', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_MIT', type = int, default = 1, help = 'dimension of whole LAX output')
                        
    parser.add_argument('--inside_h_star_layers_LAX_HUST', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_HUST', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_HUST', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_HUST', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_HUST', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_HUST', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_HUST', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_HUST', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_HUST', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_LAX_HUST', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--h_out_LAX_HUST', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_HUST', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_HUST', type = int, default = 1, help = 'dimension of whole LAX output')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=80, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.006268229580119017, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.006456520169381107, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.006158883939912325, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.005237775841792184, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.006252096490545448, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.001221299541334721, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.00011720460436102193, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0007215089581483758, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.5973286128658127e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.1099224334559236e-05, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=1.447250132592353e-06, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.747286485446226e-05, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00881767573907948, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004916804735455082, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=2.181510135329394e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=4.8990846543190585e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.11809194837918663, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.015956048434866418, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.8843241457826115, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02840429623105723, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.5999950741786956, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.01804601857570287, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.2795906470184894, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.1308092300047794, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='/RUL Group/DOPLAX/results of reviewer/XJTU results/', help='A folder to save the results')

    args = parser.parse_args()
    return args

    
def get_Bagging_u_args(): 
    parser = argparse.ArgumentParser('Hyper Parameters for training DOPLAX using a bagging net')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F') 
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=128, help='The hidden dim of F') # 256->in best trial, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=80, help='The hidden dim of F') # 128->in best trial, 80
    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')

    # Combination related
    parser.add_argument('--bagging_NN_lr_XJTU', type=float, default=0.02, help='bagging_NN_lr')
    parser.add_argument('--bagging_NN_lr_TJU', type=float, default=0.006456520169381107, help='bagging_NN_lr')
    parser.add_argument('--bagging_NN_lr_MIT', type=float, default=0.006663648646643286, help='bagging_NN_lr')
    parser.add_argument('--bagging_NN_lr_HUST', type=float, default=0.00022334184499247844, help='bagging_NN_lr')
    parser.add_argument('--bag_hidden_dim_XJTU', type=list, default=[50, 50], help='The hidden dim of Bagging Neural Network')
    parser.add_argument('--bag_hidden_dim_TJU', type=list, default=[50, 50], help='The hidden dim of Bagging Neural Network')
    parser.add_argument('--bag_hidden_dim_MIT', type=list, default=[50, 50], help='The hidden dim of Bagging Neural Network')
    parser.add_argument('--bag_hidden_dim_HUST', type=list, default=[100, 100], help='The hidden dim of Bagging Neural Network')
    parser.add_argument('--mono_bag_XJTU', type=float, default=0.4, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--mono_bag_TJU', type=float, default=0.01804601857570287, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--mono_bag_MIT', type=float, default=0.014599210344919067, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--mono_bag_HUST', type=float, default=0.9933428738041119, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')

    # Lax Model related
    parser.add_argument('--lr_net_LAX_XJTU', type=float, default=0.04, help='net learning rate')
    parser.add_argument('--lr_net_LAX_TJU', type=float, default=0.024314095112883502, help='net learning rate')
    parser.add_argument('--lr_net_LAX_MIT', type=float, default=0.011935902068420634, help='net learning rate')
    parser.add_argument('--lr_net_LAX_HUST', type=float, default=0.014526149878088328, help='net learning rate')
    parser.add_argument('--lr_y_LAX', type=float, default=0.08, help='y learning rate')

    parser.add_argument('--epoch_y_LAX', type=int, default=600, help='numbemr of y epoch')
    parser.add_argument('--epoch_th_LAX', type=int, default=3000, help='number of epochs without y optimization')

    parser.add_argument('--warmup_epochs_net_LAX', type=int, default=100, help='Number of warmup epochs of net')
    parser.add_argument('--warmup_epochs_y_LAX', type=int, default=100, help='Number of warmup epochs of y')
    parser.add_argument('--restart_period_LAX', type=int, default=100, help='Epochs between LR restarts')
    parser.add_argument('--min_lr_net_LAX', type=float, default=1e-5, help='Minimum learning rate for net')
    parser.add_argument('--min_lr_y_LAX', type=float, default=1e-4, help='Minimum learning rate for y')
    parser.add_argument('--F_hidden_dim_LAX', type=int, default=25, help='the hidden dim of F')

    parser.add_argument('--theta_LAX_XJTU', type=float, default=0.0, help='weight for mape loss')
    parser.add_argument('--theta_LAX_TJU', type=float, default=0.9290600824788182, help='weight for mape loss')
    parser.add_argument('--theta_LAX_MIT', type=float, default=0.4808822091755273, help='weight for mape loss')
    parser.add_argument('--theta_LAX_HUST', type=float, default=0.809772111877841, help='weight for mape loss')
    parser.add_argument('--zeta_LAX', type=float, default=1.0, help='wieght for mse loss')
    parser.add_argument('--kata_LAX', type=float, default=0.0, help='wieght for mae loss')
    parser.add_argument('--betha_LAX_XJTU', type=float, default=.0256, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_TJU', type=float, default=0.7447313058846572, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_MIT', type=float, default=0.5927778785956674, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_HUST', type=float, default=0.846945564070922, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--dual_LAX_XJTU', type=float, default=.0, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_TJU', type=float, default=0.008827241117241124, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_MIT', type=float, default=0.0003537265723182956, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_HUST', type=float, default=0.570865721084567, help='weight for dual_H loss')
    parser.add_argument('--h_dim_LAX_XJTU', type=int, default=49, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_TJU', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_MIT', type=int, default=58, help='number of inputs of H module.') # 58
    parser.add_argument('--h_dim_LAX_HUST', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--run_for_LAX', type=bool, help='run_for_LAX')
    parser.add_argument('--trained_by_y_opt_LAX', default= False, type=bool, help='Whether the loaded model was trained with y_opt in LAX')
    parser.add_argument('--flagsoori', type=bool, help='??')
    parser.add_argument('--beta_LAX_XJTU', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_TJU', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_MIT', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_HUST', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--dynamical_F_LAX', type=bool, default=True, help='this boolean specifies existence of Dynamical_F in our network.')

    parser.add_argument('--distance_block_LAX_XJTU', type= str, default="SumProductNetwork",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_TJU', type= str, default="SumProductNetwork",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_MIT', type= str, default="SumProductNetwork",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_HUST', type= str, default="SumProductNetwork",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    
    parser.add_argument('--center_block_LAX_XJTU', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_TJU', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_MIT', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_HUST', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    
    parser.add_argument('--time_block_LAX_XJTU', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_TJU', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_MIT', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_HUST', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    
    parser.add_argument('--s_LAX', type=str, default='MLP', choices=['ordinary_sum', 'MLP', 'Transformer'], 
                        help= 'ordinary Sum or an MLP block to train it.')
    # LAX architecture 
    
    parser.add_argument('--inside_h_star_layers_LAX_XJTU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_XJTU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_XJTU', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_XJTU', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_XJTU', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_XJTU', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_XJTU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_XJTU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_XJTU', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_LAX_XJTU', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--H_out_LAX_XJTU', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_XJTU', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_XJTU', type = int, default = 1, help = 'dimension of whole LAX output')

    parser.add_argument('--inside_h_star_layers_LAX_TJU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_TJU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_TJU', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_TJU', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_TJU', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_TJU', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_TJU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_TJU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_TJU', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_LAX_TJU', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--H_out_LAX_TJU', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_TJU', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_TJU', type = int, default = 1, help = 'dimension of whole LAX output')
                        
    parser.add_argument('--inside_h_star_layers_LAX_MIT', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_MIT', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_MIT', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_MIT', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_MIT', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_MIT', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_MIT', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_MIT', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_MIT', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_LAX_MIT', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--H_out_LAX_MIT', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_MIT', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_MIT', type = int, default = 1, help = 'dimension of whole LAX output')
                        
    parser.add_argument('--inside_h_star_layers_LAX_HUST', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_HUST', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_HUST', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_HUST', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_HUST', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_HUST', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_HUST', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_HUST', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_HUST', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_LAX_HUST', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--H_out_LAX_HUST', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_HUST', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_HUST', type = int, default = 1, help = 'dimension of whole LAX output')
    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=80, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.006268229580119017, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.006456520169381107, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.006158883939912325, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.005237775841792184, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.006252096490545448, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.001221299541334721, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.00011720460436102193, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0007215089581483758, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.5973286128658127e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.1099224334559236e-05, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=1.447250132592353e-06, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.747286485446226e-05, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00881767573907948, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004916804735455082, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=2.181510135329394e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=4.8990846543190585e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.11809194837918663, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.015956048434866418, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.8843241457826115, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02840429623105723, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.5999950741786956, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.01804601857570287, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.2795906470184894, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.1308092300047794, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='/RUL Group/DOPLAX/results of reviewer/XJTU results/', help='A folder to save the results')

    args = parser.parse_args()
    return args


def get_LAX_predictor_args(): 
    parser = argparse.ArgumentParser('Hyper Parameters for training DOPLAX using LAX as a predictor')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F') 
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=128, help='The hidden dim of F') # 256->in best trial, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=80, help='The hidden dim of F') # 128->in best trial, 80
    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')

    # Lax Model related
    parser.add_argument('--lr_net_LAX_XJTU', type=float, default=0.04, help='net learning rate')
    parser.add_argument('--lr_net_LAX_TJU', type=float, default=0.024314095112883502, help='net learning rate')
    parser.add_argument('--lr_net_LAX_MIT', type=float, default=0.011935902068420634, help='net learning rate')
    parser.add_argument('--lr_net_LAX_HUST', type=float, default=0.014526149878088328, help='net learning rate')
    parser.add_argument('--lr_y_LAX', type=float, default=0.08, help='y learning rate')
    parser.add_argument('--dynamical_F_LAX', type=bool, default=True, help='this boolean specifies existence of Dynamical_F in our network.')
    parser.add_argument('--F_hidden_dim_LAX', type=int, default=25, help='the hidden dim of F')

    parser.add_argument('--epoch_y_LAX', type=int, default=600, help='numbemr of y epoch')
    parser.add_argument('--epoch_th_LAX', type=int, default=3000, help='number of epochs without y optimization')

    parser.add_argument('--warmup_epochs_net_LAX', type=int, default=100, help='Number of warmup epochs of net')
    parser.add_argument('--warmup_epochs_y_LAX', type=int, default=100, help='Number of warmup epochs of y')
    parser.add_argument('--restart_period_LAX', type=int, default=100, help='Epochs between LR restarts')
    parser.add_argument('--min_lr_net_LAX', type=float, default=1e-5, help='Minimum learning rate for net')
    parser.add_argument('--min_lr_y_LAX', type=float, default=1e-4, help='Minimum learning rate for y')

    parser.add_argument('--theta_LAX_XJTU', type=float, default=0.0, help='weight for mape loss')
    parser.add_argument('--theta_LAX_TJU', type=float, default=0.9290600824788182, help='weight for mape loss')
    parser.add_argument('--theta_LAX_MIT', type=float, default=0.4808822091755273, help='weight for mape loss')
    parser.add_argument('--theta_LAX_HUST', type=float, default=0.809772111877841, help='weight for mape loss')
    parser.add_argument('--zeta_LAX', type=float, default=1.0, help='wieght for mse loss')
    parser.add_argument('--kata_LAX', type=float, default=0.0, help='wieght for mae loss')
    parser.add_argument('--betha_LAX_XJTU', type=float, default=.0256, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_TJU', type=float, default=0.7447313058846572, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_MIT', type=float, default=0.5927778785956674, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_HUST', type=float, default=0.846945564070922, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--dual_LAX_XJTU', type=float, default=.0, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_TJU', type=float, default=0.008827241117241124, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_MIT', type=float, default=0.0003537265723182956, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_HUST', type=float, default=0.570865721084567, help='weight for dual_H loss')
    parser.add_argument('--h_dim_LAX_XJTU', type=int, default=49, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_TJU', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_MIT', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_HUST', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--run_for_LAX', type=bool, help='run_for_LAX')
    parser.add_argument('--trained_by_y_opt_LAX', default= False, type=bool, help='Whether the loaded model was trained with y_opt in LAX')
    parser.add_argument('--flagsoori', type=bool, help='??')
    parser.add_argument('--beta_LAX_XJTU', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_TJU', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_MIT', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_HUST', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')

    parser.add_argument('--distance_block_LAX_XJTU', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_TJU', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_MIT', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_HUST', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    
    parser.add_argument('--center_block_LAX_XJTU', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_TJU', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_MIT', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_HUST', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    
    parser.add_argument('--time_block_LAX_XJTU', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_TJU', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_MIT', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_HUST', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    
    parser.add_argument('--s_LAX', type=str, default='MLP', choices=['ordinary_sum', 'MLP', 'Transformer'], 
                        help= 'ordinary Sum or an MLP block to train it.')
    # LAX architecture 
    
    parser.add_argument('--inside_h_star_layers_XJTU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_XJTU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_XJTU', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_XJTU', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_XJTU', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_XJTU', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_XJTU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_XJTU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_XJTU', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_XJTU', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--h_out_XJTU', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_XJTU', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_XJTU', type = int, default = 1, help = 'dimension of whole LAX output')
                        
    parser.add_argument('--inside_h_star_layers_TJU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_TJU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_TJU', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_TJU', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_TJU', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_TJU', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_TJU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_TJU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_TJU', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_TJU', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--h_out_TJU', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_TJU', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_TJU', type = int, default = 1, help = 'dimension of whole LAX output')
                        
    parser.add_argument('--inside_h_star_layers_MIT', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_MIT', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_MIT', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_MIT', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_MIT', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_MIT', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_MIT', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_MIT', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_MIT', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_MIT', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--h_out_MIT', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_MIT', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_MIT', type = int, default = 1, help = 'dimension of whole LAX output')
                        
    parser.add_argument('--inside_h_star_layers_HUST', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_HUST', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_HUST', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_HUST', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_HUST', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_HUST', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_HUST', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_HUST', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_HUST', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_HUST', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--h_out_HUST', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_HUST', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_HUST', type = int, default = 1, help = 'dimension of whole LAX output')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=80, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.006268229580119017, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.006456520169381107, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.006158883939912325, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.005237775841792184, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.006252096490545448, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.001221299541334721, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.00011720460436102193, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0007215089581483758, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.5973286128658127e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.1099224334559236e-05, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=1.447250132592353e-06, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.747286485446226e-05, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00881767573907948, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004916804735455082, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=2.181510135329394e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=4.8990846543190585e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.11809194837918663, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.015956048434866418, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.8843241457826115, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02840429623105723, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.5999950741786956, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.01804601857570287, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.2795906470184894, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.1308092300047794, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='/RUL Group/DOPLAX/results of reviewer/XJTU results/', help='A folder to save the results')

    args = parser.parse_args()
    return args


def get_DeepOLAX_args(): 
    parser = argparse.ArgumentParser('Hyper Parameters for training DOPLAX using LAX as a predictor')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F') 
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=128, help='The hidden dim of F') # 256->in best trial, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=80, help='The hidden dim of F') # 128->in best trial, 80
    parser.add_argument('--F_hidden_dim_LAX', type=int, default=25, help='the hidden dim of F')
    parser.add_argument('--dynamical_F_LAX', type=bool, default=True, help='this boolean specifies existence of Dynamical_F in our network.')

    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')

    # Lax Model related
    parser.add_argument('--lr_net_LAX_XJTU', type=float, default=0.04, help='net learning rate')
    parser.add_argument('--lr_net_LAX_TJU', type=float, default=0.024314095112883502, help='net learning rate')
    parser.add_argument('--lr_net_LAX_MIT', type=float, default=0.011935902068420634, help='net learning rate')
    parser.add_argument('--lr_net_LAX_HUST', type=float, default=0.014526149878088328, help='net learning rate')
    parser.add_argument('--lr_y_LAX', type=float, default=0.08, help='y learning rate')

    parser.add_argument('--epoch_y_LAX', type=int, default=600, help='numbemr of y epoch')
    parser.add_argument('--epoch_th_LAX', type=int, default=3000, help='number of epochs without y optimization')

    parser.add_argument('--warmup_epochs_net_LAX', type=int, default=100, help='Number of warmup epochs of net')
    parser.add_argument('--warmup_epochs_y_LAX', type=int, default=100, help='Number of warmup epochs of y')
    parser.add_argument('--restart_period_LAX', type=int, default=100, help='Epochs between LR restarts')
    parser.add_argument('--min_lr_net_LAX', type=float, default=1e-5, help='Minimum learning rate for net')
    parser.add_argument('--min_lr_y_LAX', type=float, default=1e-4, help='Minimum learning rate for y')

    parser.add_argument('--theta_LAX_XJTU', type=float, default=0.0, help='weight for mape loss')
    parser.add_argument('--theta_LAX_TJU', type=float, default=0.9290600824788182, help='weight for mape loss')
    parser.add_argument('--theta_LAX_MIT', type=float, default=0.4808822091755273, help='weight for mape loss')
    parser.add_argument('--theta_LAX_HUST', type=float, default=0.809772111877841, help='weight for mape loss')
    parser.add_argument('--zeta_LAX', type=float, default=1.0, help='wieght for mse loss')
    parser.add_argument('--kata_LAX', type=float, default=0.0, help='wieght for mae loss')
    parser.add_argument('--betha_LAX_XJTU', type=float, default=.0256, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_TJU', type=float, default=0.7447313058846572, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_MIT', type=float, default=0.5927778785956674, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_HUST', type=float, default=0.846945564070922, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--dual_LAX_XJTU', type=float, default=.0, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_TJU', type=float, default=0.008827241117241124, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_MIT', type=float, default=0.0003537265723182956, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_HUST', type=float, default=0.570865721084567, help='weight for dual_H loss')
    parser.add_argument('--h_dim_LAX_XJTU', type=int, default=49, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_TJU', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_MIT', type=int, default=30, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_HUST', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--run_for_LAX', type=bool, help='run_for_LAX')
    parser.add_argument('--trained_by_y_opt_LAX', default= False, type=bool, help='Whether the loaded model was trained with y_opt in LAX')
    parser.add_argument('--flagsoori', type=bool, help='??')
    parser.add_argument('--beta_LAX_XJTU', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_TJU', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_MIT', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_HUST', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')

    parser.add_argument('--distance_block_LAX_XJTU', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_TJU', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_MIT', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_HUST', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    
    parser.add_argument('--center_block_LAX_XJTU', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_TJU', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_MIT', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_HUST', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    
    parser.add_argument('--time_block_LAX_XJTU', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_TJU', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_MIT', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_HUST', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    
    parser.add_argument('--s_LAX', type=str, default='MLP', choices=['ordinary_sum', 'MLP', 'Transformer'], 
                        help= 'ordinary Sum or an MLP block to train it.')
    # LAX architecture 
    
    parser.add_argument('--inside_h_star_layers_XJTU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_XJTU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_XJTU', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_XJTU', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_XJTU', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_XJTU', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_XJTU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_XJTU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_XJTU', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_XJTU', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--h_out_XJTU', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_XJTU', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_XJTU', type = int, default = 1, help = 'dimension of whole LAX output')
                        
    parser.add_argument('--inside_h_star_layers_TJU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_TJU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_TJU', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_TJU', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_TJU', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_TJU', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_TJU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_TJU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_TJU', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_TJU', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--h_out_TJU', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_TJU', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_TJU', type = int, default = 1, help = 'dimension of whole LAX output')
                        
    parser.add_argument('--inside_h_star_layers_MIT', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_MIT', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_MIT', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_MIT', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_MIT', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_MIT', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_MIT', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_MIT', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_MIT', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_MIT', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--h_out_MIT', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_MIT', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_MIT', type = int, default = 1, help = 'dimension of whole LAX output')
                        
    parser.add_argument('--inside_h_star_layers_HUST', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_HUST', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_HUST', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_HUST', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_HUST', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_HUST', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_HUST', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_HUST', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_HUST', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_HUST', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--h_out_HUST', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_HUST', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_HUST', type = int, default = 1, help = 'dimension of whole LAX output')
    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=80, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.006268229580119017, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.006456520169381107, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.006158883939912325, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.005237775841792184, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.006252096490545448, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.001221299541334721, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.00011720460436102193, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0007215089581483758, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.5973286128658127e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.1099224334559236e-05, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=1.447250132592353e-06, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.747286485446226e-05, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00881767573907948, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004916804735455082, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=2.181510135329394e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=4.8990846543190585e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.11809194837918663, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.015956048434866418, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.8843241457826115, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02840429623105723, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.5999950741786956, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.01804601857570287, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.2795906470184894, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.1308092300047794, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='logging file name, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='/RUL Group/DOPLAX/results of reviewer/XJTU results/', help='A folder to save the results')

    args = parser.parse_args()
    return args

    
def get_FE_args():
    parser = argparse.ArgumentParser('Hyper Parameters for feature extractors')

    # basic config
    parser.add_argument('--charge_discharge_length', type=int, default=100, help='The resampled length for charge and discharge curves')
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
    parser.add_argument('--save_folder', type=str, default='/RUL Group/PINN4SOH/results of reviewer/', help='A folder to save the results')

    args = parser.parse_args()
    return args


def get_finetuning_PINN_args():
    parser = argparse.ArgumentParser('Hyper Parameters for fine-tuning PINN')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch',type=str,default='2C',choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.01, help='base lr')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
    parser.add_argument('--lr_F', type=float, default=0.01, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')
    
    # model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F') # battery 60, ours 64

    # loss related
    parser.add_argument('--alpha', type=float, default=0.7, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_XJTU', type=float, default=0.7, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.2, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.05, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.5, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # The AdaModel class inherits the PINN class, and the above parameters are all parameters of PINN.
    # The following are the parameters of AdaModel.
    # adaption related
    parser.add_argument('--pretrained_model', type=str, default=None, help='The saving path of the model trained in the source domain')
    parser.add_argument('--adaptation_lr', type=float, default=4e-4, help='adaption lr')
    parser.add_argument('--adaptation_epochs', type=int, default=200, help='adaption epochs')

    parser.add_argument('--target_data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--target_batch', type=int, default=-1, choices=[-1,0,1,2,3,4,5],
                        help='XJTU dataset is divided into 6 batches, and TJU dataset is divided into 3 batches. '
                             'If target_data is XJTU, the value range of target_batch is [-1,0,1,2,3,4,5];'
                             'If target_data is TJU, the value range of target_batch is [-1,0,1,2];'
                             'If it is other datasets, ignore target_batch')
    
    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='AdaModel_test', help='save folder')

    args = parser.parse_args()
    return args


def get_finetuning_DeepOPINN_args(): 
    parser = argparse.ArgumentParser('Hyper Parameters for fine-tuning DeepOPINN')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F') 
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=128, help='The hidden dim of F') # 256->in best trial, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=80, help='The hidden dim of F') # 128->in best trial, 80
    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')
    
    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=40, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.006268229580119017, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.006456520169381107, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.006158883939912325, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.005237775841792184, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.006252096490545448, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.001221299541334721, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.00011720460436102193, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0007215089581483758, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.5973286128658127e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.1099224334559236e-05, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=1.447250132592353e-06, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.747286485446226e-05, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00881767573907948, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004916804735455082, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=2.181510135329394e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=4.8990846543190585e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.11809194837918663, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.015956048434866418, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.8843241457826115, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02840429623105723, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.5999950741786956, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.01804601857570287, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.2795906470184894, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.1308092300047794, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # The AdaModel class inherits the DeepOPINN class, and the above parameters are all parameters of DeepOPINN.
    # The following are the parameters of AdaModel.
    # adaption related
    parser.add_argument('--pretrained_model', type=str, default=None, help='The saving path of the model trained in the source domain')

    parser.add_argument('--target_data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--target_batch', type=int, default=-1, choices=[-1,0,1,2,3,4,5],
                        help='XJTU dataset is divided into 6 batches, and TJU dataset is divided into 3 batches. '
                             'If target_data is XJTU, the value range of target_batch is [-1,0,1,2,3,4,5];'
                             'If target_data is TJU, the value range of target_batch is [-1,0,1,2];'
                             'If it is other datasets, ignore target_batch')
    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='AdaModel_test', help='save folder')
    
    args = parser.parse_args()
    return args


def get_finetuning_DOPFormer_args(): 
    parser = argparse.ArgumentParser('Hyper Parameters for fine-tuning DOPFormer')
   # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F')
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=256, help='The hidden dim of F') # 256, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')
    parser.add_argument('--d_out', type=int, default=1, help='the dimension of the model output')
    parser.add_argument('--d_model_XJTU', type=int, default=32, help='dimension of model')
    parser.add_argument('--d_model_TJU', type=int, default=64, help='dimension of model')
    parser.add_argument('--d_model_MIT', type=int, default=32, help='dimension of model')
    parser.add_argument('--d_model_HUST', type=int, default=128, help='dimension of model')
    parser.add_argument('--d_hidden_XJTU', type=int, default=32, help='dimension of model')
    parser.add_argument('--d_hidden_TJU', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_hidden_MIT', type=int, default=64, help='dimension of model')
    parser.add_argument('--d_hidden_HUST', type=int, default=256, help='dimension of model')
    parser.add_argument('--N_XJTU', type=int, default=3, help='the number of encoder and decoder blocks')
    parser.add_argument('--N_TJU', type=int, default=5, help='the number of encoder and decoder blocks')
    parser.add_argument('--N_MIT', type=int, default=1, help='the number of encoder and decoder blocks')
    parser.add_argument('--N_HUST', type=int, default=3, help='the number of encoder and decoder blocks')
    parser.add_argument('--heads_XJTU', type=int, default=2, help='the number of attention heads')
    parser.add_argument('--heads_TJU', type=int, default=4, help='the number of attention heads')
    parser.add_argument('--heads_MIT', type=int, default=2, help='the number of attention heads')
    parser.add_argument('--heads_HUST', type=int, default=2, help='the number of attention heads')
    parser.add_argument('--transformer_dropout_XJTU', type=float, default=0.1784448096682799, help='transformer_dropout')
    parser.add_argument('--transformer_dropout_TJU', type=float, default=0.2389404128864696, help='transformer_dropout')
    parser.add_argument('--transformer_dropout_MIT', type=float, default=0.15129809797862956, help='transformer_dropout')
    parser.add_argument('--transformer_dropout_HUST', type=float, default=0.2867026986359125, help='transformer_dropout')
    
    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=40, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.002934555763005439, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.0006182130774160487, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.00803896011172995, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.0010338093783875922, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.00015094796720626922, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.0007409057899817629, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.0014405097378976264, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0006627661895345457, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.1692515949240485e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.055699105475732e-06, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=9.764939561657171e-05, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.1379872064564874e-06, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00016323908889202802, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004052220886310363, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=1.233238529948589e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=1.662064130220059e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.5811991915960922, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.011545788022743975, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.5825806482497653, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.010090398659589862, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.3449460599994166, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.6455116881575989, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.21995781876948328, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.07151490190404086, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # The AdaModel class inherits the DOPFormer class, and the above parameters are all parameters of DOPFormer.
    # The following are the parameters of AdaModel.
    # adaption related
    parser.add_argument('--pretrained_model', type=str, default=None, help='The saving path of the model trained in the source domain')

    parser.add_argument('--target_data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--target_batch', type=int, default=-1, choices=[-1,0,1,2,3,4,5],
                        help='XJTU dataset is divided into 6 batches, and TJU dataset is divided into 3 batches. '
                             'If target_data is XJTU, the value range of target_batch is [-1,0,1,2,3,4,5];'
                             'If target_data is TJU, the value range of target_batch is [-1,0,1,2];'
                             'If it is other datasets, ignore target_batch')
    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='AdaModel_test', help='save folder')
    
    args = parser.parse_args()
    return args


def get_finetuning_DeepONet_args(): 
    parser = argparse.ArgumentParser('Hyper Parameters for fine-tuning DeepONet')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F') 
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=128, help='The hidden dim of F') # 256->in best trial, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=80, help='The hidden dim of F') # 128->in best trial, 80
    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')
    
    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=40, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.006268229580119017, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.006456520169381107, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.006158883939912325, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.005237775841792184, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.006252096490545448, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.001221299541334721, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.00011720460436102193, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0007215089581483758, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.5973286128658127e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.1099224334559236e-05, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=1.447250132592353e-06, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.747286485446226e-05, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00881767573907948, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004916804735455082, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=2.181510135329394e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=4.8990846543190585e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.11809194837918663, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.015956048434866418, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.8843241457826115, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02840429623105723, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.5999950741786956, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.01804601857570287, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.2795906470184894, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.1308092300047794, help='loss = l_data + alpha * l_PDE + beta * l_physics')


    # The AdaModel class inherits the DOPFormer class, and the above parameters are all parameters of DOPFormer.
    # The following are the parameters of AdaModel.
    # adaption related
    parser.add_argument('--pretrained_model', type=str, default=None, help='The saving path of the model trained in the source domain')

    parser.add_argument('--target_data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--target_batch', type=int, default=-1, choices=[-1,0,1,2,3,4,5],
                        help='XJTU dataset is divided into 6 batches, and TJU dataset is divided into 3 batches. '
                             'If target_data is XJTU, the value range of target_batch is [-1,0,1,2,3,4,5];'
                             'If target_data is TJU, the value range of target_batch is [-1,0,1,2];'
                             'If it is other datasets, ignore target_batch')
    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='AdaModel_test', help='save folder')
    
    args = parser.parse_args()
    return args


def get_finetuning_Bagging_u_args():
    parser = argparse.ArgumentParser('Hyper Parameters for fine-tuning DOPLAX')
    # Dataloader related
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--batch_size_XJTU', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_TJU', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_MIT', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_HUST', type=int, default=256, help='batch size')
    parser.add_argument('--split_mode', type=str, default='charge', help="It could be ‘charge’, ‘discharge’, or ‘None’")
    parser.add_argument('--dtypes', type=list, default=['current', 'capacity'], help="It could be ‘['current', 'capacity']’, ‘['voltage', 'capacity']’, ‘['voltage', 'current', 'capacity']’")

    # Model related
    parser.add_argument('--F_layers_num_XJTU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_TJU', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_MIT', type=int, default=5, help='The layers num of F')
    parser.add_argument('--F_layers_num_HUST', type=int, default=4, help='The layers num of F') 
    parser.add_argument('--F_hidden_dim_XJTU', type=int, default=80, help='The hidden dim of F') # battery 60, ours 64
    parser.add_argument('--F_hidden_dim_TJU', type=int, default=128, help='The hidden dim of F')
    parser.add_argument('--F_hidden_dim_MIT', type=int, default=128, help='The hidden dim of F') # 256->in best trial, 128
    parser.add_argument('--F_hidden_dim_HUST', type=int, default=80, help='The hidden dim of F') # 128->in best trial, 80
    parser.add_argument('--dropout_XJTU', type=float, default=0.1263514868613022, help='dropout')
    parser.add_argument('--dropout_TJU', type=float, default=0.12374924038427233, help='dropout')
    parser.add_argument('--dropout_MIT', type=float, default=0.1235582146396002, help='dropout')
    parser.add_argument('--dropout_HUST', type=float, default=0.10014057317121909, help='dropout')

    # Combination related
    parser.add_argument('--bagging_NN_lr_XJTU', type=float, default=0.02, help='bagging_NN_lr')
    parser.add_argument('--bagging_NN_lr_TJU', type=float, default=0.006456520169381107, help='bagging_NN_lr')
    parser.add_argument('--bagging_NN_lr_MIT', type=float, default=0.006663648646643286, help='bagging_NN_lr')
    parser.add_argument('--bagging_NN_lr_HUST', type=float, default=0.00022334184499247844, help='bagging_NN_lr')
    parser.add_argument('--bag_hidden_dim_XJTU', type=list, default=[50, 50], help='The hidden dim of Bagging Neural Network')
    parser.add_argument('--bag_hidden_dim_TJU', type=list, default=[50, 50], help='The hidden dim of Bagging Neural Network')
    parser.add_argument('--bag_hidden_dim_MIT', type=list, default=[50, 50], help='The hidden dim of Bagging Neural Network')
    parser.add_argument('--bag_hidden_dim_HUST', type=list, default=[100, 100], help='The hidden dim of Bagging Neural Network')
    parser.add_argument('--mono_bag_XJTU', type=float, default=0.4, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--mono_bag_TJU', type=float, default=0.01804601857570287, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--mono_bag_MIT', type=float, default=0.014599210344919067, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--mono_bag_HUST', type=float, default=0.9933428738041119, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')

    # Lax Model related
    parser.add_argument('--lr_net_LAX_XJTU', type=float, default=0.04, help='net learning rate')
    parser.add_argument('--lr_net_LAX_TJU', type=float, default=0.024314095112883502, help='net learning rate')
    parser.add_argument('--lr_net_LAX_MIT', type=float, default=0.011935902068420634, help='net learning rate')
    parser.add_argument('--lr_net_LAX_HUST', type=float, default=0.014526149878088328, help='net learning rate')
    parser.add_argument('--lr_y_LAX', type=float, default=0.08, help='y learning rate')

    parser.add_argument('--epoch_y_LAX', type=int, default=600, help='numbemr of y epoch')
    parser.add_argument('--epoch_th_LAX', type=int, default=3000, help='number of epochs without y optimization')

    parser.add_argument('--warmup_epochs_net_LAX', type=int, default=100, help='Number of warmup epochs of net')
    parser.add_argument('--warmup_epochs_y_LAX', type=int, default=100, help='Number of warmup epochs of y')
    parser.add_argument('--restart_period_LAX', type=int, default=100, help='Epochs between LR restarts')
    parser.add_argument('--min_lr_net_LAX', type=float, default=1e-5, help='Minimum learning rate for net')
    parser.add_argument('--min_lr_y_LAX', type=float, default=1e-4, help='Minimum learning rate for y')
    parser.add_argument('--F_hidden_dim_LAX', type=int, default=25, help='the hidden dim of F')

    parser.add_argument('--theta_LAX_XJTU', type=float, default=0.0, help='weight for mape loss')
    parser.add_argument('--theta_LAX_TJU', type=float, default=0.9290600824788182, help='weight for mape loss')
    parser.add_argument('--theta_LAX_MIT', type=float, default=0.4808822091755273, help='weight for mape loss')
    parser.add_argument('--theta_LAX_HUST', type=float, default=0.809772111877841, help='weight for mape loss')
    parser.add_argument('--zeta_LAX', type=float, default=1.0, help='wieght for mse loss')
    parser.add_argument('--kata_LAX', type=float, default=0.0, help='wieght for mae loss')
    parser.add_argument('--betha_LAX_XJTU', type=float, default=.0256, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_TJU', type=float, default=0.7447313058846572, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_MIT', type=float, default=0.5927778785956674, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--betha_LAX_HUST', type=float, default=0.846945564070922, help='loss = l_data + betha_lax * l_monoton')
    parser.add_argument('--dual_LAX_XJTU', type=float, default=.0, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_TJU', type=float, default=0.008827241117241124, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_MIT', type=float, default=0.0003537265723182956, help='weight for dual_H loss')
    parser.add_argument('--dual_LAX_HUST', type=float, default=0.570865721084567, help='weight for dual_H loss')
    parser.add_argument('--h_dim_LAX_XJTU', type=int, default=49, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_TJU', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_MIT', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--h_dim_LAX_HUST', type=int, default=58, help='number of inputs of H module.')
    parser.add_argument('--run_for_LAX', type=bool, help='run_for_LAX')
    parser.add_argument('--trained_by_y_opt_LAX', default= False, type=bool, help='Whether the loaded model was trained with y_opt in LAX')
    parser.add_argument('--flagsoori', type=bool, help='??')
    parser.add_argument('--beta_LAX_XJTU', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_TJU', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_MIT', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--beta_LAX_HUST', type=float, default= 0.0, help= 'it is the pde loss: loss = l_data + betha_lax * l_monoton + beta_lax * l_PDE')
    parser.add_argument('--dynamical_F_LAX', type=bool, default=True, help='this boolean specifies existence of Dynamical_F in our network.')

    parser.add_argument('--distance_block_LAX_XJTU', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_TJU', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_MIT', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    parser.add_argument('--distance_block_LAX_HUST', type= str, default="MLP",choices= ['SumProductNetwork', 'MLP', 'Transformer', 'Mahalanobis', 'Lp_norm'], 
                         help= 'choose one of these options for distance block: SumProductNetwork, MLP')
    
    parser.add_argument('--center_block_LAX_XJTU', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_TJU', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_MIT', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    parser.add_argument('--center_block_LAX_HUST', type= str, default= 'PhI', choices=['H*', 'PhiIntegrator ', 'PhI'],
                        help= 'choose one of these options for center block: H*, PhiIntegrator, PhI')
    
    parser.add_argument('--time_block_LAX_XJTU', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_TJU', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_MIT', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    parser.add_argument('--time_block_LAX_HUST', type=str, choices=['1/t', 'theta', 'multi_var_theta'], default='theta', 
                        help='Choose one of these options fo time_block: 1/t, theta')
    
    parser.add_argument('--s_LAX', type=str, default='MLP', choices=['ordinary_sum', 'MLP', 'Transformer'], 
                        help= 'ordinary Sum or an MLP block to train it.')
    # LAX architecture 
    
    parser.add_argument('--inside_h_star_layers_LAX_XJTU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_XJTU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_XJTU', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_XJTU', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_XJTU', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_XJTU', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_XJTU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_XJTU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_XJTU', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_LAX_XJTU', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--H_out_LAX_XJTU', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_XJTU', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_XJTU', type = int, default = 1, help = 'dimension of whole LAX output')

    parser.add_argument('--inside_h_star_layers_LAX_TJU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_TJU', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_TJU', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_TJU', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_TJU', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_TJU', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_TJU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_TJU', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_TJU', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_LAX_TJU', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--H_out_LAX_TJU', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_TJU', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_TJU', type = int, default = 1, help = 'dimension of whole LAX output')
                        
    parser.add_argument('--inside_h_star_layers_LAX_MIT', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_MIT', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_MIT', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_MIT', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_MIT', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_MIT', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_MIT', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_MIT', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_MIT', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_LAX_MIT', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--H_out_LAX_MIT', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_MIT', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_MIT', type = int, default = 1, help = 'dimension of whole LAX output')
                        
    parser.add_argument('--inside_h_star_layers_LAX_HUST', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_phi_layers_LAX_HUST', type=int, nargs= '+', default=[64, 32], help= 'List of number of neurons inside hidden layers of H_Star block.')
    parser.add_argument('--inside_g_layers_LAX_HUST', type= int, nargs= '+', default=[16], help = 'List of number of neurons inside hidden layers of g block.')
    parser.add_argument('--inside_S_MLP_layers_LAX_HUST', type= int, nargs= '+', default=[32, 16, 8], help = 'List of number of neurons inside hidden layers of S_MLP block.')
    parser.add_argument('--inside_betan_layers_LAX_HUST', type= int, nargs= '+', default=[16, 8], help = 'List of number of neurons inside hidden layers of betan block.')
    parser.add_argument('--inside_distance_block_MLP_layers_LAX_HUST', type= int, nargs= '+', default=[64], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_theta_layers_LAX_HUST', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--inside_multivar_theta_layers_LAX_HUST', type= int, nargs= '+', default=[32], help = 'List of number of neurons inside hidden layers of distance block MLP.')
    parser.add_argument('--g_dim_LAX_HUST', type= int, default=16, help= 'g input dimension')
    parser.add_argument('--g_out_LAX_HUST', type= int, default=1, help= 'g output dimension')
    parser.add_argument('--H_out_LAX_HUST', type= int, default=1, help= 'h output dimension')
    parser.add_argument('--phi_out_LAX_HUST', type= int, default=1, help= 'phi output dimension')
    parser.add_argument('--dim_output_LAX_HUST', type = int, default = 1, help = 'dimension of whole LAX output')
    # scheduler related
    parser.add_argument('--epochs', type=int, default=2000, help='epoch') # 2000, 200
    parser.add_argument('--early_stop', type=int, default=40, help='early stop') # 80, 20
    parser.add_argument('--warmup_epochs_XJTU', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_epochs_TJU', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_MIT', type=int, default=50, help='warmup epoch')
    parser.add_argument('--warmup_epochs_HUST', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr_XJTU', type=float, default=0.006268229580119017, help='warmup lr')
    parser.add_argument('--warmup_lr_TJU', type=float, default=0.006456520169381107, help='warmup lr')
    parser.add_argument('--warmup_lr_MIT', type=float, default=0.006158883939912325, help='warmup lr')
    parser.add_argument('--warmup_lr_HUST', type=float, default=0.005237775841792184, help='warmup lr')
    parser.add_argument('--lr_XJTU', type=float, default=0.006252096490545448, help='base lr')
    parser.add_argument('--lr_TJU', type=float, default=0.001221299541334721, help='base lr')
    parser.add_argument('--lr_MIT', type=float, default=0.00011720460436102193, help='base lr')
    parser.add_argument('--lr_HUST', type=float, default=0.0007215089581483758, help='base lr')
    parser.add_argument('--final_lr_XJTU', type=float, default=1.5973286128658127e-06, help='final lr')
    parser.add_argument('--final_lr_TJU', type=float, default=2.1099224334559236e-05, help='final lr')
    parser.add_argument('--final_lr_MIT', type=float, default=1.447250132592353e-06, help='final lr')
    parser.add_argument('--final_lr_HUST', type=float, default=1.747286485446226e-05, help='final lr')
    parser.add_argument('--lr_F_XJTU', type=float, default=0.00881767573907948, help='lr of F')
    parser.add_argument('--lr_F_TJU', type=float, default=0.0004916804735455082, help='lr of F')
    parser.add_argument('--lr_F_MIT', type=float, default=2.181510135329394e-05, help='lr of F')
    parser.add_argument('--lr_F_HUST', type=float, default=4.8990846543190585e-05, help='lr of F')

    # Branch net related
    parser.add_argument('--m', type=int, default=500, help='Branch Input dimension')
    # Trunk net related
    parser.add_argument('--dim_x', type=int, default=1, help='Trunk Input dimension')

    # Loss related
    parser.add_argument('--loss', type=str, default='MSE', help='Calculating gradients for each weights')
    parser.add_argument('--alpha_XJTU', type=float, default=0.11809194837918663, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_XJTU', type=float, default=0.015956048434866418, help='total_loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_MIT', type=float, default=0.8843241457826115, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_MIT', type=float, default=0.02840429623105723, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_TJU', type=float, default=0.5999950741786956, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_TJU', type=float, default=0.01804601857570287, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_HUST', type=float, default=0.2795906470184894, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta_HUST', type=float, default=0.1308092300047794, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    
    # The AdaModel class inherits the DOPFormer class, and the above parameters are all parameters of DOPFormer.
    # The following are the parameters of AdaModel.
    # adaption related
    parser.add_argument('--pretrained_model', type=str, default=None, help='The saving path of the model trained in the source domain')

    parser.add_argument('--target_data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--target_batch', type=int, default=-1, choices=[-1,0,1,2,3,4,5],
                        help='XJTU dataset is divided into 6 batches, and TJU dataset is divided into 3 batches. '
                             'If target_data is XJTU, the value range of target_batch is [-1,0,1,2,3,4,5];'
                             'If target_data is TJU, the value range of target_batch is [-1,0,1,2];'
                             'If it is other datasets, ignore target_batch')
    # Directory related
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='AdaModel_test', help='save folder')
    
    args = parser.parse_args()
    return args
