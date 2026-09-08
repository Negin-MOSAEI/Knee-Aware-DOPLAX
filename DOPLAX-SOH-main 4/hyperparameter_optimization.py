from dataloader.data_helper import load_XJTU_data
from dataloader.data_helper import load_TJU_data
from dataloader.data_helper import load_MIT_data
from dataloader.data_helper import load_HUST_data
from Model.PI_nets import PINN, DeepOPINN, DOPFormer
# from Model.DD_nets import DeepONet
from Model.Combination_nets import Bagging_u, LAX_predictor, DOPDeepOLAX
from Model.PI_nets.LAX import run_training_and_evaluation
from utils.util import write_to_file, write_to_json
from utils.arguments import get_PINN_args, get_DeepONet_args, get_DeepOPINN_args, get_DOPFormer_args, get_LAX_args, get_Bagging_u_args, get_LAX_predictor_args
import os
import optuna
import math
import warnings
warnings.filterwarnings('ignore')



def set_XJTU_suggestions_args(trial, args, run_info):
    if run_info['run_for_a_model_explicitly']:
        if run_info['model_name'] == 'DeepOPINN':
            args.lr_XJTU = trial.suggest_loguniform('lr_XJTU', 1e-4, 1e-1)
            args.warmup_lr_XJTU = trial.suggest_loguniform('warmup_lr_XJTU', 1e-5, 1e-2)
            args.lr_F_XJTU = trial.suggest_loguniform('lr_F_XJTU', 1e-5, 1e-2)
            args.final_lr_XJTU = trial.suggest_loguniform('final_lr_XJTU', 1e-6, 1e-3)
            args.F_hidden_dim_XJTU = trial.suggest_categorical('F_hidden_dim_XJTU', [32, 64, 80, 128, 256])
            args.F_layers_num_XJTU = trial.suggest_int('F_layers_num_XJTU', 2, 5)
            args.dropout_XJTU = trial.suggest_uniform('dropout_XJTU', 0.1, 0.5)
            args.alpha_XJTU = trial.suggest_uniform('alpha_XJTU', 0.1, 1.0)
            args.beta_XJTU = trial.suggest_uniform('beta_XJTU', 0.01, 10)
            args.batch_size_XJTU = trial.suggest_categorical('batch_size_XJTU', [64, 128, 256, 512])
            args.warmup_epochs_XJTU = trial.suggest_categorical('warmup_epochs_XJTU', [10, 20, 30, 40, 50])  
        elif run_info['model_name'] == 'DOPFormer':
            args.lr_XJTU = trial.suggest_loguniform('lr_XJTU', 1e-4, 1e-2)
            args.warmup_lr_XJTU = trial.suggest_loguniform('warmup_lr_XJTU', 1e-4, 1e-2)
            args.lr_F_XJTU = trial.suggest_loguniform('lr_F_XJTU', 1e-5, 1e-2)
            args.final_lr_XJTU = trial.suggest_loguniform('final_lr_XJTU', 1e-6, 1e-4)
            # args.F_hidden_dim_XJTU = trial.suggest_categorical('F_hidden_dim_XJTU', [32, 64, 128, 256, 512])
            # args.F_layers_num_XJTU = trial.suggest_int('F_layers_num_XJTU', 2, 5)
            # args.dropout_XJTU = trial.suggest_uniform('dropout_XJTU', 0.1, 0.5)
            args.alpha_XJTU = trial.suggest_uniform('alpha_XJTU', 0.1, 1.0)
            args.beta_XJTU = trial.suggest_uniform('beta_XJTU', 0.01, 2)
            # args.batch_size_XJTU = trial.suggest_categorical('batch_size_XJTU', [64, 128, 256, 512])
            # args.warmup_epochs_XJTU = trial.suggest_categorical('warmup_epochs_XJTU', [10, 20, 30, 40, 50])
            args.d_model_XJTU = trial.suggest_categorical('d_model_XJTU', [32, 64, 128, 256, 512])
            args.d_hidden_XJTU = trial.suggest_categorical('d_hidden_XJTU', [32, 64, 128, 256, 512])
            args.N_XJTU = trial.suggest_int('N_XJTU', 1, 5)
            args.heads_XJTU = trial.suggest_categorical('heads_XJTU', [2, 4])
            args.transformer_dropout_XJTU = trial.suggest_uniform('transformer_dropout_XJTU', 0.01, 0.5)
        else:
            args.betha_LAX_XJTU = trial.suggest_float("betha_LAX_XJTU", 1e-2, 1.0)
            args.dual_LAX_XJTU = trial.suggest_float("dual_LAX_XJTU", .002, 1.2)
            args.lr_net_LAX_XJTU = trial.suggest_float("lr_net_LAX_XJTU", .01 ,.3)
            args.theta_LAX_XJTU = trial.suggest_float("theta_LAX_XJTU", .0, 1.)

    else:
        args.bagging_NN_lr_XJTU = trial.suggest_loguniform('bagging_NN_lr_XJTU', 1e-5, 1e-2)
        args.bag_hidden_dim_XJTU = [
            trial.suggest_int(f"bag_hidden_dim_{i}_XJTU", 1, 101) 
            for i in range(trial.suggest_int("bag_hidden_dim_length_XJTU", 1, 4))
        ]
        args.mono_bag_XJTU = trial.suggest_uniform('mono_bag_XJTU', 0.01, 10)
        # args.alpha_XJTU = trial.suggest_uniform('alpha_XJTU', 0.1, 1.0)
        # args.beta_XJTU = trial.suggest_uniform('beta_XJTU', 0.01, 10)
        # args.betha_LAX_XJTU = trial.suggest_float("betha_LAX_XJTU", 1e-2, 1.0)
        # args.dual_LAX_XJTU = trial.suggest_float("dual_LAX_XJTU", .002, 1.2)


def set_TJU_suggestions_args(trial, args, run_info):
    if run_info['run_for_a_model_explicitly']:
        if run_info['model_name'] == 'DeepOPINN':
            args.lr_TJU = trial.suggest_loguniform('lr_TJU', 1e-4, 1e-1)
            args.warmup_lr_TJU = trial.suggest_loguniform('warmup_lr_TJU', 1e-5, 1e-2)
            args.lr_F_TJU = trial.suggest_loguniform('lr_F_TJU', 1e-5, 1e-2)
            args.final_lr_TJU = trial.suggest_loguniform('final_lr_TJU', 1e-6, 1e-3)
            args.F_hidden_dim_TJU = trial.suggest_categorical('F_hidden_dim_TJU', [32, 64, 80, 128, 256])
            args.F_layers_num_TJU = trial.suggest_int('F_layers_num_TJU', 2, 5)
            args.dropout_TJU = trial.suggest_uniform('dropout_TJU', 0.1, 0.5)
            args.alpha_TJU = trial.suggest_uniform('alpha_TJU', 0.1, 1.0)
            args.beta_TJU = trial.suggest_uniform('beta_TJU', 0.01, 10)
            args.batch_size_TJU = trial.suggest_categorical('batch_size_TJU', [64, 128, 256, 512])
            args.warmup_epochs_TJU = trial.suggest_categorical('warmup_epochs_TJU', [10, 20, 30, 40, 50])
        elif run_info['model_name'] == 'DOPFormer':
            args.lr_TJU = trial.suggest_loguniform('lr_TJU', 1e-4, 1e-2)
            args.warmup_lr_TJU = trial.suggest_loguniform('warmup_lr_TJU', 1e-4, 1e-2)
            args.lr_F_TJU = trial.suggest_loguniform('lr_F_TJU', 1e-5, 1e-2)
            args.final_lr_TJU = trial.suggest_loguniform('final_lr_TJU', 1e-6, 1e-4)
            # args.F_hidden_dim_TJU = trial.suggest_categorical('F_hidden_dim_TJU', [32, 64, 128, 256, 512])
            # args.F_layers_num_TJU = trial.suggest_int('F_layers_num_TJU', 2, 5)
            # args.dropout_TJU = trial.suggest_uniform('dropout_TJU', 0.1, 0.5)
            args.alpha_TJU = trial.suggest_uniform('alpha_TJU', 0.1, 1.0)
            args.beta_TJU = trial.suggest_uniform('beta_TJU', 0.01, 2)
            # args.batch_size_TJU = trial.suggest_categorical('batch_size_TJU', [64, 128, 256, 512])
            # args.warmup_epochs_TJU = trial.suggest_categorical('warmup_epochs_TJU', [10, 20, 30, 40, 50])
            args.d_model_TJU = trial.suggest_categorical('d_model_TJU', [32, 64, 128, 256, 512])
            args.d_hidden_TJU = trial.suggest_categorical('d_hidden_TJU', [32, 64, 128, 256, 512])
            args.N_TJU = trial.suggest_int('N_TJU', 1, 5)
            args.heads_TJU = trial.suggest_categorical('heads_TJU', [2, 4])
            args.transformer_dropout_TJU = trial.suggest_uniform('transformer_dropout_TJU', 0.01, 0.5)
        else:
            args.betha_LAX_TJU = trial.suggest_float("betha_LAX_TJU", 1e-2, 1.0)
            args.dual_LAX_TJU = trial.suggest_float("dual_LAX_TJU", .002, 1.2)
            args.lr_net_LAX_TJU = trial.suggest_float("lr_net_LAX_TJU", .01 ,.3)
            args.theta_LAX_TJU = trial.suggest_float("theta_LAX_TJU", .0, 1.)

    else:
        args.bagging_NN_lr_TJU = trial.suggest_loguniform('bagging_NN_lr_TJU', 1e-5, 1e-2)
        args.bag_hidden_dim_TJU = [
            trial.suggest_int(f"bag_hidden_dim_{i}_TJU", 1, 101) 
            for i in range(trial.suggest_int("bag_hidden_dim_length_TJU", 1, 4))
        ]
        args.mono_bag_TJU = trial.suggest_uniform('mono_bag_TJU', 0.01, 10)
        # args.alpha_TJU = trial.suggest_uniform('alpha_TJU', 0.1, 1.0)
        # args.beta_TJU = trial.suggest_uniform('beta_TJU', 0.01, 10)
        # args.betha_LAX_TJU = trial.suggest_float("betha_LAX_TJU", 1e-2, 1.0)
        # args.dual_LAX_TJU = trial.suggest_float("dual_LAX_TJU", .002, 1.2)
        

def set_MIT_suggestions_args(trial, args, run_info):
    if run_info['run_for_a_model_explicitly']:
        if run_info['model_name'] == 'DeepOPINN':
            args.lr_MIT = trial.suggest_loguniform('lr_MIT', 1e-4, 1e-1)
            args.warmup_lr_MIT = trial.suggest_loguniform('warmup_lr_MIT', 1e-5, 1e-2)
            args.lr_F_MIT = trial.suggest_loguniform('lr_F_MIT', 1e-5, 1e-2)
            args.final_lr_MIT = trial.suggest_loguniform('final_lr_MIT', 1e-6, 1e-3)
            args.F_hidden_dim_MIT = trial.suggest_categorical('F_hidden_dim_MIT', [32, 64, 80, 128, 256])
            args.F_layers_num_MIT = trial.suggest_int('F_layers_num_MIT', 2, 5)
            args.dropout_MIT = trial.suggest_uniform('dropout_MIT', 0.1, 0.5)
            args.alpha_MIT = trial.suggest_uniform('alpha_MIT', 0.1, 1.0)
            args.beta_MIT = trial.suggest_uniform('beta_MIT', 0.01, 10)
            args.batch_size_MIT = trial.suggest_categorical('batch_size_MIT', [64, 128, 256, 512])
            args.warmup_epochs_MIT = trial.suggest_categorical('warmup_epochs_MIT', [10, 20, 30, 40, 50])
        elif run_info['model_name'] == 'DOPFormer':
            args.lr_MIT = trial.suggest_loguniform('lr_MIT', 1e-4, 1e-2)
            args.warmup_lr_MIT = trial.suggest_loguniform('warmup_lr_MIT', 1e-4, 1e-2)
            args.lr_F_MIT = trial.suggest_loguniform('lr_F_MIT', 1e-5, 1e-3)
            args.final_lr_MIT = trial.suggest_loguniform('final_lr_MIT', 1e-6, 1e-4)
            # args.F_hidden_dim_MIT = trial.suggest_categorical('F_hidden_dim_MIT', [32, 64, 128, 256, 512])
            # args.F_layers_num_MIT = trial.suggest_int('F_layers_num_MIT', 2, 5)
            # args.dropout_MIT = trial.suggest_uniform('dropout_MIT', 0.1, 0.5)
            args.alpha_MIT = trial.suggest_uniform('alpha_MIT', 0.1, 1.0)
            args.beta_MIT = trial.suggest_uniform('beta_MIT', 0.01, 2)
            # args.batch_size_MIT = trial.suggest_categorical('batch_size_MIT', [64, 128, 256, 512])
            # args.warmup_epochs_MIT = trial.suggest_categorical('warmup_epochs_MIT', [10, 20, 30, 40, 50])
            args.d_model_MIT = trial.suggest_categorical('d_model_MIT', [32, 64, 128, 256, 512])
            args.d_hidden_MIT = trial.suggest_categorical('d_hidden_MIT', [32, 64, 128, 256, 512])
            args.N_MIT = trial.suggest_int('N_MIT', 1, 5)
            args.heads_MIT = trial.suggest_categorical('heads_MIT', [2, 4])
            args.transformer_dropout_MIT = trial.suggest_uniform('transformer_dropout_MIT', 0.01, 0.5)
        else:
            args.betha_LAX_MIT = trial.suggest_float("betha_LAX_MIT", 1e-2, 1.0)
            args.dual_LAX_MIT = trial.suggest_float("dual_LAX_MIT", .002, 1.2)
            args.lr_net_LAX_MIT = trial.suggest_float("lr_net_LAX_MIT", .01 ,.3)
            args.theta_LAX_MIT = trial.suggest_float("theta_LAX_MIT", .0, 1.)

    else:
        args.bagging_NN_lr_MIT = trial.suggest_loguniform('bagging_NN_lr_MIT', 1e-5, 1e-3)
        args.bag_hidden_dim_MIT = [
            trial.suggest_int(f"bag_hidden_dim_{i}_MIT", 1, 101) 
            for i in range(trial.suggest_int("bag_hidden_dim_length_MIT", 1, 4))
        ]
        args.mono_bag_MIT = trial.suggest_uniform('mono_bag_MIT', 0.01, 2)
        # args.alpha_MIT = trial.suggest_uniform('alpha_MIT', 0.1, 1.0)
        # args.beta_MIT = trial.suggest_uniform('beta_MIT', 0.01, 10)
        # args.betha_LAX_MIT = trial.suggest_float("betha_LAX_MIT", 1e-2, 1.0)
        # args.dual_LAX_MIT = trial.suggest_float("dual_LAX_MIT", .002, 1.2)
        

def set_HUST_suggestions_args(trial, args, run_info):
    if run_info['run_for_a_model_explicitly']:
        if run_info['model_name'] == 'DeepOPINN':
            args.lr_HUST = trial.suggest_loguniform('lr_HUST', 1e-4, 1e-1)
            args.warmup_lr_HUST = trial.suggest_loguniform('warmup_lr_HUST', 1e-5, 1e-2)
            args.lr_F_HUST = trial.suggest_loguniform('lr_F_HUST', 1e-5, 1e-2)
            args.final_lr_HUST = trial.suggest_loguniform('final_lr_HUST', 1e-6, 1e-3)
            args.F_hidden_dim_HUST = trial.suggest_categorical('F_hidden_dim_HUST', [32, 64, 80, 128, 256])
            args.F_layers_num_HUST = trial.suggest_int('F_layers_num_HUST', 2, 5)
            args.dropout_HUST = trial.suggest_uniform('dropout_HUST', 0.1, 0.5)
            args.alpha_HUST = trial.suggest_uniform('alpha_HUST', 0.1, 1.0)
            args.beta_HUST = trial.suggest_uniform('beta_HUST', 0.01, 10)
            args.batch_size_HUST = trial.suggest_categorical('batch_size_HUST', [64, 128, 256, 512])
            args.warmup_epochs_HUST = trial.suggest_categorical('warmup_epochs_HUST', [10, 20, 30, 40, 50])
        elif run_info['model_name'] == 'DOPFormer':
            args.lr_HUST = trial.suggest_loguniform('lr_HUST', 1e-4, 1e-2)
            args.warmup_lr_HUST = trial.suggest_loguniform('warmup_lr_HUST', 1e-4, 1e-2)
            args.lr_F_HUST = trial.suggest_loguniform('lr_F_HUST', 1e-5, 1e-3)
            args.final_lr_HUST = trial.suggest_loguniform('final_lr_HUST', 1e-6, 1e-4)
            # args.F_hidden_dim_HUST = trial.suggest_categorical('F_hidden_dim_HUST', [32, 64, 128, 256, 512])
            # args.F_layers_num_HUST = trial.suggest_int('F_layers_num_HUST', 2, 5)
            # args.dropout_HUST = trial.suggest_uniform('dropout_HUST', 0.1, 0.5)
            args.alpha_HUST = trial.suggest_uniform('alpha_HUST', 0.1, 1.0)
            args.beta_HUST = trial.suggest_uniform('beta_HUST', 0.01, 2)
            # args.batch_size_HUST = trial.suggest_categorical('batch_size_HUST', [64, 128, 256, 512])
            # args.warmup_epochs_HUST = trial.suggest_categorical('warmup_epochs_HUST', [10, 20, 30, 40, 50])
            args.d_model_HUST = trial.suggest_categorical('d_model_HUST', [32, 64, 128, 256, 512])
            args.d_hidden_HUST = trial.suggest_categorical('d_hidden_HUST', [32, 64, 128, 256, 512])
            args.N_HUST = trial.suggest_int('N_HUST', 1, 5)
            args.heads_HUST = trial.suggest_categorical('heads_HUST', [2, 4])
            args.transformer_dropout_HUST = trial.suggest_uniform('transformer_dropout_HUST', 0.01, 0.5)
        else:
            args.betha_LAX_HUST = trial.suggest_float("betha_LAX_HUST", 1e-2, 1.0)
            args.dual_LAX_HUST = trial.suggest_float("dual_LAX_HUST", .002, 1.2)
            args.lr_net_LAX_HUST = trial.suggest_float("lr_net_LAX_HUST", .01 ,.3)
            args.theta_LAX_HUST = trial.suggest_float("theta_LAX_HUST", .0, 1.)

    else:
        args.bagging_NN_lr_HUST = trial.suggest_loguniform('bagging_NN_lr_HUST', 1e-5, 1e-2)
        args.bag_hidden_dim_HUST = [
            trial.suggest_int(f"bag_hidden_dim_{i}_HUST", 1, 101) 
            for i in range(trial.suggest_int("bag_hidden_dim_length_HUST", 1, 4))
        ]
        args.mono_bag_HUST = trial.suggest_uniform('mono_bag_HUST', 0.01, 10)
        # args.alpha_HUST = trial.suggest_uniform('alpha_HUST', 0.1, 1.0)
        # args.beta_HUST = trial.suggest_uniform('beta_HUST', 0.01, 10)
        # args.betha_LAX_HUST = trial.suggest_float("betha_LAX_HUST", 1e-2, 1.0)
        # args.dual_LAX_HUST = trial.suggest_float("dual_LAX_HUST", .002, 1.2)

    
def objective(trial, run_info:dict, dataset_name:str, batch_name:str, pth_files:dict=None):
    try:
        if run_info['run_for_a_model_explicitly'] == False:
            args = get_DOPLAX_args()
        else:
            if run_info['model_name'] == 'DeepOPINN':
                args = get_DeepOPINN_args()
            elif run_info['model_name'] == 'DOPFormer':
                args = get_DOPFormer_args()
            else:
                args = get_LAX_args()
        
        # Check if this trial has already been completed
        # Get all completed trials (filter out running/pruned/failed ones)
        completed_trials = [t for t in trial.study.get_trials() if t.state.is_finished()]
        
        # Check if current trial's number exists in completed trials
        if any(t.number == trial.number for t in completed_trials):
            print(f"Trial {trial.number} already completed, skipping...")
            # Return the existing value if available, otherwise return infinity
            existing_trial = next(t for t in completed_trials if t.number == trial.number)
            return existing_trial.value if existing_trial.value is not None else float('inf')  
        
        # Hyperparameter suggestions
        if dataset_name == 'XJTU':
            set_XJTU_suggestions_args(trial, args, run_info)
        elif dataset_name == 'TJU':
            set_TJU_suggestions_args(trial, args, run_info)
        elif dataset_name == 'MIT':
            set_MIT_suggestions_args(trial, args, run_info)
        else:
            set_HUST_suggestions_args(trial, args, run_info)
            
        # Set up folders and paths
        setattr(args, 'data', dataset_name)
        setattr(args, 'batch', batch_name)
        setattr(args, 'run_optuna', True)
        optuna_root = f'tmp/{model_name}/{dataset_name}/{batch_name}'
        setattr(args, 'optuna_root', optuna_root)
        trial_folder = os.path.join(optuna_root, f'trial_{trial.number}')
        os.makedirs(trial_folder, exist_ok=True)
        
        if run_info['run_for_a_model_explicitly'] and run_info['model_name'] == 'LAX':
            setattr(args, "results_path", trial_folder)
            setattr(args, 'run_mode', 'LAX')

        if (run_info['run_for_a_model_explicitly'] == False) or (run_info['run_for_a_model_explicitly'] and run_info['model_name'] != 'LAX'):
            log_dir = 'logging.txt'
            setattr(args, "save_folder", trial_folder)
            setattr(args, "log_dir", log_dir)
            setattr(args, 'run_mode', 'others')

        if dataset_name == 'XJTU':
            if args.run_mode == 'others':
                args.batch_size = args.batch_size_XJTU
            load_data = load_XJTU_data

        elif dataset_name == 'TJU':
            if args.run_mode == 'others':
                # args.batch_size = args.batch_size_TJU
                if trial >= 100 and trial < 130:
                    args.batch_size = 128
                elif trial >= 130 and trial < 160:
                    args.batch_size = 512
                else:
                    args.batch_size = 1024
            load_data = load_TJU_data

        elif dataset_name == 'MIT':
            if args.run_mode == 'others':
                # args.batch_size = args.batch_size_MIT
                if trial >= 38 and trial < 100:
                    args.batch_size = 512
                elif trial >= 100 and trial < 150:
                    args.batch_size = 768
                else:
                    args.batch_size = 1024
            load_data = load_MIT_data

        else:
            if args.run_mode == 'others':
                # args.batch_size = args.batch_size_HUST
                if trial >= 28 and trial < 80:
                    args.batch_size = 512
                elif trial >= 80 and trial < 140:
                    args.batch_size = 768
                else:
                    args.batch_size = 1024
            load_data = load_HUST_data

        if run_info['run_for_a_model_explicitly'] == False and pth_files is not None:
            setattr(args, 'DOP_weights', pth_files['DOP'])
            setattr(args, 'LAX_weights', pth_files['LAX'])

        # Run training
        dataloader = load_data(args)

        if run_info['run_for_a_model_explicitly']:
            if run_info['model_name'] == 'DeepOPINN':
                setattr(args, 'run_mode', 'DeepOPINN')
                deepopinn = DeepOPINN.Model(args)
                train_result, rmse_val, mape_val = deepopinn.Train(
                    trainloader=dataloader['train'],
                    validloader=dataloader['valid'],
                    testloader=dataloader['test']
                )
            elif run_info['model_name'] == 'DOPFormer':
                setattr(args, 'run_mode', 'DOPFormer')
                dopformer = DOPFormer.Model(args)
                train_result, rmse_val, mape_val = dopformer.Train(
                    trainloader=dataloader['train'],
                    validloader=dataloader['valid'],
                    testloader=dataloader['test']
                )
            else:
                test_metrics, val_loss_rmse_metric = run_training_and_evaluation(args=args, dataloader=dataloader, batch_name=args.batch, experiment_id=0)
                logging_file = os.path.join(trial_folder, 'test_metrics_results.json')
                logging_info = {
                    'Batch': args.batch,
                    'Experiment': 1,
                    'MSE': test_metrics['mse'],
                    'RMSE': test_metrics['rmse'],
                    'MAPE': test_metrics['mape'],
                    'MAE': test_metrics['mae'],
                    'DUAL': test_metrics['dual']
                }
                write_to_json(file_path=logging_file, info=logging_info)  
        else:
            setattr(args, 'run_mode', 'DOPLAX')
            doplax = DOPLAX.Model(args)
            train_result, rmse_val, mape_val = doplax.Train(
                trainloader=dataloader['train'],
                validloader=dataloader['valid'],
                testloader=dataloader['test']
            )
        
        # Save results
        trial_results = {
            'dataset': dataset_name,
            'trial_number': trial.number,
            'params': trial.params,
        }
        file_path = os.path.join(trial_folder, f'trial{trial.number}_results.json')
        write_to_json(file_path=file_path, info=trial_results)
        
        # Return RMSE as objective value
        if run_info['run_for_a_model_explicitly']:
            if run_info['model_name'] == 'DeepOPINN':
                if train_result == 'valid':
                    return math.sqrt(deepopinn.valid_loss_history[-1]) if deepopinn.valid_loss_history else float('inf')
                else:
                    return float('inf')
            elif run_info['model_name'] == 'DOPFormer':
                if train_result == 'valid':
                    return math.sqrt(dopformer.valid_loss_history[-1]) if dopformer.valid_loss_history else float('inf')
                else:
                    return float('inf')
            else:
                return val_loss_rmse_metric[-1] if val_loss_rmse_metric else float('inf')
        else:
            return math.sqrt(doplax.valid_loss_history[-1]) if doplax.valid_loss_history else float('inf')
    
    except Exception as e:
        error_info = f"Trial {trial.number} failed with error:\n{str(e)}"
        print(error_info)
        return float('inf')


def run_optuna(num_of_trials:int, run_info:dict, dataset_name:str, batch_name:str, path_to_weights:str, resume:bool=False):
    if type(run_info['run_for_a_model_explicitly']) != bool:
        raise ValueError("run_for_a_model_explicitly must be either True or False")
    
    if run_info['run_for_a_model_explicitly']:
        if run_info['model_name'] not in ["PINN", "DeepONet", "DeepOPINN", "DOPFormer", "LAX"]:
            raise ValueError("model_name must be one of these models, 'PINN', 'DeepONet', 'DeepOPINN', 'DOPFormer', or 'LAX'")
    else:
        if run_info['model_name'] not in ["Bagging_u", "LAX_predictor"]:
            raise ValueError("model_name must be one of these models, 'Bagging_u', or 'LAX_predictor'") 

    if dataset_name not in ['XJTU', 'TJU', 'MIT', 'HUST']:
        raise ValueError("Dataset name must be one of 'XJTU', 'TJU', 'MIT', 'HUST'")
    
    # MIT and HUST batch validations are handled via specific data helper logic
    if run_info['run_for_a_model_explicitly'] == False:
        if dataset_name == 'XJTU' and batch_name not in ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']:
            raise ValueError("When dataset name is 'XJTU', batch name must be one of '2C', '3C', 'R2.5', 'R3', 'RW', 'satellite'")
        
        if dataset_name == 'TJU' and batch_name not in ['NCA', 'NCM', 'NCM_NCA']:
            raise ValueError("When dataset name is 'TJU', batch name must be one of 'NCA', 'NCM', 'NCM_NCA'")
    
    if run_info['run_for_a_model_explicitly'] == False:
        args = get_DOPLAX_args()
        if dataset_name in ['MIT', 'HUST']:
            DOP_pth_file = os.path.join(path_to_weights, 'DeepOPINN', f'DeepOPINN_model_{dataset_name}.pth')
            LAX_pth_file = os.path.join(path_to_weights, 'LAX', f'LAX_model_{dataset_name}.pth')
        else:
            DOP_pth_file = os.path.join(path_to_weights, 'DeepOPINN', f'DeepOPINN_model_{dataset_name}_{batch_name}.pth')
            LAX_pth_file = os.path.join(path_to_weights, 'LAX', f'LAX_model_{dataset_name}_{batch_name}.pth')
        
        pth_files = {
            'DOP': DOP_pth_file,
            'LAX': LAX_pth_file
        }
    else:
        if run_info['model_name'] == 'DeepOPINN':
            args = get_DeepOPINN_args()
        elif run_info['model_name'] == 'DOPFormer':
            args = get_DOPFormer_args()
        else:
            args = get_LAX_args()
        pth_files = None

    global model_name
    if run_info['run_for_a_model_explicitly']:
        model_name = run_info["model_name"]
    else:
        model_name = 'DOPLAX'

    setattr(args, 'data', dataset_name)
    setattr(args, 'batch', batch_name)
    optuna_root = f'tmp/{model_name}/{dataset_name}/{batch_name}'
    setattr(args, 'optuna_root', optuna_root)
    os.makedirs(optuna_root, exist_ok=True)
    
    # Database configuration
    storage_url = f"sqlite:///{optuna_root}/optuna_{dataset_name}_studies.db"
    
    if resume:
        # Load existing study
        study = optuna.load_study(
            study_name=f"hyperparameter_optimization_for_{dataset_name}",
            storage=storage_url
        )
        print(f"Resuming study with {len(study.trials)} existing trials")
    else:
        # Create a new study
        study = optuna.create_study(
            study_name=f"hyperparameter_optimization_for_{dataset_name}",
            storage=storage_url,
            direction="minimize",
            load_if_exists=True
        )
        print("Created new study")
    
    # Add callback to save progress periodically
    study.optimize(
        lambda trial: objective(trial, run_info, dataset_name, batch_name, pth_files),
        n_trials=num_of_trials,
        callbacks=[save_study_callback],
        gc_after_trial=True
    )
    
    # Save final results
    print("Best trial:")
    print(study.best_trial)
    print("Best hyperparameters:", study.best_trial.params)
    
    info = f"""
    Dataset: {dataset_name}\n    
    Best trial:
    Number: {study.best_trial.number}
    Value: {study.best_trial.value}
    Params: {study.best_trial.params}
    User attrs: {study.best_trial.user_attrs}
    """
    best_trial_file = os.path.join(optuna_root, f'{dataset_name}_best_trial_results.txt')
    write_to_file(file_path=best_trial_file, info=info)
    
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(optuna_root, f'{dataset_name}_all_trials.csv'), index=False)
    print(f"Saved best trial results to {best_trial_file}")
    print(f"Saved all trials data to {os.path.join(optuna_root, f'{dataset_name}_all_trials.csv')}")


def save_study_callback(study, trial):
    """
    Callback to save study after each trial
    """
    # Save current best trial info
    if study.best_trial.number == trial.number:
        best_info = f"New best trial: {trial.number} with value: {trial.value}\nParams: {trial.params}"
        first_key =list(trial.params.keys())[0]
        dataset_name = first_key.split('_')[-1]
        with open(os.path.join(study.user_attrs.get('optuna_root', '.'), f'tmp/{model_name}/{dataset_name}/{dataset_name}_best_progress.txt'), 'a') as f:
            f.write(best_info + "\n")
