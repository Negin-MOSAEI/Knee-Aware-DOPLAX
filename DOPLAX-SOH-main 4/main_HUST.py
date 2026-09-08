from dataloader.data_helper import load_HUST_data
from Model.PI_nets import PINN, DONG, DeepOPINN, DOPFormer, FormerPINN
from Model.DD_nets import PINNsFormer
from Model.Combination_nets import Bagging_u, LAX_predictor, DeepOLAX, DOPDeepOLAX
import os
from utils.arguments import get_DeepONet_args, get_PINNsFormer_args, get_PINN_args, get_DONG_args, get_DeepOPINN_args, get_DOPFormer_args, get_LAX_args, get_Bagging_u_args, get_LAX_predictor_args, get_DeepOLAX_args, get_DOPDeepOLAX_args, get_FormerPINN_args
from utils.util import write_to_file, write_to_Excel
from Model.utils.util import get_val_metrics_in_each_logfile
from hyperparameter_optimization import run_optuna
from Model.PI_nets.LAX import run_training_and_evaluation, save_LAX_results
from Investigating_Losses import InvestigatingLosses
import warnings
warnings.filterwarnings('ignore')
# Get the current time
from datetime import datetime
current_time = datetime.now().strftime("%d-%B-%H-%M-%S")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def main(run_info:dict, run_for_DeepOPINN:bool, run_for_LAX:bool, finetuning_mode:bool, path_to_weights:str, run_name:str=None, data_path="data/Full"):
    if type(run_info['run_for_a_model_explicitly']) != bool:
        raise ValueError("run_for_a_model_explicitly must be either True or False")
        
    if run_info['run_for_a_model_explicitly']:
        if run_info['model_name'] not in ["PINN", "DeepONet", "PINNsFormer", "DONG", "DeepOPINN", "DOPFormer", "FormerPINN", "LAX", "TCN"]:
            raise ValueError("model_name must be one of these models, 'PINN', 'DeepONet', 'FormerPINN', 'PINNsFormer', 'DONG', 'DeepOPINN', 'DOPFormer', or 'LAX'")
    else:
        if run_info['model_name'] not in ["Bagging_u", "LAX_predictor", "DeepOLAX", "DOPDeepOLAX"]:
            raise ValueError("model_name must be one of these models, 'Bagging_u', 'LAX_predictor', 'DeepOLAX', or 'DOPDeepOLAX'") 
    
    if run_name is None:
        if run_info['run_for_a_model_explicitly']:
            if run_info['model_name'] == 'PINN':
                save_folder = f'results of reviewer/PINN_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'DeepONet':
                save_folder = f'results of reviewer/DeepONet_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'TCN':
                args = get_DeepONet_args()
                args.batch_size = args.batch_size_HUST
                save_folder = f'results of reviewer/TCN_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'PINNsFormer':
                save_folder = f'results of reviewer/PINNsFormer_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'DONG':
                save_folder = f'results of reviewer/DONG_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'DeepOPINN':
                save_folder = f'results of reviewer/DeepOPINN_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'DOPFormer':
                save_folder = f'results of reviewer/DOPFormer_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'FormerPINN':
                save_folder = f'results of reviewer/FormerPINN_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'TCN':
                save_folder = f'results of reviewer/TCN_{current_time}_for_HUST/HUST results'
            else:
                save_folder = f'results of reviewer/LAX_{current_time}_for_HUST/HUST results'
        else:
            save_folder = f'results of reviewer/Combination_{current_time}_for_HUST/HUST results'
    else:
        save_folder = f'results of reviewer/{run_name}/HUST results'

    if run_info['run_for_a_model_explicitly']:
        if run_info['model_name'] == 'PINN':
            args = get_PINN_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'DeepONet':
            args = get_DeepONet_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'PINNsFormer':
            args = get_PINNsFormer_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'DONG':
            args = get_DONG_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'FormerPINN':
            args = get_FormerPINN_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'DeepOPINN':
            args = get_DeepOPINN_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'DOPFormer':
            args = get_DOPFormer_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'TCN':
            args = get_DeepONet_args()
            args.batch_size = args.batch_size_HUST
        else:
            args = get_LAX_args()
    else:
        if run_info['model_name'] == 'Bagging_u': 
            args = get_Bagging_u_args()
        elif run_info['model_name'] == 'LAX_predictor':
            args = get_LAX_predictor_args()
        elif run_info['model_name'] == 'DOPDeepOLAX':
            args = get_DOPDeepOLAX_args()
        else: # 'DeepOLAX'
            args = get_DeepOLAX_args()
            
        setattr(args, 'run_for_DeepOPINN', run_for_DeepOPINN)
        setattr(args, 'run_for_LAX', run_for_LAX)
        args.batch_size = args.batch_size_HUST

    setattr(args, 'data', 'HUST')
    # batchs = ['default']
    batchs = ['default']
    n_batches = len(batchs)

    investigating_losses = InvestigatingLosses(
        dataset_name=args.data, 
        root_path=save_folder, 
        finetuning_mode=finetuning_mode,
        n_experiments=n_experiments,
        n_batches=n_batches
    )

    # To store individual batch results for LAX
    all_batch_mean_results = []

    # To store dataset batches for DeepONet, PINN, DeepOPINN, DOPFormer, and DOPLAX
    mse_vals_for_batches = []
    rmse_vals_for_batches = []
    
    for i in range(n_batches):
        batch = batchs[i]
        setattr(args, 'batch', batch)

        if run_info['run_for_a_model_explicitly'] == False:
            DOP_pth_file = os.path.join(path_to_weights, 'DeepOPINN', f'DeepOPINN_model_HUST_{batch}.pth')
            if run_info['model_name'] == 'DOPDeepOLAX':
                LAX_pth_file = os.path.join(path_to_weights, 'DeepOLAX', f'DeepOLAX_model_HUST_{batch}.pth')
                print('LAX_pth_file:', LAX_pth_file)
            else:
                LAX_pth_file = os.path.join(path_to_weights, 'LAX-phi', f'LAX_model_HUST_{batch}.pth')
            args.DOP_weights = DOP_pth_file 
            args.LAX_weights = LAX_pth_file

        # To store individual experiment results for LAX
        batch_experiment_results = []
        
        # To store batch experiments for DeepONet, PINN, DeepOPINN, DOPFormer, and DOPLAX
        mse_vals_for_experiments = []
        rmse_vals_for_experiments = []
        
        e_repeat = 0
        while e_repeat < n_experiments:
            save_folder_cp = save_folder
            save_folder = save_folder + "/" + str(i) + '-' + str(i) + '/Experiment' + str(e_repeat + 1)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            if run_info['run_for_a_model_explicitly'] and run_info['model_name'] == 'LAX':
                setattr(args, "results_path", save_folder)
                run_mode = 'LAX'
            
            if (run_info['run_for_a_model_explicitly'] == False) or (run_info['run_for_a_model_explicitly'] and run_info['model_name'] != 'LAX'):
                log_dir = 'logging.txt'
                setattr(args, "save_folder", save_folder)
                setattr(args, "log_dir", log_dir)
                run_mode = 'others'
                
            setattr(args, 'run_mode', run_mode)
            setattr(args, 'run_optuna', False)
            setattr(args, 'run_samll_sample', run_samll_sample)
            run_mode_backup = getattr(args, 'run_mode', None)
            tcn_args = get_DeepONet_args()
            setattr(tcn_args, 'batch_size', getattr(tcn_args, f'batch_size_{args.data}'))
            setattr(tcn_args, 'lr', getattr(tcn_args, f'lr_{args.data}', 0.001))
            setattr(tcn_args, 'batch', args.batch)
            setattr(tcn_args, 'data', args.data)
            setattr(tcn_args, 'run_mode', 'TCN')
            setattr(tcn_args, 'save_folder', save_folder)
            setattr(tcn_args, 'log_dir', 'logging.txt')
            dataloader_tcn = load_HUST_data(tcn_args, data_path=data_path)
            from Model.DD_nets import TCN
            import torch
            tcn_model = TCN.Model(tcn_args)
            tcn_model.Train(trainloader=dataloader_tcn['train'], validloader=dataloader_tcn['valid'], testloader=dataloader_tcn['test'])
            setattr(args, 'run_mode', run_mode_backup)

            
            dataloader = load_HUST_data(args, data_path=data_path)
            if run_info['model_name'] in ['DeepOPINN', 'Bagging_u', 'DOPDeepOLAX']:
                from utils.util import augment_dataloader_with_kpd
                dataloader = augment_dataloader_with_kpd(dataloader, tcn_model, device='cuda' if torch.cuda.is_available() else 'cpu')
            invalid_experiment = False

            if run_info['run_for_a_model_explicitly']:
                if run_info['model_name'] == 'PINN':
                    setattr(args, 'run_mode', 'PINN')
                    pinn = PINN.Model(args)
                    train_result, metric_values = pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                    if train_result == 'invalid':
                        rmse_value, mape_value = metric_values[0], metric_values[1]
                        info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                        file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                        write_to_file(file_path, info)
                        invalid_experiment = True

                elif run_info['model_name'] == 'FormerPINN':
                    setattr(args, 'run_mode', 'FormerPINN')
                    formerpinn = FormerPINN.Model(args)
                    train_result, metric_values = formerpinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                    if train_result == 'invalid':
                        rmse_value, mape_value = metric_values[0], metric_values[1]
                        info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                        file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                        write_to_file(file_path, info)
                        invalid_experiment = True

                elif run_info['model_name'] == 'DeepONet':
                    setattr(args, 'run_mode', 'DeepONet')
                    deeponet = DeepONet.Model(args)
                    train_result, metric_values = deeponet.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                    if train_result == 'invalid':
                        rmse_value, mape_value = metric_values[0], metric_values[1]
                        info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                        file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                        write_to_file(file_path, info)
                        invalid_experiment = True

                elif run_info['model_name'] == 'PINNsFormer':
                    setattr(args, 'run_mode', 'PINNsFormer')
                    pinnsformer = PINNsFormer.Model(args)
                    train_result, metric_values = pinnsformer.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                    if train_result == 'invalid':
                        rmse_value, mape_value = metric_values[0], metric_values[1]
                        info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                        file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                        write_to_file(file_path, info)
                        invalid_experiment = True
                        
                elif run_info['model_name'] == 'DONG':
                    setattr(args, 'run_mode', 'DONG')
                    dong = DONG.Model(args)
                    train_result, metric_values = dong.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                    if train_result == 'invalid':
                        rmse_value, mape_value = metric_values[0], metric_values[1]
                        info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                        file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                        write_to_file(file_path, info)
                        invalid_experiment = True
                        
                elif run_info['model_name'] == 'DeepOPINN':
                    setattr(args, 'run_mode', 'DeepOPINN')
                    deepopinn = DeepOPINN.Model(args)
                    train_result, metric_values = deepopinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                    if train_result == 'invalid':
                        rmse_value, mape_value = metric_values[0], metric_values[1]
                        info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                        file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                        write_to_file(file_path, info)
                        invalid_experiment = True 
                        
                elif run_info['model_name'] == 'DOPFormer':
                    setattr(args, 'run_mode', 'DOPFormer')
                    dopformer = DOPFormer.Model(args)
                    train_result, metric_values = dopformer.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                    if train_result == 'invalid':
                        rmse_value, mape_value = metric_values[0], metric_values[1]
                        info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                        file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                        write_to_file(file_path, info)
                        invalid_experiment = True

                elif run_info['model_name'] == 'TCN':
                    pass
                elif run_info['model_name'] == 'LAX':  # LAX
                    test_metrics = run_training_and_evaluation(args=args, dataloader=dataloader, batch_name=args.batch, experiment_id=e_repeat) 
                    experiment_data = {
                        'Batch': args.batch,
                        'Experiment': e_repeat + 1,
                        'MSE': test_metrics['mse'],
                        'RMSE': test_metrics['rmse'],
                        'MAPE': test_metrics['mape'],
                        'MAE': test_metrics['mae'],
                        'DUAL': test_metrics['dual']
                    }  
                    batch_experiment_results.append(experiment_data)
                    if test_metrics['rmse'] > 50000.0 or test_metrics['mape'] > 50000.0:
                        info = f"Experiment {e_repeat+1} invalid (RMSE={test_metrics['rmse']:.4f}, MAPE={test_metrics['mape']:.4f}). Retrying..."
                        file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                        write_to_file(file_path, info)
                        invalid_experiment = True

            else: # Combination nets
                if run_info['model_name'] == 'Bagging_u':
                    setattr(args, 'run_mode', 'Bagging_u')
                    Learner = Bagging_u
                elif run_info['model_name'] == 'LAX_predictor': 
                    setattr(args, 'run_mode', 'LAX_predictor')
                    Learner = LAX_predictor
                elif run_info['model_name'] == 'DOPDeepOLAX': 
                    setattr(args, 'run_mode', 'DOPDeepOLAX')
                    Learner = DOPDeepOLAX
                else: # "DeepOLAX"
                    setattr(args, 'run_mode', 'DeepOLAX')
                    Learner = DeepOLAX 
                    
                doplax = Learner.Model(args) 
                train_result, metric_values = doplax.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                if train_result == 'invalid':
                    rmse_value, mape_value = metric_values[0], metric_values[1]
                    info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                    file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                    write_to_file(file_path, info)
                    invalid_experiment = True
                    
            save_folder = save_folder_cp

            if invalid_experiment:
                # Do not increment e_repeat, retry the same experiment index
                continue
            
            e_repeat += 1
            if run_mode == 'LAX':
                log_file_path = os.path.join(args.results_path, "logging.txt")
            else:   
                log_file_path = os.path.join(args.save_folder, "logging.txt")
            mse_value, rmse_value = get_val_metrics_in_each_logfile(log_file_path=log_file_path)
            mse_vals_for_experiments.append(round(mse_value, 8))
            rmse_vals_for_experiments.append(round(rmse_value, 8))

        if run_mode == 'LAX':
            save_info = {
                'type': 'experiments',
                'experiments_info': batch_experiment_results,
                'batch': args.batch,
                'save_folder': args.results_path,
                'batches_info': all_batch_mean_results
            }
            all_batch_mean_results = save_LAX_results(save_info=save_info)
        else:
            mse_vals_for_batches.append(mse_vals_for_experiments)
            rmse_vals_for_batches.append(rmse_vals_for_experiments)
            
    if run_mode == 'LAX':
        save_info = {
            'type': 'batches',
            'experiments_info': None,
            'batch': args.batch,
            'save_folder': args.results_path,
            'batches_info': all_batch_mean_results
        }
        save_LAX_results(save_info=save_info)
    else:
        investigating_losses.forward() 
        
        val_loss_path = os.path.join(save_folder, f'{args.data}_validation_losses.xlsx')
        write_to_Excel(
            file_path=val_loss_path,
            dataset_name=args.data, 
            mse_vals_for_batches=mse_vals_for_batches, 
            rmse_vals_for_batches=rmse_vals_for_batches, 
            batches=batchs
        )
        print(f"Saved {args.data}_validation_losses.xlsx in the path {val_loss_path}")


def small_sample(run_info:bool, run_for_DeepOPINN:bool, run_for_LAX:bool,
                 finetuning_mode:bool, train_with_target_cells:bool,
                 path_to_weights:str, run_name:str=None, data_path="data/Full"):
    if type(run_info['run_for_a_model_explicitly']) != bool:
        raise ValueError("run_for_a_model_explicitly must be either True or False")
        
    if run_info['run_for_a_model_explicitly']:
        if run_info['model_name'] not in ["PINN", "DeepONet", "PINNsFormer", "DONG", "DeepOPINN", "DOPFormer", "LAX"]:
            raise ValueError("model_name must be one of these models, 'PINN', 'DeepONet', 'PINNsFormer', 'DONG', 'DeepOPINN', 'DOPFormer', or 'LAX'")
    else:
        if run_info['model_name'] not in ["Bagging_u", "LAX_predictor", "DeepOLAX", "DOPDeepOLAX"]:
            raise ValueError("model_name must be one of these models, 'Bagging_u', 'LAX_predictor', 'DeepOLAX', or 'DOPDeepOLAX'") 
    
    if run_name is None:
        if run_info['run_for_a_model_explicitly']:
            if run_info['model_name'] == 'PINN':
                save_folder = f'results of reviewer/PINN_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'DeepONet':
                save_folder = f'results of reviewer/DeepONet_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'PINNsFormer':
                save_folder = f'results of reviewer/PINNsFormer_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'DONG':
                save_folder = f'results of reviewer/DONG_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'DeepOPINN':
                save_folder = f'results of reviewer/DeepOPINN_{current_time}_for_HUST/HUST results'
            elif run_info['model_name'] == 'DOPFormer':
                save_folder = f'results of reviewer/DOPFormer_{current_time}_for_HUST/HUST results'
            else:
                save_folder = f'results of reviewer/LAX_{current_time}_for_HUST/HUST results'
        else:
            save_folder = f'results of reviewer/Combination_{current_time}_for_HUST/HUST results'
    else:
        save_folder = f'results of reviewer/{run_name}/HUST results'

    if run_info['run_for_a_model_explicitly']:
        if run_info['model_name'] == 'PINN':
            args = get_PINN_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'DeepONet':
            args = get_DeepONet_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'PINNsFormer':
            args = get_PINNsFormer_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'DONG':
            args = get_DONG_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'DeepOPINN':
            args = get_DeepOPINN_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'DOPFormer':
            args = get_DOPFormer_args()
            args.batch_size = args.batch_size_XJTU
        elif run_info['model_name'] == 'TCN':
            args = get_DeepONet_args()
            args.batch_size = args.batch_size_HUST
        else:
            args = get_LAX_args()
    else:
        if run_info['model_name'] == 'Bagging_u': 
            args = get_Bagging_u_args()
        elif run_info['model_name'] == 'LAX_predictor':
            args = get_LAX_predictor_args()
        elif run_info['model_name'] == 'DOPDeepOLAX':
            args = get_DOPDeepOLAX_args()
        else: # 'DeepOLAX'
            args = get_DeepOLAX_args()
            
        setattr(args, 'run_for_DeepOPINN', run_for_DeepOPINN)
        setattr(args, 'run_for_LAX', run_for_LAX)
        args.batch_size = args.batch_size_HUST

    setattr(args, 'data', 'HUST')

    # ---------------- Dataset selection ----------------
    if train_with_target_cells:
        num_of_batteries = [1, 2]
        batchs = ['default'] 
    else:
        num_of_batteries = [1, 2, 3, 4]
        batchs = ['default']
        
    n_batches = len(batchs)
        
    # ---------------- Main loops ----------------
    for num_battery in num_of_batteries:
        base_save_folder = save_folder + f' (small sample {num_battery})/'

        investigating_losses = InvestigatingLosses(
            dataset_name=args.data, 
            root_path=base_save_folder, 
            finetuning_mode=finetuning_mode,
            n_experiments=n_experiments,
            n_batches=n_batches
        )

        all_batch_mean_results = []   # For LAX
        mse_vals_for_batches = []     # For others
        rmse_vals_for_batches = []

        for i in range(n_batches):
            batch = batchs[i]
            setattr(args, 'batch', batch)

            if run_info['run_for_a_model_explicitly'] == False:
                DOP_pth_file = os.path.join(path_to_weights, 'DeepOPINN-Full', f'DeepOPINN_model_HUST_{batch}.pth')
                if run_info['model_name'] == 'DOPDeepOLAX':
                    LAX_pth_file = os.path.join(path_to_weights, 'DeepOLAX', f'DeepOLAX_model_HUST_{batch}.pth')
                    # print('LAX_pth_file:', LAX_pth_file)
                else:
                    LAX_pth_file = os.path.join(path_to_weights, 'LAX-phi', f'LAX_model_HUST_{batch}.pth')
                args.DOP_weights = DOP_pth_file 
                args.LAX_weights = LAX_pth_file
            
            batch_folder = os.path.join(base_save_folder, f"{i}-{i}")

            batch_experiment_results = []
            mse_vals_for_experiments = []
            rmse_vals_for_experiments = []

            e_repeat = 0
            while e_repeat < n_experiments:
                exp_folder = os.path.join(batch_folder, f"Experiment{e_repeat + 1}")
                if not os.path.exists(exp_folder):
                    os.makedirs(exp_folder)

                if run_info['run_for_a_model_explicitly'] and run_info['model_name'] == 'LAX':
                    setattr(args, "results_path", exp_folder)
                    run_mode = 'LAX'
                
                if (run_info['run_for_a_model_explicitly'] == False) or (run_info['run_for_a_model_explicitly'] and run_info['model_name'] != 'LAX'):
                    log_dir = 'logging.txt'
                    setattr(args, "save_folder", exp_folder)
                    setattr(args, "log_dir", log_dir)
                    run_mode = 'others'
                    
                setattr(args, 'run_mode', run_mode)
                setattr(args, 'run_optuna', False)
                setattr(args, 'run_samll_sample', run_samll_sample)
                run_mode_backup = getattr(args, 'run_mode', None)
                tcn_args = get_DeepONet_args()
                setattr(tcn_args, 'batch_size', getattr(tcn_args, f'batch_size_{args.data}'))
                setattr(tcn_args, 'lr', getattr(tcn_args, f'lr_{args.data}', 0.001))
                setattr(tcn_args, 'batch', args.batch)
                setattr(tcn_args, 'data', args.data)
                setattr(tcn_args, 'run_mode', 'TCN')
                setattr(tcn_args, 'save_folder', exp_folder)
                setattr(tcn_args, 'log_dir', 'logging.txt')
                dataloader_tcn = load_HUST_data(tcn_args, data_path=data_path)
                from Model.DD_nets import TCN
                import torch
                tcn_model = TCN.Model(tcn_args)
                tcn_model.Train(trainloader=dataloader_tcn['train'], validloader=dataloader_tcn['valid'], testloader=dataloader_tcn['test'])
                setattr(args, 'run_mode', run_mode_backup)

                
                dataloader = load_HUST_data(args, data_path=data_path)
                if run_info['model_name'] in ['DeepOPINN', 'Bagging_u', 'DOPDeepOLAX']:
                    from utils.util import augment_dataloader_with_kpd
                    dataloader = augment_dataloader_with_kpd(dataloader, tcn_model, device='cuda' if torch.cuda.is_available() else 'cpu')
                invalid_experiment = False

                # ---------------- Training by model ----------------
                if run_info['run_for_a_model_explicitly']:
                    if run_info['model_name'] == 'PINN':
                        setattr(args, 'run_mode', 'PINN')
                        pinn = PINN.Model(args)
                        train_result, metric_values = pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                        if train_result == 'invalid':
                            rmse_value, mape_value = metric_values[0], metric_values[1]
                            info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                            file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                            write_to_file(file_path, info)
                            invalid_experiment = True
    
                    elif run_info['model_name'] == 'DeepONet':
                        setattr(args, 'run_mode', 'DeepONet')
                        deeponet = DeepONet.Model(args)
                        train_result, metric_values = deeponet.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                        if train_result == 'invalid':
                            rmse_value, mape_value = metric_values[0], metric_values[1]
                            info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                            file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                            write_to_file(file_path, info)
                            invalid_experiment = True
    
                    elif run_info['model_name'] == 'PINNsFormer':
                        setattr(args, 'run_mode', 'PINNsFormer')
                        pinnsformer = PINNsFormer.Model(args)
                        train_result, metric_values = pinnsformer.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                        if train_result == 'invalid':
                            rmse_value, mape_value = metric_values[0], metric_values[1]
                            info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                            file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                            write_to_file(file_path, info)
                            invalid_experiment = True
                            
                    elif run_info['model_name'] == 'DONG':
                        setattr(args, 'run_mode', 'DONG')
                        dong = DONG.Model(args)
                        train_result, metric_values = dong.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                        if train_result == 'invalid':
                            rmse_value, mape_value = metric_values[0], metric_values[1]
                            info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                            file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                            write_to_file(file_path, info)
                            invalid_experiment = True
                            
                    elif run_info['model_name'] == 'DeepOPINN':
                        setattr(args, 'run_mode', 'DeepOPINN')
                        deepopinn = DeepOPINN.Model(args)
                        train_result, metric_values = deepopinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                        if train_result == 'invalid':
                            rmse_value, mape_value = metric_values[0], metric_values[1]
                            info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                            file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                            write_to_file(file_path, info)
                            invalid_experiment = True 
                            
                    elif run_info['model_name'] == 'DOPFormer':
                        setattr(args, 'run_mode', 'DOPFormer')
                        dopformer = DOPFormer.Model(args)
                        train_result, metric_values = dopformer.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                        if train_result == 'invalid':
                            rmse_value, mape_value = metric_values[0], metric_values[1]
                            info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                            file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                            write_to_file(file_path, info)
                            invalid_experiment = True
    
                    elif run_info['model_name'] == 'TCN':
                        pass
                    elif run_info['model_name'] == 'LAX':  # LAX
                        test_metrics = run_training_and_evaluation(args=args, dataloader=dataloader, batch_name=args.batch, experiment_id=e_repeat) 
                        experiment_data = {
                            'Batch': args.batch,
                            'Experiment': e_repeat + 1,
                            'MSE': test_metrics['mse'],
                            'RMSE': test_metrics['rmse'],
                            'MAPE': test_metrics['mape'],
                            'MAE': test_metrics['mae'],
                            'DUAL': test_metrics['dual']
                        }  
                        batch_experiment_results.append(experiment_data)
                        if test_metrics['rmse'] > 50000.0 or test_metrics['mape'] > 50000.0:
                            info = f"Experiment {e_repeat+1} invalid (RMSE={test_metrics['rmse']:.4f}, MAPE={test_metrics['mape']:.4f}). Retrying..."
                            file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                            write_to_file(file_path, info)
                            invalid_experiment = True
    
                else: # Combination nets
                    if run_info['model_name'] == 'Bagging_u':
                        setattr(args, 'run_mode', 'Bagging_u')
                        Learner = Bagging_u
                    elif run_info['model_name'] == 'LAX_predictor': 
                        setattr(args, 'run_mode', 'LAX_predictor')
                        Learner = LAX_predictor
                    elif run_info['model_name'] == 'DOPDeepOLAX': 
                        setattr(args, 'run_mode', 'DOPDeepOLAX')
                        Learner = DOPDeepOLAX
                    else: # "DeepOLAX"
                        setattr(args, 'run_mode', 'DeepOLAX')
                        Learner = DeepOLAX 
                        
                    doplax = Learner.Model(args) 
                    train_result, metric_values = doplax.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
                    if train_result == 'invalid':
                        rmse_value, mape_value = metric_values[0], metric_values[1]
                        info = f"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying..."
                        file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')
                        write_to_file(file_path, info)
                        invalid_experiment = True

                if False:
                    continue  # retry same experiment index

                # ---------------- Collect results ----------------
                e_repeat += 1
                log_file_path = os.path.join(exp_folder, "logging.txt") if run_mode != 'LAX' else os.path.join(args.results_path, "logging.txt")
                mse_value, rmse_value = get_val_metrics_in_each_logfile(log_file_path=log_file_path)
                mse_vals_for_experiments.append(round(mse_value, 8))
                rmse_vals_for_experiments.append(round(rmse_value, 8))

            # ---------------- Save batch results ----------------
            if run_mode == 'LAX':
                save_info = {
                    'type': 'experiments',
                    'experiments_info': batch_experiment_results,
                    'batch': args.batch,
                    'save_folder': batch_folder,
                    'batches_info': all_batch_mean_results
                }
                all_batch_mean_results = save_LAX_results(save_info=save_info)
            else:
                mse_vals_for_batches.append(mse_vals_for_experiments)
                rmse_vals_for_batches.append(rmse_vals_for_experiments)

        # ---------------- Save final results ----------------
        if run_mode == 'LAX':
            save_info = {
                'type': 'batches',
                'experiments_info': None,
                'batch': args.batch,
                'save_folder': base_save_folder,
                'batches_info': all_batch_mean_results
            }
            save_LAX_results(save_info=save_info)
        else:
            investigating_losses.forward() 
            val_loss_path = os.path.join(base_save_folder, f'{args.data}_validation_losses.xlsx')
            write_to_Excel(
                file_path=val_loss_path,
                dataset_name=args.data, 
                mse_vals_for_batches=mse_vals_for_batches, 
                rmse_vals_for_batches=rmse_vals_for_batches, 
                batches=batchs
            )
            print(f"Saved {args.data}_validation_losses.xlsx in the path {val_loss_path}")




if __name__ == '__main__':
    run_main = True # True, False
    run_optimization = False # True, False
    
    run_samll_sample = False # True, False
    train_with_target_cells = True # True, False

    # If you want to train the PINN, DeepOPINN, or LAX model explicitly, You should assign True to run_for_a_model_explicitly and the name of your desired model to model_name, otherwise False to run_for_a_model_explicitly
    run_info = {
        'run_for_a_model_explicitly': True, # True, False
        'model_name': 'FormerPINN' # DeepONet, "PINNsFormer", "DONG", "PINN", "DeepOPINN", "DOPFormer", "LAX", "Bagging_u", "LAX_predictor", "DeepOLAX", "DOPDeepOLAX", "FormerPINN"
    }
    # If you want to freeze the LAX model during the training of the DOPLAX module, You should assign False to run_for_LAX otherwise True
    run_for_LAX = False # True:unfreeze, False:freeze, None:random initial weights
    # If you want to freeze the DeepOPINN model during the training of the DOPLAX module, You should assign False to run_for_DeepOPINN otherwise True
    run_for_DeepOPINN = False # True:unfreeze, False:freeze, None:random initial weights
    
    finetuning_mode = False # True, False

    path_to_weights = 'pretrained_models' # str
    run_name = "FormerPINN" # DOP_unfreeze_LAX_freeze # both_freeze # str, None
    data_path = "data/Full" # "Full", "AK" (Average Kernel), "OFS" (Online Fourier Selective)
    n_experiments = 1

    
    if run_main:
        main(
            run_info=run_info,
            run_for_DeepOPINN=run_for_DeepOPINN,
            run_for_LAX=run_for_LAX,
            finetuning_mode=finetuning_mode,
            path_to_weights=path_to_weights, 
            run_name=run_name,
            data_path=data_path
        )
        run_samll_sample = False 
        run_optimization = False 

    if run_samll_sample:
        small_sample(
            run_info=run_info,
            run_for_DeepOPINN=run_for_DeepOPINN,
            run_for_LAX=run_for_LAX,
            finetuning_mode=finetuning_mode,
            train_with_target_cells=train_with_target_cells,
            path_to_weights=path_to_weights, 
            run_name=run_name,
            data_path=data_path
        )
        run_main = False
        run_optimization = False

    if run_optimization:
        resume = True # True, False
        num_of_trials = 40
        dataset_name = 'HUST' 
        batch_name = 'default'
        run_optuna(num_of_trials, run_info, dataset_name, batch_name, path_to_weights, resume=resume) 
        run_main = False
        run_samll_sample = False
    

