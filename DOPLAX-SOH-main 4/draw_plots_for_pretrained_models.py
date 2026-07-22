from Model import PINN, DeepONet, DeepOPINN, DOPFormer, DOPLAX
from Model import LAX
from dataloader.data_helper import load_XJTU_data, load_TJU_data, load_MIT_data, load_HUST_data
from utils.arguments import get_DeepONet_args, get_PINN_args, get_DeepOPINN_args, get_DOPFormer_args, get_LAX_args, get_DOPLAX_args
from utils.util import save_differences_plot
import numpy as np
import os
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def main(dataset_name:str, model_name:str, raw_data:bool, save_labels:bool=False):
    dataset_batches = {
        'XJTU': ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite'],
        'TJU': ['NCA', 'NCM', 'NCM_NCA'],
        'MIT': ['one_batch'],
        'HUST': ['one_batch']
    }
    batches = dataset_batches[dataset_name]
    n_batches = len(batches)
    
    if model_name == 'DeepONet':
        args = get_DeepONet_args()
        setattr(args, 'run_mode', 'DeepONet')
        Learner = DeepONet
    elif model_name == 'PINN':
        args = get_PINN_args()
        setattr(args, 'run_mode', 'PINN')
        Learner = PINN
    elif model_name == 'DeepOPINN':
        args = get_DeepOPINN_args() 
        setattr(args, 'run_mode', 'DeepOPINN')
        Learner = DeepOPINN
    elif model_name == 'DOPFormer':
        args = get_DOPFormer_args() 
        setattr(args, 'run_mode', 'DOPFormer')
        Learner = DOPFormer
    elif 'DOPLAX' in model_name:
        args = get_DOPLAX_args()
        setattr(args, 'run_mode', model_name)
        Learner = DOPLAX
    else:
        args = get_LAX_args() 
        setattr(args, 'run_mode', 'LAX')
        Learner = LAX

    setattr(args, 'data', dataset_name)
    setattr(args, 'run_optuna', False)

    weights_path = f'pretrained_models/{model_name}'
    files = os.listdir(weights_path)
    save_folder = f'results of reviewer/plots_for_pretrained_models/{model_name}_plots'
    save_folder_cp = save_folder
    args.save_folder = save_folder + f'/{args.data} results'

    if raw_data:
        load_data_func = load_data
        args.main = True
        
    if args.data == 'XJTU':
        if raw_data == False:
            load_data_func = load_XJTU_data
        if model_name != 'LAX':
            args.batch_size = args.batch_size_XJTU
    elif args.data == 'TJU':
        if raw_data == False:
            load_data_func = load_TJU_data
        if model_name != 'LAX':
            args.batch_size = args.batch_size_TJU
    elif args.data == 'MIT':
        if raw_data == False:
            load_data_func = load_MIT_data    
        if model_name != 'LAX':
            args.batch_size = args.batch_size_MIT
    else:
        if raw_data == False:
            load_data_func = load_HUST_data
        if model_name != 'LAX':
            args.batch_size = args.batch_size_HUST
        
        
    for i in range(n_batches):
        batch = dataset_batches[args.data][i]
        setattr(args, 'batch', batch)
        save_folder_tmp = args.save_folder
        args.save_folder += f'/Batch-{i+1}'
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        
        file_name = f'{model_name}_model_{args.data}_{i}' if args.data in ['XJTU', 'TJU'] else f'{model_name}_model_{args.data}'
        status = ['exist' if f'{file_name}.pth' in file else 'not exist' for file in files]
        if status[0] == 'not exist':
            file_name = f'{model_name}_model_{args.data}_{batch}' if args.data in ['XJTU', 'TJU'] else f'{model_name}_model_{args.data}'
        # print('files:', files)
        # print('status:', status)
        # print('file_name:', file_name)
        
        pth_path = os.path.join(weights_path, f'{file_name}.pth')
        setattr(args, 'weight_path', pth_path)
        
        dataloader = load_data_func(args)
        trainloader = dataloader['train']
        
        # Create CUDA generator for train and valid DataLoaders
        generator = torch.Generator(device=device)
        if hasattr(trainloader, 'sampler') and hasattr(trainloader.sampler, 'generator'):
            trainloader.sampler.generator = generator
            
        if model_name == 'LAX':
            x_dim, y_dim = 16, 16 
            
            # specifying the stats. 
            sum_ = 0.0
            sum_squared = 0.0
            n_samples = 0
        
            for x1, x2, _, _ in trainloader:
                x1, x2 = x1.to(device), x2.to(device)
                x2 = x2[:, :16]
                x1 = x1[:, :16]
                sum_ += x2.sum(dim=0)  # sum across the batch (dim=0), result shape: [num_features]
                sum_squared += (x2 ** 2).sum(dim=0)
                n_samples += x2.size(0)  # batch_size
            X_mean = sum_ / n_samples
            X_std = torch.sqrt((sum_squared / n_samples) - (X_mean ** 2))

            if args.data == 'XJTU':
                args.betha_LAX = args.betha_LAX_XJTU
                args.dual_LAX = args.dual_LAX_XJTU
                args.theta_LAX = args.theta_LAX_XJTU
                args.lr_net = args.lr_net_LAX_XJTU
                args.h_dim_LAX = args.h_dim_LAX_XJTU
            elif args.data == 'TJU':
                args.betha_LAX = args.betha_LAX_TJU
                args.dual_LAX = args.dual_LAX_TJU
                args.theta_LAX = args.theta_LAX_TJU
                args.lr_net = args.lr_net_LAX_TJU
                args.h_dim_LAX = args.h_dim_LAX_TJU
            elif args.data == 'MIT':
                args.betha_LAX = args.betha_LAX_MIT
                args.dual_LAX = args.dual_LAX_MIT
                args.theta_LAX = args.theta_LAX_MIT
                args.lr_net = args.lr_net_LAX_MIT
                args.h_dim_LAX = args.h_dim_LAX_MIT
            else:
                args.betha_LAX = args.betha_LAX_HUST
                args.dual_LAX = args.dual_LAX_HUST
                args.theta_LAX = args.theta_LAX_HUST
                args.lr_net = args.lr_net_LAX_HUST
                args.h_dim_LAX = args.h_dim_LAX_HUST

            # print('args:', args)
            # print('epoch_y_LAX:', args.epoch_y_LAX)
            model = Learner.OptimizationNetwork(
                x_sts=(X_mean, X_std),
                y_dim=y_dim,
                x_dim=x_dim,
                args= args
            ).to(device)
            Learner.load_model(
                model=model, 
                model_path=pth_path, 
                freeze=True, 
                batch_size=args.batch_size
            )
            true_label, pred_label = Learner.evaluate_on_test_set(
                model=model,
                test_dataloader=dataloader['test'],
                args=args,
                return_labels=True
            )
        else:
            learner = Learner.Model(args).to(device)
            learner.load_model(model_path=pth_path, freeze=True)   
            true_label, pred_label = learner.Test(testloader=dataloader['test'])
            
        if args.data in ['XJTU', 'TJU']:
            batch_num = i
        else:
            batch_num = "one_batch"  

        y_true_path = os.path.join(args.save_folder, 'true_label.npy')
        y_pred_path = os.path.join(args.save_folder, 'pred_label.npy')
        
        if save_labels:
            np.save(y_true_path, true_label)
            np.save(y_pred_path, pred_label)

        plot_info = {
            'y_true_path' : y_true_path,
            'y_pred_path' : y_pred_path,
            'y_true' : true_label,
            'y_pred' : pred_label,
            'dataset_name' : args.data,
            'batch_num': batch_num,
            'model_name': model_name
        }
        save_differences_plot(plot_info=plot_info)
        
        if model_name != 'LAX':
            learner._clear_logger()
            
        args.save_folder = save_folder_tmp
        
    save_folder = save_folder_cp
        


if __name__ == '__main__':
    process_all_datasets = True # True, False
    model_name = "DOPFormer" # "DeepONet", "PINN", "DeepOPINN", "DOPFormer", "LAX"
    dataset_name = 'XJTU' # 'XJTU', 'TJU', 'MIT', 'HUST'
    datasets = ['XJTU', 'TJU', 'MIT', 'HUST']
    save_labels = True # True, False
    raw_data = False # True, False
    
    if process_all_datasets:
       for dataset in datasets:
           main(dataset_name=dataset, model_name=model_name, raw_data=raw_data, save_labels=save_labels)
    else:
        main(dataset_name=dataset_name, model_name=model_name, raw_data=raw_data, save_labels=save_labels)
