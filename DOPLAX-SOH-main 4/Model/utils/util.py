from sklearn import metrics
import numpy as np
import logging
import torch
import json
import os
from torch.utils.data import DataLoader
from Model.utils.losses import mae, mse, mape, rmse
import torch.nn.functional as F
import statistics as stats
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from natsort import natsorted
import math
from pandas import ExcelWriter
import regex as re
import pandas as pd
from utils.util import write_to_Excel



def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def eval_metrix(true_label, pred_label):
    MAE = mae(true_label,pred_label)
    MAPE = mape(true_label,pred_label)
    MSE = mse(true_label,pred_label)
    RMSE = rmse(true_label,pred_label)
    
    return [MAE,MAPE,MSE,RMSE]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_feature_extractor_training_info(current_batch, dataset_name=None):
    if dataset_name == 'None':  
        file_path = os.path.join('/RUL Group/PINN4SOH', 'results of reviewer', 'Feature_Extractor', f'{current_batch}_train_info_for_feature_extractor.json')
    else:
        file_path = os.path.join('/RUL Group/PINN4SOH', 'results of reviewer', 'Feature_Extractor', dataset_name, f'{current_batch}_train_info_for_feature_extractor.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError('First run for_feature_extractor with process_batches=True, then run this function again :)')
    return load_json(file_path)


def find_input_dim_from_checkpoint(checkpoint_path):
    """
    Extracts the input dimension from a checkpoint file contains L and t.
    Assumes the first linear layer in the branch network contains the input dimension.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    branch_weights = [k for k in checkpoint['feature_extractor'].keys() if 'branch' in k]
    
    if branch_weights:
        first_branch_weight = next(k for k in branch_weights if 'weight' in k)
        m = checkpoint['feature_extractor'][first_branch_weight].shape[1]
        return m
    else:
        raise ValueError("No branch network weights found in the checkpoint.")


def save_differences_plot(plot_info:dict):
    if plot_info is not None:
        y_true_path = plot_info['y_true_path']
        y_pred_path = plot_info['y_pred_path']
        y_true = plot_info['y_true']
        y_pred = plot_info['y_pred']
        dataset_name = plot_info['dataset_name']
        batch_num = plot_info['batch_num']
        model_name = plot_info['model_name']
        
    if (y_true_path is None) and (y_pred_path is None):
        raise ValueError("It must be assigned an special path to variables, y_true_path and y_pred_path")
        
    if (y_true is None) and (y_pred is None) and (y_true_path is not None) and (y_pred_path is not None):
        y_true = np.load(y_true_path) 
        y_pred = np.load(y_pred_path) 

    if type(batch_num) == int:
        batch_num += 1 

    if len(y_true[0]) == 1:
        cycle_ids = [1]
    else:
        cycles_num = y_true.shape[0]
        cycle_ids = np.random.choice(cycles_num, 25, replace=False)
        info = {
            'dataset_name': dataset_name,
            'batch_num': batch_num,
            'selected_cycles': cycle_ids.tolist(),
        }
        cycles_root = os.path.join(os.path.dirname(y_true_path), 'plots')
        if not os.path.exists(cycles_root):
            os.mkdir(cycles_root)

        if type(batch_num) == int:
            cycles_path = os.path.join("results of reviewer", "plots_for_pretrained_models", "selected_cycles", dataset_name, f'selected_cycles_info_for_{dataset_name}_batch{batch_num}.json')
        else:
            cycles_path = os.path.join("results of reviewer", "plots_for_pretrained_models", "selected_cycles", dataset_name, f'selected_cycles_info_for_{dataset_name}.json')
        if not os.path.exists(cycles_path):
            write_to_json(cycles_path, info)
        else:
            cycle_ids = load_json(cycles_path)['selected_cycles']

    for i, cycle_id in enumerate(cycle_ids):
        plt.figure(figsize=(10, 5))

        if y_true.shape[1] == 1:
            y_true_cp = y_true.copy()
            y_true_cp = y_pred.copy()
            y_true_cp = y_true_cp.reshape(-1)
            y_true_cp = y_true_cp.reshape(-1)
        else:
            y_true_cp = y_true

        if len(y_true_cp.shape) == 1:
            actuals = y_true
            preds = y_pred
            title = f"SOH Degradation trend for the Cycles of the Lithium-ion Batteries - (Model:{model_name}-Dataset:{dataset_name}-Batch:{batch_num})"
            x_label = "Cycle Index"
            if batch_num == 'one_batch':
                file_name = f"{model_name}-differences_plot_{dataset_name}"
            else:
                file_name = f"{model_name}-differences_plot_{dataset_name}_batch-{batch_num}"

        elif len(y_true_cp.shape) == 2: 
            actuals = y_true[cycle_id, :]
            preds = y_pred[cycle_id, :]
            title = f"SOH Degradation trend for a Cycle of a Lithium-ion Battery - (Model:{model_name}-Dataset:{dataset_name}-Batch:{batch_num}-Cycle:{cycle_id})"
            x_label = "Time"
            if batch_num == 'one_batch':
                file_name = f"{model_name}-differences_plot_{dataset_name}_cycle-{cycle_id}"
            else:
                file_name = f"{model_name}-differences_plot_{dataset_name}_batch-{batch_num}_cycle-{cycle_id}"

        plt.plot(actuals, label='Actual SOH', alpha=.5, color='blue')
        plt.plot(preds, label='Predicted SOH', alpha=.5, color='red')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel('SOH Value')
        plt.legend()
        plt.grid()

        if len(y_true_cp.shape) == 1:
            output_path = os.path.join(y_true_path.split('true_label.npy')[0], file_name)
        elif len(y_true_cp.shape) == 2: 
            output_path_root = os.path.join(y_true_path.split('true_label.npy')[0], 'plots')
            output_path = os.path.join(output_path_root, file_name)

        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        
        if len(y_true_cp.shape) == 1:
            print(f'Saved sucessfully (iteration:{i+1}): {output_path}')
        elif len(y_true_cp.shape) == 2: 
            print(f'Saved sucessfully (iteration:{i+1} - cycle:{cycle_id}): {output_path}')


def get_val_metrics_in_each_logfile(log_file_path):
    with open(log_file_path, 'r') as f:
        text = f.readlines()
        text = ''.join(text)
        
    best_epoch_match = re.search(r"The best model created at epoch.*", text)
    if best_epoch_match:
        best_epoch_number = best_epoch_match.group(0)
        best_epoch_number = int(re.findall(r"The best model created at epoch (\d+)", best_epoch_number)[0])
        best_epoch_line = re.search(r"\[Valid\] epoch:{}.*".format(best_epoch_number), text).group(0)
    else:
        valid_lines = re.findall(r"\[Valid\] epoch:\d+.*", text)
        if valid_lines:
            best_epoch_line = valid_lines[-1]
        else:
            return 0.0, 0.0
            
    mse_val = float(re.findall(r"MSE: (\d+\.\d+e[+-]\d+|\d+\.\d+)", best_epoch_line)[0])
    rmse_val = math.sqrt(mse_val)
    return mse_val, rmse_val

    
def extract_val_metrics_from_logfiles(root_path:str, n_batches:int=None):
    try:
        folders = os.listdir(root_path)
    except:
        raise FileNotFoundError(f"The directory {root_path} does not exist or is not accessible.")

    datasets = {
        "XJTU": [f'{i}-{i}' for i in range(6)] if n_batches is None else [f'{i}-{i}' for i in range(n_batches)],
        "TJU": [f'{i}-{i}' for i in range(3)] if n_batches is None else [f'{i}-{i}' for i in range(n_batches)],
        "HUST": None,
        "MIT": None
    }
    batchs = {
        "XJTU" : ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite'],
        "TJU" : ["NCA", "NCM", "NCM_NCA"],
        "MIT" : None, 
        "HUST" : None
    }
    
    final_results = []
    metric_vals_for_datasets = {}
    for folder in folders:
        if folder.endswith('.csv') or folder.endswith('.xlsx') or folder.endswith('.ipynb_checkpoints') or folder.endswith('.tar.xz'):
            continue
            
        dataset_name = folder.split(' results')[0] if 'results' in folder else dataset_name
        dataset_path= os.path.join(root_path, folder)
        
        if dataset_name in ['XJTU', 'TJU']:
            dataset_batches = datasets[dataset_name]
            batches = natsorted(os.listdir(dataset_path))

            mse_vals_for_batches = []
            rmse_vals_for_batches = []
            for batch in batches:
                if batch.endswith('.csv') or batch.endswith('.xlsx') or batch.endswith('.ipynb_checkpoints') or batch.endswith('.tar.xz'):
                    continue

                batch_path = os.path.join(dataset_path, batch)
                experiments = os.listdir(batch_path)
                
                mse_vals_for_experiments = []
                rmse_vals_for_experiments = []
                for exp in experiments:
                    if exp.endswith('.csv') or exp.endswith('.xlsx') or exp.endswith('.ipynb_checkpoints') or exp.endswith('.tar.xz') or exp.endswith('.txt'):
                        continue

                    log_file_path = os.path.join(batch_path, exp, 'logging.txt')
                    mse_value, rmse_value = get_val_metrics_in_each_logfile(log_file_path)
                    mse_vals_for_experiments.append(round(mse_value, 8))
                    rmse_vals_for_experiments.append(round(rmse_value, 8))

                mse_vals_for_batches.append(mse_vals_for_experiments)
                rmse_vals_for_batches.append(rmse_vals_for_experiments)

            metric_vals_for_datasets[dataset_name] = (mse_vals_for_batches, rmse_vals_for_batches)
            
        elif dataset_name in ['MIT', 'HUST']:
            dataset_batches = datasets[dataset_name]
            experiments = os.listdir(dataset_path)
            mse_vals_for_batches, rmse_vals_for_batches = [], []
            mse_vals_for_experiments, rmse_vals_for_experiments = [], []
            for exp in experiments:
                if exp.endswith('.csv') or exp.endswith('.xlsx') or exp.endswith('.ipynb_checkpoints') or exp.endswith('.tar.xz') or exp.endswith('.txt'):
                    continue

                log_file_path = os.path.join(dataset_path, exp, 'logging.txt')
                mse_value, rmse_value = get_val_metrics_in_each_logfile(log_file_path)
                mse_vals_for_experiments.append(round(mse_value, 8))
                rmse_vals_for_experiments.append(round(rmse_value, 8))

            mse_vals_for_batches.append(mse_vals_for_experiments)
            rmse_vals_for_batches.append(rmse_vals_for_experiments)
            
            metric_vals_for_datasets[dataset_name] = (mse_vals_for_batches, rmse_vals_for_batches)
            
        file_path = os.path.join(dataset_path, f'{dataset_name}_validation_losses.xlsx')
        write_to_Excel(
            file_path=file_path,
            dataset_name=dataset_name, 
            mse_vals_for_batches=metric_vals_for_datasets[dataset_name][0], 
            rmse_vals_for_batches=metric_vals_for_datasets[dataset_name][1], 
            batches=batchs[dataset_name]
        )
        print(f"Saved the Excel file {dataset_name}_validation_losses.xlsx in the path {dataset_path}")
