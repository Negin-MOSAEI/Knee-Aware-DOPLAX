from sklearn import metrics
import numpy as np
import logging
import torch
import json
import os
from torch.utils.data import DataLoader
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



class NoIter1Filter(logging.Filter):
    def filter(self, record):
        return "iter:1" not in record.getMessage()

        
def get_logger(log_name='log.txt'):
    """
    Create a logger object that outputs to both console and a file.
    Training progress (INFO level) goes to both console and file
    Debug messages (DEBUG level) only go to the log file
    """
    logger = logging.getLogger('mylogger')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - function:%(funcName)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    console.addFilter(NoIter1Filter())
    logger.addHandler(console)

    if log_name is not None:
        handler = logging.FileHandler(log_name)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def write_to_file(file_path, info):
    with open(file_path,'a') as f:
        f.write(info)
        f.write('\n')


def write_to_json(file_path, info):
    """Properly writes JSON data (overwrites existing file)"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(info, f, indent=4)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def write_to_Excel(file_path:str, dataset_name:str, mse_vals_for_batches:list, rmse_vals_for_batches:list, batches:list=None):
    with ExcelWriter(file_path) as writer:
        if dataset_name in ['XJTU', 'TJU']:
            for i, batch in enumerate(batches):
                df = pd.DataFrame(data={
                    'MSE' : mse_vals_for_batches[i],
                    'RMSE' : rmse_vals_for_batches[i]
                })
                df.index = [f'Experiment {exp_i+1}' for exp_i, _ in enumerate(mse_vals_for_batches[i])] 
                df.to_excel(writer, sheet_name=f'{i}-{i}')
        else:
            df = pd.DataFrame(data={
                'MSE' : mse_vals_for_batches[0],
                'RMSE' : rmse_vals_for_batches[0]
            })
            df.index = [f'Experiment {exp_i+1}' for exp_i, _ in enumerate(mse_vals_for_batches[0])] 
            df.to_excel(writer, sheet_name=f'one_batch')

            
def get_datasets_info():
    dataset_info_path = "/RUL Group/PINN4SOH/utils/datasets_info.json"
    if not os.path.exists(dataset_info_path):
        raise FileNotFoundError(f'Not found {os.path.basename(dataset_info_path)} in {dataset_info_path}')
    return load_json(dataset_info_path)


def get_max_matrix(matrix_list:list):
    """
    Get a matrix with the maximum number of columns in the input list.
    """
    max_cols = matrix_list[0].shape[1]
    max_value = matrix_list[0]
    for matrix in matrix_list:
        if matrix.shape[1] > max_cols:
            max_cols = matrix.shape[1]
            max_value = matrix

    return max_cols, max_value


def match_dimensions(matrix_list:list):
    """
    Match the number of columns in each matrix with the largest matrix of max_list.
    """
    new_matrix_list = list()
    max_cols, max_matrix = get_max_matrix(matrix_list)
    for matrix in matrix_list:
        if matrix.shape[1] == max_cols:
            new_matrix_list.append(matrix)
        else:
            rows = matrix.shape[0]
            columns = max_cols - matrix.shape[1]
            dummy_matrix = np.zeros(shape=(rows, columns))
            matched_matrix = np.concatenate([matrix, dummy_matrix], axis=1)
            new_matrix_list.append(matched_matrix)

    return new_matrix_list


def get_cycle_values(X:torch.Tensor, cycle_i:int, num_cycle_variables=int):
    all_in_one = []
    cycle_i_values = X[cycle_i:cycle_i + num_cycle_variables, :]
    cycle_i_values = list(zip(*cycle_i_values))
    for value in cycle_i_values:
        all_in_one.extend(all_in_one)

    return all_in_one


def pad_or_trim(tensor, target_len):
    """Pad or trim 1D or 2D tensors using linear interpolation
    Args:
        tensor: Input tensor of shape [L] or [B, L]
        target_len: Desired output length
    Returns:
        Tensor of shape [target_len] or [B, target_len]
    """
    # Handle 1D case by adding batch dimension
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # [1, L]
        was_1d = True
    elif tensor.dim() == 2:
        was_1d = False
        dim = 1
        metadata = tensor[:, -3:]
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {tensor.shape}")
    
    B, L = tensor.shape
    
    if L < target_len:  # Pad via interpolation
        # Reshape to [B, 1, L] for interpolation
        tensor = tensor.unsqueeze(1)  # [B, 1, L]
        
        # Interpolate along last dimension
        padded = F.interpolate(
            tensor.float(),
            size=target_len,
            mode='linear',
            align_corners=True
        )
        result = padded.squeeze(1)  # [B, target_len]
    
    elif L > target_len:  # Trim
        result = tensor[..., :target_len]
    else:  # Unchanged
        result = tensor
    
    # Remove batch dimension if input was 1D
    return result.squeeze(0) if was_1d else torch.concat([result, metadata], dim=dim)


def collate_fn(batch, current_batch, main, path='/RUL Group/PINN4SOH/results of reviewer/Feature_Extractor', dataset_name=None): 
    # Calculate unique lengths and max_len if not provided
    original_lengths = [x1.shape[-1] for x1, _, _, _ in batch]
    unique_lengths = dict.fromkeys(original_lengths)

    for x1, x2, y1, y2 in batch:
        length = len(x1)
        if unique_lengths[length] is None:
            unique_lengths[length] = current_batch

    # Save info 
    data = get_datasets_info()
    if dataset_name != 'All_datasets':
        data = {dataset_name: data[dataset_name]}
    
    unique_lengths = {v: k for k, v in unique_lengths.items()}
    info = {
        'dataset_info': data,
        'data_loader': {
            'unique_length': unique_lengths
        }
    }
    if main:  
        file_path = os.path.join(path, f'{current_batch}_train_info_for_feature_extractor.json')
    else:
        file_path = os.path.join(path, dataset_name, f'{current_batch}_train_info_for_feature_extractor.json')
    if not os.path.exists(file_path):
        print(f"Feature extractor info for {current_batch}:\n{info}")
        print('*' * 45)
        write_to_json(file_path, info)
    
    return torch.utils.data.default_collate(batch)


def match_dtype_with_dataset_dtype(split_mode, dtypes):
    if split_mode == None:
        dtypes = dtypes
    else:
        prefix_part = np.array([f"{split_mode.lower()}_"] * len(dtypes))
        main_part = np.array(dtypes)
        if (len(dtypes[:-1]) == 1) or (len(dtypes[:-1]) == 2):
            dtypes = np.char.add(prefix_part, main_part)
    return dtypes
    


def augment_dataloader_with_kpd(dataloader, tcn_model, device='cuda'):
    from torch.utils.data import TensorDataset, DataLoader
    for split in ['train', 'valid', 'test', 'train_3', 'test_3']:
        if split in dataloader:
            loader = dataloader[split]
            dataset = loader.dataset
            X1, X2, Y1, Y2 = dataset.tensors
            with torch.no_grad():
                KPD1 = tcn_model.predict(X1.unsqueeze(0).to(device)).cpu().squeeze(0)
                KPD2 = tcn_model.predict(X2.unsqueeze(0).to(device)).cpu().squeeze(0)
            new_dataset = TensorDataset(X1, X2, Y1, Y2, KPD1, KPD2)
            dataloader[split] = DataLoader(new_dataset, batch_size=loader.batch_size, shuffle=isinstance(loader.sampler, torch.utils.data.RandomSampler), drop_last=loader.drop_last, generator=None)
    return dataloader

