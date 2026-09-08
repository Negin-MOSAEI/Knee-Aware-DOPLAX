import os
os.environ['DDE_BACKEND'] = 'pytorch'
import torch
torch.set_default_tensor_type(torch.FloatTensor)
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Add current dir to sys path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_HUST import main as main_hust
from main_MIT import main as main_mit
from main_TJU import main as main_tju
from main_XJTU import main as main_xjtu
from dataloader.data_helper import load_HUST_data, load_MIT_data, load_TJU_data, load_XJTU_data
from utils.arguments import get_DeepOPINN_args

from post_processing import postprocess_capacity, _mape

DATASETS = {
    'HUST': main_hust,
    'MIT': main_mit,
    'TJU': main_tju,
    'XJTU': main_xjtu
}

for main_func in DATASETS.values():
    main_func.__globals__['n_experiments'] = 1
    main_func.__globals__['run_samll_sample'] = False

MODELS_EXPLICIT = ['TCN', 'DeepOPINN', 'LAX']
MODELS_IMPLICIT = ['DeepOLAX', 'DOPDeepOLAX']

def run_pipeline():
    print("Starting pipeline...")
    
    # 1. Run the models to generate the raw predictions
    for dataset, main_func in DATASETS.items():
        print(f"--- Processing Dataset: {dataset} ---")
        
        BATCHES = {
            'HUST': ['default'],
            'MIT': ['2017-05-12', '2017-06-30', '2018-04-12'],
            'TJU': ['NCA', 'NCM', 'NCM_NCA'],
            'XJTU': ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
        }
        
        for model in MODELS_EXPLICIT:
            print(f"Training explicit model: {model}")
            run_info = {'run_for_a_model_explicitly': True, 'model_name': model}
            main_func(run_info, run_for_DeepOPINN=False, run_for_LAX=False, finetuning_mode=False, path_to_weights='pretrained_models', run_name=model, data_path='data/Processed')
            
            import shutil
            for i, batch in enumerate(BATCHES[dataset]):
                src = f"results of reviewer/{model}/{dataset} results/{i}-{i}/Experiment1/best_weights.pth"
                if not os.path.exists(src):
                    src = f"results of reviewer/{model}/{dataset} results/{i}-{i}/Experiment1/best_model.pth"
                if model == 'LAX':
                    dst_dir = 'pretrained_models/LAX-phi'
                    dst = f"pretrained_models/LAX-phi/LAX_model_{dataset}_{batch}.pth"
                else:
                    dst_dir = f"pretrained_models/{model}"
                    dst = f"pretrained_models/{model}/{model}_model_{dataset}_{batch}.pth"
                
                if os.path.exists(src):
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy(src, dst)
                    print(f"Copied {src} to {dst}")
                
        for model in MODELS_IMPLICIT:
            print(f"Training implicit model: {model}")
            run_info = {'run_for_a_model_explicitly': False, 'model_name': model}
            main_func(run_info, run_for_DeepOPINN=False, run_for_LAX=False, finetuning_mode=False, path_to_weights='pretrained_models', run_name=model, data_path='data/Processed')
            
            import shutil
            for i, batch in enumerate(BATCHES[dataset]):
                src = f"results of reviewer/{model}/{dataset} results/{i}-{i}/Experiment1/best_weights.pth"
                if not os.path.exists(src):
                    src = f"results of reviewer/{model}/{dataset} results/{i}-{i}/Experiment1/best_model.pth"
                dst_dir = f"pretrained_models/{model}"
                dst = f"pretrained_models/{model}/{model}_model_{dataset}_{batch}.pth"
                
                if os.path.exists(src):
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy(src, dst)
                    print(f"Copied {src} to {dst}")

    print("Pipeline run complete.")

if __name__ == '__main__':
    run_pipeline()


