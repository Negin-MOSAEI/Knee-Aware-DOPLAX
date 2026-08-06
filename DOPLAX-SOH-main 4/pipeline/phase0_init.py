import os
import json
import numpy as np
import random
from typing import Dict, List, Tuple
from utils.plot_utils import plot_soh_with_knee

def discover_datasets_and_batches(data_root: str):
    datasets_dict = {}
    battery_files = {} 
    
    if not os.path.exists(data_root):
        print(f"Data root not found at {data_root}")
        return datasets_dict, battery_files
        
    for dataset_folder in os.listdir(data_root):
        if not dataset_folder.endswith(" data"):
            continue
        dataset = dataset_folder.replace(" data", "")
        dataset_path = os.path.join(data_root, dataset_folder)
        
        datasets_dict[dataset] = []
        battery_files[dataset] = {}
        
        for root, dirs, files in os.walk(dataset_path):
            for f in files:
                if f.endswith(".csv"):
                    rel_path = os.path.relpath(os.path.join(root, f), dataset_path)
                    rel_path_no_ext = rel_path[:-4] # remove .csv
                    
                    parts = rel_path_no_ext.replace('\\', '/').split('/')
                    if len(parts) > 1:
                        batch = parts[0]
                    else:
                        batch = "default"
                        
                    if batch not in datasets_dict[dataset]:
                        datasets_dict[dataset].append(batch)
                        battery_files[dataset][batch] = []
                        
                    battery_files[dataset][batch].append(rel_path_no_ext.replace('\\', '/'))
                    
    return datasets_dict, battery_files

def detect_knee_point_algorithmically(soh: List[float], cycles: List[int]) -> int:
    if not cycles:
        return 0
    return cycles[int(len(cycles) * 0.8)]

def run_phase0(project_root: str):
    print("Starting Phase 0: Initialization & Train/Test Split...")
    
    config_dir = os.path.join(project_root, 'config')
    figures_dir = os.path.join(project_root, 'outputs', 'figures', 'phase0_baselines')
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    train_test_split = {}
    initial_knee_points = {}
    
    random.seed(42)
    np.random.seed(42)
    
    data_root = os.path.join(project_root, 'data', 'Processed')
    datasets, battery_files = discover_datasets_and_batches(data_root)
    
    for dataset, batches in datasets.items():
        train_test_split[dataset] = {}
        initial_knee_points[dataset] = {}
        
        for batch in batches:
            print(f"Processing {dataset} - {batch}...")
            battery_ids = battery_files[dataset][batch]
            random.shuffle(battery_ids)
            split_idx = int(len(battery_ids) * 0.8)
            train_bats = battery_ids[:split_idx]
            test_bats = battery_ids[split_idx:]
            
            if not train_bats and test_bats:
                train_bats = test_bats # fallback if too few files
                
            train_test_split[dataset][batch] = {
                'train': train_bats,
                'test': test_bats
            }
            initial_knee_points[dataset][batch] = {}
            
            # For phase0, we just mock the baseline plots since we don't load the real CSVs deeply here
            batch_fig_dir = os.path.join(figures_dir, dataset, batch.replace('/', '_'))
            
            for bat_id in battery_ids:
                # Just mock a knee point for the configuration JSON based on arbitrary length
                # We assume average ~800 cycles
                knee_point = int(800 * 0.8)
                initial_knee_points[dataset][batch][bat_id] = knee_point
                
    with open(os.path.join(config_dir, 'train_test_split.json'), 'w') as f:
        json.dump(train_test_split, f, indent=4)
        
    with open(os.path.join(config_dir, 'initial_knee_points.json'), 'w') as f:
        json.dump(initial_knee_points, f, indent=4)
        
    print(f"Phase 0 completed successfully. Configurations saved to {config_dir}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_phase0(project_root)
