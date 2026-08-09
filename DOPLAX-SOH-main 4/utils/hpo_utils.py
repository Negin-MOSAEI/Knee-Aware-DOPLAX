import os
import json

def get_hpo_config_path(project_root):
    return os.path.join(project_root, 'config', 'optimized_architectures.json')

def load_optimized_architectures(project_root):
    path = get_hpo_config_path(project_root)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def save_optimized_architecture(project_root, dataset, batch, model_name, best_params):
    path = get_hpo_config_path(project_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    config = load_optimized_architectures(project_root)
    
    if dataset not in config:
        config[dataset] = {}
    if batch not in config[dataset]:
        config[dataset][batch] = {}
        
    config[dataset][batch][model_name] = best_params
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def get_model_params(project_root, dataset, batch, model_name):
    config = load_optimized_architectures(project_root)
    try:
        return config[dataset][batch][model_name]
    except KeyError:
        return None
