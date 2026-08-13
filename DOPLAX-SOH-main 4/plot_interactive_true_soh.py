import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
                        filename = parts[0]
                        if "_battery" in filename:
                            batch = filename.split('_battery')[0]
                        else:
                            batch = "default"
                        
                    if batch not in datasets_dict[dataset]:
                        datasets_dict[dataset].append(batch)
                        battery_files[dataset][batch] = []
                        
                    battery_files[dataset][batch].append(rel_path_no_ext.replace('\\', '/'))
                    
    return datasets_dict, battery_files

def get_nominal_capacity(dataset_name, bat_id):
    if dataset_name == 'XJTU':
        return 2.0
    elif dataset_name == 'MIT':
        return 1.1
    elif dataset_name == 'HUST':
        return 1.1
    elif dataset_name == 'TJU':
        if 'NCM_NCA' in bat_id:
            return 2.5
        elif 'NCA' in bat_id or 'NCM' in bat_id:
            return 3.5
        else:
            return 3.5
    return None

def main():
    project_root = os.path.abspath(os.path.dirname(__file__))
    data_root = os.path.join(project_root, 'data', 'Processed')
    
    output_dir = os.path.join(project_root, 'outputs', 'interactive_true_soh_plots')
    
    datasets, battery_files = discover_datasets_and_batches(data_root)
    
    for dataset, batches in datasets.items():
        dataset_dir = os.path.join(data_root, f"{dataset} data")
        
        for batch in batches:
            bat_ids = battery_files[dataset][batch]
            if not bat_ids:
                continue
                
            batch_output_dir = os.path.join(output_dir, dataset, batch)
            os.makedirs(batch_output_dir, exist_ok=True)
            
            for bat_id in bat_ids:
                file_path = os.path.join(dataset_dir, f"{bat_id}.csv")
                try:
                    df = pd.read_csv(file_path)
                    
                    df = df.replace([np.inf, -np.inf], np.nan)
                    df = df.dropna()
                    df = df.reset_index(drop=True)
                    
                    capacity = df.iloc[:, -1].values
                    nom_cap = get_nominal_capacity(dataset, bat_id)
                    if nom_cap is None:
                        nom_cap = capacity[0] if capacity[0] != 0 else 1e-6
                        
                    soh = capacity / nom_cap
                    cycles = np.arange(1, len(soh) + 1)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=cycles, y=soh, mode='lines', name='True SOH', line=dict(color='blue')))
                    
                    fig.update_layout(
                        title=f'True SOH for {dataset} - {batch} - {bat_id.split("/")[-1]}',
                        xaxis_title='Cycle Index',
                        yaxis_title='True SOH',
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    
                    safe_bat_id = bat_id.replace('/', '_').replace('\\', '_')
                    filename = f"{safe_bat_id}_TrueSOH.html"
                    
                    fig.write_html(os.path.join(batch_output_dir, filename))
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            
            print(f"Saved interactive plots for {dataset} - {batch} in {batch_output_dir}")

if __name__ == '__main__':
    main()
