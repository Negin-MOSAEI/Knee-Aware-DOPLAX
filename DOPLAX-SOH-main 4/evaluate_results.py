import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from post_processing import postprocess_capacity

def _mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

def get_kpis(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = _mape(y_true, y_pred)
    return mse, rmse, mae, mape

def load_splits(dataset_name, batch_name):
    with open('config/train_test_split.json', 'r') as f:
        splits = json.load(f)
    if dataset_name == 'TJU':
        tju_map = {'NCA': 'Dataset_1_NCA_battery', 'NCM': 'Dataset_2_NCM_battery', 'NCM_NCA': 'Dataset_3_NCM_NCA_battery'}
        batch_name = tju_map.get(batch_name, batch_name)
    elif dataset_name == 'XJTU':
        batch_name = 'Sim_satellite' if batch_name == 'satellite' else batch_name
        
    return splits[dataset_name][batch_name]

def get_test_lengths(dataset, batch):
    splits = load_splits(dataset, batch)
    test_batteries = splits['test']
    
    lengths = []
    for b in test_batteries:
        path = f"data/Processed/{dataset} data/{b}.csv"
        df = pd.read_csv(path)
        lengths.append(len(df))
    return test_batteries, lengths

def find_latest_experiment_dir(dataset, batch_index, model):
    if model in ['DOPDeepOLAX', 'Bagging_u', 'DeepOLAX', 'LAX_predictor']:
        patterns = [
            f"results of reviewer/{model}/{dataset} results/{batch_index}-{batch_index}/Experiment1",
            f"results of reviewer/Combination_*_for_{dataset}/{dataset} results/{batch_index}-{batch_index}/Experiment1"
        ]
    else:
        patterns = [
            f"results of reviewer/{model}/{dataset} results/{batch_index}-{batch_index}/Experiment1",
            f"results of reviewer/{model}_*_for_{dataset}/{dataset} results/{batch_index}-{batch_index}/Experiment1"
        ]
    
    dirs = []
    for pattern in patterns:
        dirs.extend(glob.glob(pattern))
        
    if not dirs:
        return None
    dirs.sort(key=os.path.getctime, reverse=True)
    return dirs[0]

def main():
    datasets = ['HUST', 'MIT', 'TJU', 'XJTU']
    
    dataset_batches = {
        'HUST': ['default'],
        'MIT': ['2017-05-12', '2017-06-30', '2018-04-12'],
        'TJU': ['NCA', 'NCM', 'NCM_NCA'],
        'XJTU': ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    }
    
    models = ['TCN', 'DeepOPINN', 'LAX', 'DOPDeepOLAX']
    
    battery_kpis = []
    batch_kpis = []
    
    os.makedirs('EvaluationPlots', exist_ok=True)
    
    for dataset in datasets:
        batches = dataset_batches[dataset]
        for b_idx, batch in enumerate(batches):
            print(f"Evaluating {dataset} - {batch}")
            
            try:
                test_batteries, lengths = get_test_lengths(dataset, batch)
            except Exception as e:
                print(f"Error getting lengths for {dataset} - {batch}: {e}")
                continue
                
            model_preds = {}
            for model in models:
                exp_dir = find_latest_experiment_dir(dataset, b_idx, model)
                if exp_dir is None:
                    print(f"Could not find results for {model} on {dataset} - {batch}")
                    continue
                
                true_path = os.path.join(exp_dir, 'true_label.npy')
                pred_path = os.path.join(exp_dir, 'pred_label.npy')
                
                if not os.path.exists(true_path) or not os.path.exists(pred_path):
                    print(f"Missing npy files for {model} on {dataset} - {batch}")
                    continue
                
                y_true = np.load(true_path).reshape(-1)
                y_pred = np.load(pred_path).reshape(-1)
                
                model_preds[model] = {'y_true': y_true, 'y_pred': y_pred}
                
                mse, rmse, mae, mape = get_kpis(y_true, y_pred)
                batch_kpis.append({
                    'Dataset': dataset, 'Batch': batch, 'Model': model,
                    'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape
                })
                
                if model == 'DOPDeepOLAX':
                    y_pred_pp = postprocess_capacity(y_pred)
                    mse, rmse, mae, mape = get_kpis(y_true, y_pred_pp)
                    batch_kpis.append({
                        'Dataset': dataset, 'Batch': batch, 'Model': 'KADOPLAX (PostProcessed)',
                        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape
                    })
                    model_preds['KADOPLAX (PostProcessed)'] = {'y_true': y_true, 'y_pred': y_pred_pp}
            
            start_idx = 0
            for bat_idx, length in enumerate(lengths):
                battery_name = test_batteries[bat_idx].replace('/', '_').replace('\\', '_')
                end_idx = start_idx + length
                
                if 'TCN' in model_preds:
                    y_true_tcn = model_preds['TCN']['y_true'][start_idx:end_idx]
                    y_pred_tcn = model_preds['TCN']['y_pred'][start_idx:end_idx]
                    
                    if len(y_true_tcn) > 0:
                        mse, rmse, mae, mape = get_kpis(y_true_tcn, y_pred_tcn)
                        battery_kpis.append({
                            'Dataset': dataset, 'Batch': batch, 'Battery': battery_name, 'Model': 'TCN',
                            'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape
                        })
                        
                        plt.figure()
                        plt.plot(y_true_tcn, label='True KPD')
                        plt.plot(y_pred_tcn, label='Pred KPD')
                        plt.title(f'KPD Prediction - {dataset} {batch} - {battery_name}')
                        plt.legend()
                        plt.savefig(f"EvaluationPlots/KPD_{dataset}_{batch}_{battery_name}.png")
                        plt.close()
                    
                plt.figure()
                for m in ['DeepOPINN', 'LAX', 'DOPDeepOLAX', 'KADOPLAX (PostProcessed)']:
                    if m in model_preds:
                        y_true_soh = model_preds[m]['y_true'][start_idx:end_idx]
                        y_pred_soh = model_preds[m]['y_pred'][start_idx:end_idx]
                        
                        if len(y_true_soh) > 0:
                            mse, rmse, mae, mape = get_kpis(y_true_soh, y_pred_soh)
                            battery_kpis.append({
                                'Dataset': dataset, 'Batch': batch, 'Battery': battery_name, 'Model': m,
                                'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape
                            })
                            
                            plt.plot(y_pred_soh, label=f'Pred SOH ({m})')
                
                if 'DOPDeepOLAX' in model_preds:
                    y_true_dop = model_preds['DOPDeepOLAX']['y_true'][start_idx:end_idx]
                    if len(y_true_dop) > 0:
                        plt.plot(y_true_dop, label='True SOH', color='black', linewidth=2)
                
                plt.title(f'SOH Prediction - {dataset} {batch} - {battery_name}')
                plt.legend()
                plt.savefig(f"EvaluationPlots/SOH_{dataset}_{batch}_{battery_name}.png")
                plt.close()
                
                start_idx = end_idx

    pd.DataFrame(batch_kpis).to_csv('batch_level_kpis.csv', index=False)
    pd.DataFrame(battery_kpis).to_csv('battery_level_kpis.csv', index=False)
    print("Evaluation complete. Saved to batch_level_kpis.csv, battery_level_kpis.csv and EvaluationPlots/")

if __name__ == '__main__':
    main()
