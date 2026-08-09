import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.metrics import calculate_rmse, calculate_mae, calculate_mape

DATASETS = {
    'HUST': ['default'],
    'MIT': ['default'],
    'TJU': ['NCA', 'NCM', 'NCA_NCM'],
    'XJTU': ['2C', '3C', 'R2.5', 'R3', 'RW', 'sim_satellite']
}

def run_phase7(project_root: str):
    """Executes Phase 7: Inference KPI Reporting for TEST batteries."""
    print("Starting Phase 7: Inference KPI Reporting...")
    
    config_dir = os.path.join(project_root, 'config')
    preds_dir = os.path.join(project_root, 'outputs', 'final_results')
    reports_dir = os.path.join(project_root, 'outputs', 'reports', 'inference_kpis')
    
    with open(os.path.join(config_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
        
    all_metrics = []
    
    for dataset, batches in train_test_split.items():
        dataset_reports_dir = os.path.join(reports_dir, dataset)
        os.makedirs(dataset_reports_dir, exist_ok=True)
        
        for batch in batches:
            test_bats = train_test_split[dataset][batch]['test']
            
            for bat_id in test_bats:
                pred_file = os.path.join(preds_dir, dataset, batch, f"{bat_id.replace('/', '_')}_results.npz")
                if not os.path.exists(pred_file):
                    continue
                    
                data = np.load(pred_file)
                true_soh = data['true_soh']
                
                for method, key in [('KaDOPLAX', 'predicted_soh'), ('Post_KaDOPLAX', 'post_processed_soh')]:
                    pred_soh = data[key]
                    rmse = calculate_rmse(true_soh, pred_soh)
                    mae = calculate_mae(true_soh, pred_soh)
                    mape = calculate_mape(true_soh, pred_soh)
                    
                    all_metrics.append({
                        'Dataset': dataset,
                        'Batch': batch,
                        'Battery': bat_id,
                        'Method': method.upper(),
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape
                    })
                    
    if not all_metrics:
        print("No predictions found to evaluate.")
        return
        
    df = pd.DataFrame(all_metrics)
    
    # Save overall CSV
    df.to_csv(os.path.join(reports_dir, 'all_inference_metrics.csv'), index=False)
    
    # Generate grouped bar charts for each metric
    sns.set_theme(style="whitegrid")
    
    for metric in ['RMSE', 'MAE', 'MAPE']:
        plt.figure(figsize=(14, 7))
        # Group by Dataset and Batch for the x-axis, hue by Method
        df['Dataset_Batch'] = df['Dataset'] + "_" + df['Batch']
        sns.barplot(data=df, x='Dataset_Batch', y=metric, hue='Method', palette='Set2', errorbar=None)
        
        plt.title(f'Comparison of {metric} on Test Set across Datasets', fontsize=16)
        plt.ylabel(metric, fontsize=14)
        plt.xlabel('Dataset & Batch', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Method')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'test_{metric.lower()}_comparison.png'), dpi=300)
        plt.close()

    print("Phase 7 completed successfully. KPIs saved.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_phase7(project_root)
