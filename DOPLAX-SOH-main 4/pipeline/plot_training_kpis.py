import os
import json
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2,
    "figure.autolayout": True,
})

import seaborn as sns

def plot_training_kpis(project_root: str):
    reports_dir = os.path.join(project_root, 'outputs', 'reports')
    save_dir = os.path.join(reports_dir, 'training_kpis_plots')
    os.makedirs(save_dir, exist_ok=True)
    
    # Define mapping of files to model names
    kpi_files = {
        'DeepOPINN': 'deepopinn_kpis.json',
        'LAX': 'lax_training_kpis.json',
        'FusionMLP': 'fusion_training_kpis.json',
        'TST (Phase1)': 'phase1_kpi_report.json'
    }
    
    all_data = []
    
    for model_name, filename in kpi_files.items():
        filepath = os.path.join(reports_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for dataset, batches in data.items():
            for batch, metrics in batches.items():
                metrics['Model'] = model_name
                metrics['Dataset'] = dataset
                metrics['Batch'] = batch
                metrics['Dataset_Batch'] = f"{dataset}_{batch}"
                all_data.append(metrics)
                
    if not all_data:
        print("No KPI data found to plot.")
        return
        
    df = pd.DataFrame(all_data)
    
    sns.set_theme(style="whitegrid")
    
    metrics_to_plot = {
        'training_time_seconds': 'Training Time (Seconds)',
        'peak_ram_mb': 'Peak RAM Usage (MB)',
        'num_batteries': 'Number of Batteries',
        'total_cycles': 'Total Cycles'
    }
    
    for col, title in metrics_to_plot.items():
        if col not in df.columns:
            continue
            
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df, x='Dataset_Batch', y=col, hue='Model', palette='viridis')
        plt.title(f'Comparison of {title} across Datasets/Batches', fontsize=16)
        plt.ylabel(title, fontsize=14)
        plt.xlabel('Dataset & Batch', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{col}_comparison.png"), dpi=300)
        plt.close()
        
    # Generate and save Dataset Level Training KPIs table
    dataset_kpi = df.groupby(['Dataset', 'Model'])[['training_time_seconds', 'training_time_minutes', 'peak_ram_mb']].mean().reset_index()
    dataset_kpi.to_csv(os.path.join(save_dir, 'dataset_level_training_kpis.csv'), index=False)
    with open(os.path.join(save_dir, 'dataset_level_training_kpis.md'), 'w') as f:
        f.write("# Dataset Level Training KPIs\n\n")
        f.write(dataset_kpi.to_markdown(index=False))
        
    # Generate and save Batch Level Training KPIs table
    batch_kpi = df.groupby(['Dataset', 'Batch', 'Model'])[['training_time_seconds', 'training_time_minutes', 'peak_ram_mb', 'num_batteries', 'total_cycles']].mean().reset_index()
    batch_kpi.to_csv(os.path.join(save_dir, 'batch_level_training_kpis.csv'), index=False)
    with open(os.path.join(save_dir, 'batch_level_training_kpis.md'), 'w') as f:
        f.write("# Batch Level Training KPIs\n\n")
        f.write(batch_kpi.to_markdown(index=False))

    print(f"Training KPI plots and tables saved to {save_dir}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    plot_training_kpis(project_root)
