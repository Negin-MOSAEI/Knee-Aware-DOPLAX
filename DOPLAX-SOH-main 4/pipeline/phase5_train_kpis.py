import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def load_kpis(reports_dir: str) -> pd.DataFrame:
    """
    Parses the generated JSON KPI files and creates a consolidated Pandas DataFrame.
    """
    files = {
        'TST': 'phase1_kpi_report.json',
        'DeepOpinn': 'deepopinn_kpis.json',
        'Bagging_MLP': 'bagging_mlp_kpis.json'
    }
    
    records = []
    
    for model_name, filename in files.items():
        filepath = os.path.join(reports_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: KPI file {filepath} not found. Skipping {model_name}.")
            continue
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for dataset, batches in data.items():
            for batch, kpis in batches.items():
                record = {
                    'Model': model_name,
                    'Dataset': dataset,
                    'Batch': batch,
                    'Dataset_Batch': f"{dataset}_{batch}",
                    'Training_Time_Sec': kpis['training_time_seconds'],
                    'Peak_RAM_MB': kpis['peak_ram_mb'],
                    'Num_Batteries': kpis['num_batteries'],
                    'Total_Cycles': kpis['total_cycles']
                }
                records.append(record)
                
    return pd.DataFrame(records)

def generate_kpi_plots(df: pd.DataFrame, reports_dir: str):
    """
    Generates professional comparative visualizations for the KPIs.
    """
    sns.set_theme(style="whitegrid")
    
    # Ensure plots directory exists
    plots_dir = os.path.join(reports_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Peak RAM Usage
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x='Dataset_Batch', y='Peak_RAM_MB', hue='Model', palette='viridis')
    plt.title('Peak RAM Usage by Model across Datasets and Batches', fontsize=16)
    plt.ylabel('Peak RAM (MB)', fontsize=14)
    plt.xlabel('Dataset & Batch', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'peak_ram_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Training Time
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x='Dataset_Batch', y='Training_Time_Sec', hue='Model', palette='magma')
    plt.title('Total Training Time by Model across Datasets and Batches', fontsize=16)
    plt.ylabel('Training Time (Seconds)', fontsize=14)
    plt.xlabel('Dataset & Batch', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_time_comparison.png'), dpi=300)
    plt.close()
    
    # 3. Data Volume
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x='Dataset_Batch', y='Total_Cycles', hue='Model', palette='coolwarm')
    plt.title('Data Volume (Total Cycles) Used for Training', fontsize=16)
    plt.ylabel('Total Cycles', fontsize=14)
    plt.xlabel('Dataset & Batch', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'data_volume_comparison.png'), dpi=300)
    plt.close()

def run_phase5(project_root: str):
    """
    Executes Phase 5: KPI Aggregation and Visualization.
    """
    print("Starting Phase 5: Aggregating KPIs and Generating Reports...")
    
    reports_dir = os.path.join(project_root, 'outputs', 'reports')
    
    df = load_kpis(reports_dir)
    
    if df.empty:
        print("No KPI data found. Ensure prior training phases ran successfully.")
        return
        
    # Save consolidated table
    csv_path = os.path.join(reports_dir, 'consolidated_kpis.csv')
    df.to_csv(csv_path, index=False)
    print(f"Consolidated KPI table saved to {csv_path}")
    
    # Generate Plots
    generate_kpi_plots(df, reports_dir)
    print(f"KPI Visualizations saved to {os.path.join(reports_dir, 'plots')}")
    
    print("\nPhase 5 completed successfully.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_phase5(project_root)
