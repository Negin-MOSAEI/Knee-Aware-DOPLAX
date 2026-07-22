import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.util import write_to_json
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")



def get_y_true_y_pred_paths(root_path, num_of_batch:int, n_experiments:int=10):
    y_labels = []
    for batch in range(num_of_batch):
        for exp in range(n_experiments):
            if num_of_batch == 1:
                y_true_path = os.path.join(root_path, f'Experiment{exp+1}', 'true_label.npy')
                y_pred_path = os.path.join(root_path, f'Experiment{exp+1}', 'pred_label.npy')
            else:
                y_true_path = os.path.join(root_path, f'{batch}-{batch}', f'Experiment{exp+1}', 'true_label.npy')
                y_pred_path = os.path.join(root_path, f'{batch}-{batch}', f'Experiment{exp+1}', 'pred_label.npy')
            y_labels.append((y_true_path, y_pred_path))
    return y_labels

    
def save_differences_plot(y_true_path, y_pred_path):
    if "XJTU" in y_true_path or "TJU" in y_true_path:
        batch_num, exp_num = y_true_path.split('/')[-3:-1]
        batch_num, exp_num = int(batch_num.split('-')[0]), int(exp_num.split('Experiment')[1])
    else:
        batch_num, exp_num = "one_batch", y_true_path.split('/')[-2]
        exp_num = int(exp_num.split('Experiment')[1])
        
    if "XJTU" in y_true_path:
        dataset_name = "XJTU"
    elif "TJU" in y_true_path:
        dataset_name = "TJU"
    elif "MIT" in y_true_path:
        dataset_name = "MIT"
    else:
        dataset_name = "HUST"

    try:
        y_true = np.load(y_true_path) 
        y_pred = np.load(y_pred_path) 
    except:
        print
        return

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
    cycles_path = os.path.join(cycles_root, 'selected_cycles_info.json')
    write_to_json(cycles_path, info)

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
        elif len(y_true_cp.shape) == 2: 
            actuals = y_true[cycle_id, :]
            preds = y_pred[cycle_id, :]

        plt.plot(actuals, label='Actual SOH', alpha=.5, color='blue')
        plt.plot(preds, label='Predicted SOH', alpha=.5, color='red')
        if batch_num == "one_batch":
            title = f"Differences Plot - (Dataset:{dataset_name}-Experiment:{exp_num}-Cycle:{cycle_id})"
            file_name = f"differences_plot_experiment{exp_num}_cycle{cycle_id}"
        else:
            title = f"Differences Plot - (Dataset:{dataset_name}-Batch:{batch_num}-Experiment:{exp_num}-Cycle:{cycle_id})"
            file_name = f"differences_plot_batch{batch_num}_experiment{exp_num}_cycle{cycle_id}"
            
        plt.title(title)
        plt.xlabel('Cycles')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        output_path_root = os.path.join(y_true_path.split('true_label.npy')[0], 'plots')
        if not os.path.exists(output_path_root):
            os.mkdir(output_path_root)
        output_path = os.path.join(output_path_root, file_name)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f'Saved sucessfully (iteration:{i+1} - cycle:{cycle_id}): {output_path}')


def main(root_path:str):
    if "XJTU" in root_path:
        y_labels = get_y_true_y_pred_paths(root_path, num_of_batch=6)         
    elif "TJU" in root_path:
        y_labels = get_y_true_y_pred_paths(root_path, num_of_batch=3) 
    else:
        y_labels = get_y_true_y_pred_paths(root_path, num_of_batch=1)

    for y_true_path, y_pred_path in y_labels:
        save_differences_plot(y_true_path, y_pred_path)
     


if __name__ == "__main__":
    datasets = ['XJTU', 'HUST', 'TJU', 'MIT']
    for dataset in datasets:
        root_path = f"/RUL Group/PINN4SOH/results of reviewer/deeponet+pinn_fully-sin_e2000-p80_related_coefficients/{dataset} results"
        print(f'Processing the {dataset} dataset ....')
        main(root_path)
        print('*' * 90)
        print('\n\n')