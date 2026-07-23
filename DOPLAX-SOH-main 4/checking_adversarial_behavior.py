import os
import shutil
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')



def plot_detailed_losses_from_log(dataset_name:str, log_file:str, output_dir:str, batch:str=None):
    """
    Extracts data loss, PDE loss, and physics loss for both Train and Validation
    from the log file, then plots and saves them.
    
    Args:
        log_file (str): Path to the logfile.
        output_dir (str): Directory where plots will be saved.
    """
    # Regex patterns
    train_detail_pattern = re.compile(
        r"\[Train Details\] epoch:(\d+), data loss:([0-9.eE+-]+), PDE loss:([0-9.eE+-]+), physics loss:([0-9.eE+-]+)"
    )
    valid_detail_pattern = re.compile(
        r"\[Valid Details\] epoch:(\d+), data loss:([0-9.eE+-]+), PDE loss:([0-9.eE+-]+), physics loss:([0-9.eE+-]+)"
    )

    # Containers
    epochs_train, data_train, pde_train, phys_train = [], [], [], []
    epochs_valid, data_valid, pde_valid, phys_valid = [], [], [], []

    # Read file
    with open(log_file, "r") as f:
        for line in f:
            # Train details
            m = train_detail_pattern.search(line)
            if m:
                epochs_train.append(int(m.group(1)))
                data_train.append(float(m.group(2)))
                pde_train.append(float(m.group(3)))
                phys_train.append(float(m.group(4)))

            # Validation details
            m = valid_detail_pattern.search(line)
            if m:
                epochs_valid.append(int(m.group(1)))
                data_valid.append(float(m.group(2)))
                pde_valid.append(float(m.group(3)))
                phys_valid.append(float(m.group(4)))

    # Ensure output dir exists
    os.makedirs(output_dir + '/adversarial_plots_for_losses', exist_ok=True)

    # Combined plots
    def plot_combined(train_epochs, train_losses, valid_epochs, valid_losses, label, fname):
        plt.figure(figsize=(8, 5))
        plt.plot(train_epochs, train_losses, label=f"Train {label}", color="blue")
        plt.plot(valid_epochs, valid_losses, label=f"Valid {label}", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Train vs Validation {label}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'adversarial_plots_for_losses', fname))
        plt.close()

    if batch is None:
        data_lable = f"Data Loss (dataset_name:{dataset_name})"
        data_fname = f"train_vs_valid_data_loss_{dataset_name}.png"
        pde_lable = f"PDE Loss (dataset_name:{dataset_name})"
        pde_fname = f"train_vs_valid_pde_loss_{dataset_name}.png"
        phys_lable = f"Physics Loss (dataset_name:{dataset_name})"
        phys_fname = f"train_vs_valid_physics_loss_{dataset_name}.png"
    else:
        data_lable = f"Data Loss (dataset_name:{dataset_name}-batch:{batch})"
        data_fname = f"train_vs_valid_data_loss_{dataset_name}_{batch}.png"
        pde_lable = f"PDE Loss (dataset_name:{dataset_name}-batch:{batch})"
        pde_fname = f"train_vs_valid_pde_loss_{dataset_name}_{batch}.png"
        phys_lable = f"Physics Loss (dataset_name:{dataset_name}-batch:{batch})"
        phys_fname = f"train_vs_valid_physics_loss_{dataset_name}_{batch}.png"
        
    plot_combined(epochs_train, data_train, epochs_valid, data_valid, data_lable, data_fname)
    plot_combined(epochs_train, pde_train, epochs_valid, pde_valid, pde_lable, pde_fname)
    plot_combined(epochs_train, phys_train, epochs_valid, phys_valid, phys_lable, phys_fname)


def iterate_experiments(output_path:str, dataset_name:str, experiments:list, last_path:str, batch:str=None):
    for exp in experiments:
        if exp.endswith('.csv') or exp.endswith('.xlsx') or exp.endswith('.ipynb_checkpoints') or exp.endswith('.tar.xz') or exp.endswith('.txt'):
            continue
        experiment_path = os.path.join(last_path, exp)
        if batch is None:
            dest_logfile_path = os.path.join(output_path, f'{model_name}_logfile_{dataset_name}.txt')
        else:
            dest_logfile_path = os.path.join(output_path, f'{model_name}_logfile_{dataset_name}_{batch}.txt')
        files = os.listdir(experiment_path)
        for file in files:
            if 'log' in file:
                logfile_path = os.path.join(experiment_path, file)
                shutil.copy(
                    logfile_path,
                    dest_logfile_path
                )
                print(f'Copied logfile to {dest_logfile_path}')
                plot_detailed_losses_from_log(dataset_name=dataset_name, batch=batch, log_file=logfile_path, output_dir=output_dir)


        
if __name__ == "__main__":
    output_dir = root_path = '/handcrafted_features/results of reviewer/DeepOPINN/Adversarial_behavior'
    model_name = 'DeepOPINN'
    datasets = ['XJTU', 'TJU', 'HUST', 'MIT']
    output_path = os.path.join(output_dir, 'logfiles')
    os.makedirs(output_dir + '/logfiles', exist_ok=True)

    for dataset in datasets:
        if dataset.split(' results')[0] not in datasets:
            continue
        dataset_path = os.path.join(root_path, dataset + ' results')
        
        if dataset in ['MIT', 'HUST']:
            experiments = os.listdir(dataset_path)
            iterate_experiments(output_path=output_path, dataset_name=dataset, experiments=experiments, last_path=batch_path)
        else:
            batches = os.listdir(dataset_path)
            for batch in batches:
                if batch.endswith('.csv') or batch.endswith('.xlsx') or batch.endswith('.ipynb_checkpoints') or batch.endswith('.tar.xz') or batch.endswith('.txt'):
                    continue
                batch_path = os.path.join(dataset_path, batch)
                experiments = os.listdir(batch_path)
                iterate_experiments(output_path=output_path, dataset_name=dataset, batch=batch, experiments=experiments, last_path=batch_path)