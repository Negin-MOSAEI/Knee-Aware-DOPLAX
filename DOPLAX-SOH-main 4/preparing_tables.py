import numpy as np
import pandas as pd
import os
import shutil
import regex as re
import openpyxl



def prepare_source_only_results(root_path:str, experiment_number:int=None):
    """
    Extracts the RMSE values from the logging files of source-only experiments for each dataset pair
    """
    rmse_vals, indexes = [], []
    datasets = ['XJTU', 'TJU', 'MIT', 'HUST']
    for source in datasets:
        for target in datasets:
            if source == target:
                continue

            subroot_path = os.path.join(root_path, f'{source}-{target}')

            if experiment_number is None:
                exp = None
            elif type(experiment_number) is int:
                exp = 'Experiment' + f'{experiment_number}'

            if f'{source}-{target}' == 'XJTU-TJU':
                batch = 'batch2'
                if exp is None:
                    log_file_path = os.path.join(subroot_path, batch, 'logging.txt')
                else:
                    log_file_path = os.path.join(subroot_path, batch, exp, 'logging.txt')
            elif f'{source}-{target}' == 'TJU-XJTU':
                batch = 'batch0'
                if exp is None:
                    log_file_path = os.path.join(subroot_path, batch, 'logging.txt')
                else:
                    log_file_path = os.path.join(subroot_path, batch, exp, 'logging.txt')
            else:
                log_file_path = os.path.join(subroot_path, exp, 'logging.txt')

            with open(log_file_path, 'r') as f:
                text = f.readlines()
                text = ''.join(text)

            # Find the line containing "Source only"
            source_line = re.search(r"Source only:.*", text).group(0)
            # Extract the RMSE value using regex
            rmse_match = float(re.findall(r"RMSE: (\d+\.\d+)", source_line)[0])
            
            rmse_vals.append(round(rmse_match, 4))
            indexes.append(f'{source}-{target}')
    
    # Create a DataFrame to store the results
    result_df = pd.DataFrame(data=rmse_vals, index=indexes, columns=['RMSE'])
    result_df.index.name = 'Source-Target'
    result_df.to_csv(os.path.join(root_path, 'source_only_results.csv'), index=True)
    print(f'Successfully saved source_only_results.csv in the path {root_path}')

                
def extract_metrics(root_path:str):
    """
    Extracts metrics (MAPE and RMSE) from the loss files in the specified root path and saves them into a final CSV file.
    """
    try:
        folders = os.listdir(root_path)
    except:
        raise FileNotFoundError(f"The directory {root_path} does not exist or is not accessible.")
    
    dfs = []
    for folder in folders:
        if folder.endswith('.csv') or folder.endswith('.xlsx') or folder.endswith('.ipynb_checkpoints') or folder.endswith('.tar.xz'):
            continue

        dataset_name = folder.split(' results')[0] if 'results' in folder else folder
        file_path = os.path.join(root_path, folder, f'{dataset_name}_test_losses.xlsx')
        workbook = openpyxl.load_workbook(file_path)
        sheetnames = workbook.sheetnames
        
        for sheetname in sheetnames:
            if (folder == 'XJTU-TJU' and sheetname != 'batch2') or \
                (folder == 'TJU-XJTU' and sheetname != 'batch0'):
                continue

            df = pd.read_excel(
                file_path,
                sheet_name=sheetname,
                index_col=0,  # 'Unnamed: 0' as index
                usecols=['Unnamed: 0', 'Average_of_10_Experiments']
            ).iloc[-2:]
            index = f'{folder}_{sheetname}'
            data = {
                idx: val 
                for idx, val in zip(
                    df.index,
                    df['Average_of_10_Experiments'].values
                )
            }
            dfs.append(pd.DataFrame(data, index=[index]))
         
    final_df = pd.concat(dfs, axis=0)
    final_df.index.name = 'Average_of_10_Experiments'
    final_df.to_csv(os.path.join(root_path, 'final_metrics.csv'))
    print(f'Successfully saved final_metrics.csv in the path {root_path}')

    
def find_the_best_experiment(root_path:str, model_name:str, on_validation:bool=True):
    """
    Finds the best experiment among 10 experiments for each dataset based on the minimum MAPE or RMSE.
    """
    try:
        folders = os.listdir(root_path)
    except:
        raise FileNotFoundError(f"The directory {root_path} does not exist or is not accessible.")
    
    batches = {
        'XJTU': ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite'],
        'TJU': ['NCA', 'NCM', 'NCM_NCA']
    }

    dfs = []
    for folder in folders:
        if folder.endswith('.csv') or folder.endswith('.xlsx') or folder.endswith('.ipynb_checkpoints') or folder.endswith('.tar.xz'):
            continue
            
        dataset_name = folder.split(" results")[0]
        if on_validation:
            file_path = os.path.join(root_path, folder, f'{dataset_name}_validation_losses.xlsx')
        else:
            file_path = os.path.join(root_path, folder, f'{dataset_name}_test_losses.xlsx')
        workbook = openpyxl.load_workbook(file_path)
        sheetnames = workbook.sheetnames

        for sheetname in sheetnames:  
            if on_validation:
                df = pd.read_excel(file_path, sheet_name=sheetname, index_col=0)
                exp_nums = df.index.tolist()
                rmse_vals = df['RMSE'].tolist()
                min_rmse = min(rmse_vals)
                index_rmse = rmse_vals.index(min_rmse)
                min_exp = exp_nums[index_rmse]
                
            else:
                df = pd.read_excel(file_path, sheet_name=sheetname, index_col=0).iloc[-2:]
                first_col = df.columns[0]
                index = int(first_col.split('Experiment')[1].split('_')[0])
                mapes = df.iloc[0, :-5].dropna().tolist()
                mapes = [(i, round(mape, 4)) for i, mape in enumerate(mapes, start=index)]
                min_of_mape = df['Minimum_of_10_Experiments'].iloc[0]
                rmses = df.iloc[1, :-5].dropna().tolist()
                rmses = [(i, round(rmse, 4)) for i, rmse in enumerate(rmses, start=index)]
                min_of_rmse = df['Minimum_of_10_Experiments'].iloc[1]
    
                if min_of_mape < min_of_rmse:
                    exp_nums, mape_vals = zip(*mapes)
                    min_mape = min(mape_vals)
                    index_mape = mape_vals.index(min_mape)
                    min_exp = exp_nums[index_mape]
                else:
                    exp_nums, rmse_vals = zip(*rmses)
                    min_rmse = min(rmse_vals)
                    index_rmse = rmse_vals.index(min_rmse)
                    min_exp = exp_nums[index_rmse]
                
            if dataset_name in ['MIT', 'HUST', 'NASA']:
                temp_index = f'{model_name}_model_{dataset_name}.pth'
            else:
                batch_i = int(sheetname.split('-')[0])

                if dataset_name == 'XJTU':
                    batch = batches['XJTU'][batch_i]
                elif dataset_name == 'TJU':
                    batch = batches['TJU'][batch_i]
                temp_index = f'{model_name}_model_{dataset_name}_{batch}.pth'
                
            temp_df = pd.DataFrame(data=[min_exp], columns=['Experiment'], index=[temp_index])
            dfs.append(temp_df)   

    final_df = pd.concat(dfs, axis=0)
    if on_validation:
        final_df.index.name = 'The_best_experiment_among_10_experiments_on_validation'
        csv_file_name = 'The_best_experiment_among_10_experiments_on_validation.csv'
        final_df.to_csv(os.path.join(root_path, csv_file_name))
        print(f'Successfully saved {csv_file_name} in the path {root_path}')
    else:
        final_df.index.name = 'The_best_experiment_among_10_experiments_on_test'
        csv_file_name = 'The_best_experiment_among_10_experiments_on_test.csv'
        final_df.to_csv(os.path.join(root_path, csv_file_name))
        print(f'Successfully saved {csv_file_name} in the path {root_path}')


def find_the_best_pretrained_model(root_path, best_exp_file_path, model_name:str, on_validation:bool=True):
    """
    Moves the best pretrained model files to a designated directory based on the best experiment file.
    """
    try:
        folders = os.listdir(root_path)
    except:
        raise FileNotFoundError(f"The directory {root_path} does not exist or is not accessible.")

    df = pd.read_csv(best_exp_file_path)
    
    current_path = os.getcwd()
    destination_root_path = os.path.join(current_path, 'pretrained_models', model_name)
    if not os.path.exists(destination_root_path):
        os.makedirs(destination_root_path)
        
    number_of_experiments = 10
    if on_validation:
        key = "The_best_experiment_among_10_experiments" + "_" + "on_validation"
    else:
        key = "The_best_experiment_among_10_experiments" + "_" + "on_test"
        
    batches = {
        'XJTU': ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite'],
        'TJU': ['NCA', 'NCM', 'NCM_NCA']
    }
        
    for folder in folders:
        if folder.endswith('.csv') or folder.endswith('.xlsx') or folder.endswith('.ipynb_checkpoints') or folder.endswith('.tar.xz'):
            continue
            
        dataset_name = folder.split(" results")[0]
        df_dataset = df[df[key].str.contains(dataset_name)]
        # print('df_dataset:', df_dataset)
        
        if dataset_name == 'XJTU':
            batch_numbers = 6 
        elif dataset_name == 'TJU':
            batch_numbers = 3
        else:
            batch_numbers = 1
        
        for batch_i in range(batch_numbers):
            for exp_i in range(1, number_of_experiments + 1):
                if batch_numbers > 1:
                    df_sample = df_dataset[df_dataset[key].str.contains(f'{batches[dataset_name][batch_i]}')]
                    model_src_path = os.path.join(root_path, folder, f'{batch_i}-{batch_i}', f'Experiment{exp_i}', 'best_model.pth')
                    logfile_src_path = os.path.join(root_path, folder, f'{batch_i}-{batch_i}', f'Experiment{exp_i}', 'logging.txt')
                else:
                    df_sample = df_dataset
                    model_src_path = os.path.join(root_path, folder, f'Experiment{exp_i}', 'best_model.pth')
                    logfile_src_path = os.path.join(root_path, folder, f'Experiment{exp_i}', 'logging.txt')

                if on_validation:
                    if exp_i != int(df_sample['Experiment'].iloc[0].split('Experiment ')[1]):
                        continue
                else:
                    if exp_i != int(df_sample['Experiment'].iloc[0]):
                        continue
                    
                model_dest_path = os.path.join(
                    destination_root_path, 
                    f"{df_sample[key].iloc[0]}"
                )
                
                logfile_name = f'{model_name}_logfile' + df_sample[key].iloc[0].split(f'{model_name}_model')[1].split('.pth')[0] + '.txt'
                logfile_dest_path = os.path.join(
                    destination_root_path, 
                    f"{logfile_name}"
                )
                src_files = [model_src_path, logfile_src_path]
                dest_files = [model_dest_path, logfile_dest_path]

                for i in range(2):
                    shutil.copy(
                        src_files[i],
                        dest_files[i]
                    )
                    if i == 0:
                        print(f"Successfully copied {df_sample[key].iloc[0]} to {destination_root_path}") 
                    else:
                        print(f"Successfully copied {logfile_name} to {destination_root_path}")          



if __name__ == '__main__':
    # prepare the metrics
    metric_results = True # True, False
    metric_path = "results of reviewer/Combination/Bagging_u/small-samples_both_freeze_mlp_Phi"

    # prepare pretrained model
    prepare_pretrained_model = False # True, False
    # raw_data = True # True, False
    model_path = 'results of reviewer/Combination/Bagging_u/both_freeze_MLP_Phi'
    model_name = 'DOPLAX-DOPLAX' # "PINN", "DeepONet", "DeepOPINN", "DOPFormer", # DOPLAX,
    on_validation = True # True, False
    if on_validation:
        csv_file_name = "The_best_experiment_among_10_experiments" + "_" + "on_validation.csv"
    else:
        csv_file_name = "The_best_experiment_among_10_experiments" + "_" + "on_test.csv"
    best_exp_file_path = os.path.join(model_path, csv_file_name)

    # prepare the results of the table 4
    table_4_results = False # True, False
    pretrained_model_path = f'pretrained_models/{model_name}'
    finetuning_path = 'results_fine-tuning/DOPLAX-MLP'
    experiment_number = 1
    
    # prepare the metrics
    if metric_results:
        if not os.path.exists(metric_path):
            raise FileNotFoundError(f"The directory {metric_path} does not exist. \
                                    First run the main modules with the mentioned path, then run this script again.")
            
        if not os.path.exists(os.path.join(metric_path, 'final_metrics.csv')):
            extract_metrics(metric_path)
        else:
            print(f'final_metrics.csv has been created before in this path {metric_path}')

    # prepare pretrained model
    if prepare_pretrained_model:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The directory {model_path} does not exist. \
                                    First run the main modules with this path {metric_path}, then run this script again.")
            
        if not os.path.exists(best_exp_file_path):
            find_the_best_experiment(model_path, model_name=model_name, on_validation=on_validation)
        else:
            print(f'the_best_experiment_among_10_experiments.csv has been created before in this path {model_path}')
            
        find_the_best_pretrained_model(model_path, best_exp_file_path, model_name=model_name, on_validation=on_validation)

    # prepare the results of the table 4
    if table_4_results:
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"The directory {pretrained_model_path} does not exist. \
                                    First run the main modules with this path {metric_path}, then run this script again.")
        if not os.path.exists(finetuning_path):
            raise FileNotFoundError(f"The directory {finetuning_path} does not exist. \
                                    First run the fine-tuning modules with the mentioned path, then run this script again.")
            
        if not os.path.exists(os.path.join(finetuning_path, 'source_only_results.csv')):
            prepare_source_only_results(finetuning_path, experiment_number)
        else:
            print(f'source_only_results.csv has been created before in this path {finetuning_path}')
            
        if not os.path.exists(os.path.join(finetuning_path, 'final_metrics.csv')):
            extract_metrics(finetuning_path)
        else:
            print(f'final_metrics.csv has been created before in this path {finetuning_path}')
