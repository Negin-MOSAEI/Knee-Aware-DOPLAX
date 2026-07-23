import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from pandas import ExcelWriter
from Model.utils.losses import mae as MAE
from Model.utils.losses import mse as MSE
from Model.utils.losses import mape as MAPE
from Model.utils.losses import rmse as RMSE
import regex as re
import warnings
warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'



class InvestigatingLosses(object):
    def __init__(self, dataset_name, root_path, finetuning_mode, n_experiments=None, n_batches=None):
        self.dataset_name = dataset_name
        self.root_path = root_path
        self.finetuning_mode = finetuning_mode
        self.n_experiments = n_experiments
        self.n_batches = n_batches
        if self.finetuning_mode == True:
            XJTUTJU_pattern = re.compile(r"^(XJTU|TJU)-(XJTU|TJU)$")
            HUSTMIT_pattern = re.compile(r"^(HUST|MIT)-(HUST|MIT)$")
            if re.match(XJTUTJU_pattern, dataset_name):
                if dataset_name == "XJTU-TJU":
                    self.dataset_batches = [f'batch{i}' for i in range(3)]
                    self.experiments = [f'Experiment{i}' for i in range(10)] 
                elif dataset_name == "TJU-XJTU":
                    self.dataset_batches = [f'batch{i}' for i in range(6)]
                    self.experiments = [f'Experiment{i}' for i in range(10)]
            elif re.match(HUSTMIT_pattern, dataset_name):
                if dataset_name in ["HUST-MIT", "MIT-HUST"]:
                    self.dataset_batches = None
                    self.experiments = [f'Experiment{i}' for i in range(10)]
            else:                
                self.dataset_batches = None
                self.experiments = [f'Experiment{i}' for i in range(1, 11)]
        else:
            batches = {
                "XJTU": [f'{i}-{i}' for i in range(6)] if n_batches is None else [f'{i}-{i}' for i in range(n_batches)],
                "TJU": [f'{i}-{i}' for i in range(3)] if n_batches is None else [f'{i}-{i}' for i in range(n_batches)],
                "HUST": None,
                "MIT": None, 
                "NASA": None
            }
            self.dataset_batches = batches[dataset_name]
            self.experiments = [f'Experiment{i}' for i in range(1, 11)] if n_experiments is None else [f'Experiment{i}' for i in range(1, n_experiments+1)]
            
        self.df_batch_experiments_losses = pd.DataFrame()
        self.batches_experiments = []
        self.batch = None
        self.actual_lable = None
        self.predicted_lable = None
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.losses = []
        

    def _calculate_losses(self):
        """Calculate and store MSE, MAE, MAPE, and RMSE losses"""
        # MSE Loss
        # mse is a float object
        mse = MSE(self.predicted_lable.cpu(), self.actual_lable.cpu()) 
        self.losses.append(mse)
        print(f"MSE: {mse}")
        
        # MAE Loss
        mae = MAE(self.predicted_lable.cpu(), self.actual_lable.cpu())
        self.losses.append(mae)
        print(f"MAE: {mae}")

        mape = MAPE(self.predicted_lable.cpu(), self.actual_lable.cpu())
        self.losses.append(mape)
        print(f"MAPE: {mape}")
        
        # RMSE
        rmse = np.sqrt(mse)
        self.losses.append(rmse)
        print(f"RMSE: {rmse}")
        print('\n')


    def _manipulate_datasets(self):
        """Calculate statistics across all experiments"""
        for batch_experiments in self.batches_experiments:
            original_rows = [np.nan] * (batch_experiments.shape[0] - 4)
            avg_10_exp, med_10_exp, std_10_exp = [], [], []
            min_10_exp, max_10_exp = [], []
            
            for i in range(4, 0, -1):
                avg_10_exp.append(round(np.nanmean(batch_experiments.iloc[-i, :].values), 4))
                med_10_exp.append(round(np.nanmedian(batch_experiments.iloc[-i, :].values), 4))
                std_10_exp.append(round(np.nanstd(batch_experiments.iloc[-i, :].values), 4))
                min_10_exp.append(round(np.nanmin(batch_experiments.iloc[-i, :].values), 4))
                max_10_exp.append(round(np.nanmax(batch_experiments.iloc[-i, :].values), 4))

            batch_experiments["Average_of_10_Experiments"] = original_rows + avg_10_exp
            batch_experiments["Median_of_10_Experiments"] = original_rows + med_10_exp
            batch_experiments["Standard_Deviation_of_10_Experiments"] = original_rows + std_10_exp
            batch_experiments["Minimum_of_10_Experiments"] = original_rows + min_10_exp
            batch_experiments["Maximum_of_10_Experiments"] = original_rows + max_10_exp


    def get_all_batches_experiments(self): 
        """Process all batches of experiments"""
        for self.batch in self.dataset_batches:
            print(f'Batch {self.batch}:\n\n')
            self.df_batch_experiments_losses = pd.DataFrame()
            self.get_one_batch_experiments()


    def get_one_batch_experiments(self):
        """Process one batch of experiments"""
        for experiment in self.experiments:
            print(f'{experiment}:')
            try:
                # Load data
                if self.dataset_batches:
                    actual_lable = np.load(os.path.join(self.root_path, self.batch, experiment, 'true_label.npy'))
                    predicted_lable = np.load(os.path.join(self.root_path, self.batch, experiment, 'pred_label.npy'))
                else:
                    actual_lable = np.load(os.path.join(self.root_path, experiment, 'true_label.npy'))
                    predicted_lable = np.load(os.path.join(self.root_path, experiment, 'pred_label.npy')) 
                
                # Create DataFrame for actual and predicted values
                if actual_lable.size == actual_lable.shape[0]:  # 1D case
                    data = {
                        f'{experiment}_Actual_Label': actual_lable.reshape(-1), 
                        f'{experiment}_Predicted_Label': predicted_lable.reshape(-1)
                    }
                    df_experiment = pd.DataFrame(data)
                    num_data_rows = len(df_experiment)
                else:  # 2D case
                    df_experiment = pd.DataFrame()
                    num_data_rows = 0

                # Convert to tensors and calculate losses
                self.actual_lable = torch.tensor(actual_lable, dtype=torch.float32)
                self.predicted_lable = torch.tensor(predicted_lable, dtype=torch.float32)
                self.losses = []
                self._calculate_losses()
                
                # Create loss DataFrame (always 4 rows)
                loss_df = pd.DataFrame({
                    f'{experiment}_Actual_Label': [np.nan]*4,
                    f'{experiment}_Predicted_Label': self.losses
                }, index=['MSE', 'MAE', 'MAPE', 'RMSE'])
                
                # Concatenate data and losses
                concatenated_df = pd.concat([df_experiment, loss_df])
                
                # Add to batch results
                self.df_batch_experiments_losses = pd.concat(
                    [self.df_batch_experiments_losses, concatenated_df], 
                    axis=1
                )
                
            except Exception as e:
                print(f"Error processing {experiment}: {str(e)}")
                placeholder_df = pd.DataFrame({
                    f'{experiment}_Actual_Label': [np.nan]*4,
                    f'{experiment}_Predicted_Label': [np.nan]*4
                }, index=['MSE', 'MAE', 'MAPE', 'RMSE'])
                self.df_batch_experiments_losses = pd.concat(
                    [self.df_batch_experiments_losses, placeholder_df], 
                    axis=1
                )

        self.batches_experiments.append(self.df_batch_experiments_losses)
        print('\n\n')
        print('-' * 90)


    def save_losses(self):
        """Save results to Excel file"""
        file_path = os.path.join(self.root_path, f'{self.dataset_name}_test_losses.xlsx')
        with ExcelWriter(file_path) as writer:
            if self.dataset_batches:
                for i, batch in enumerate(self.dataset_batches):
                    self.batches_experiments[i].to_excel(writer, sheet_name=batch)
            else:
                self.batches_experiments[0].to_excel(writer, sheet_name="one_batch")
            print(f"Saved {self.dataset_name}_test_losses.xlsx in the path {self.root_path}")


    def forward(self):
        """Main execution method"""
        if self.dataset_batches:
            self.get_all_batches_experiments()
        else:
            self.get_one_batch_experiments()
        self._manipulate_datasets()
        self.save_losses()



if __name__ == "__main__":
    finetuning_mode = False # True, False
    process_all = True # True, False
    small_sample = True # True, False
    samples = [1, 2] # [1, 2], [1, 2, 3, 4]
    datasets = ["XJTU", "TJU", "HUST", "MIT"] # "NASA"
    dataset_name = "TJU"  # "XJTU", "HUST", "MIT", "TJU", "NASA"
    sub_root_path = "results of reviewer/Combination/Bagging_u"
    
    if process_all:
        if finetuning_mode:
            for source in datasets:
                for target in datasets:
                    if source == target:
                        continue
                        
                    dataset_name = f'{source}-{target}'
                    root_path = os.path.join(sub_root_path, f"{dataset_name}") 
                    losses = InvestigatingLosses(dataset_name, root_path, finetuning_mode)
                    losses.forward()
        else:
            for dataset_name in datasets:
                if small_sample:
                    for sample in samples:
                        root_path = os.path.join(sub_root_path, f"{dataset_name} results (small sample {sample})") 
                        losses = InvestigatingLosses(dataset_name, root_path, finetuning_mode)
                        losses.forward()
                else:
                    root_path = os.path.join(sub_root_path, f"{dataset_name} results") 
                    losses = InvestigatingLosses(dataset_name, root_path, finetuning_mode)
                    losses.forward()
    else: 
        if small_sample:
            for sample in samples:
                root_path = os.path.join(sub_root_path, f"{dataset_name} results (small sample {sample})") 
                losses = InvestigatingLosses(dataset_name, root_path, finetuning_mode)
                losses.forward()
        else:
            root_path = os.path.join(sub_root_path, f"{dataset_name} results") 
            losses = InvestigatingLosses(dataset_name, root_path, finetuning_mode)
            losses.forward()
