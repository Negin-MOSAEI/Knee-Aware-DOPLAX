import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
from utils.util import write_to_file
import warnings
warnings.filterwarnings('ignore')



class DF():
    def __init__(self, args):
        self.normalization = True
        self.normalization_method = args.normalization_method # min-max, z-score
        self.args = args


    def _3_sigma(self, Ser1):
        '''
        :param Ser1:
        :return: index
        '''
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        return index


    def delete_3_sigma(self, df):
        '''
        :param df: DataFrame
        :return: DataFrame
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.reset_index(drop=True)
        out_index = []
        for col in df.columns:
            index = self._3_sigma(df[col])
            out_index.extend(index)
        out_index = list(set(out_index))
        df = df.drop(out_index, axis=0)
        df = df.reset_index(drop=True)
        return df


    def read_one_csv(self, file_name, nominal_capacity=None):
        '''
        read a csv file and return a DataFrame
        :param file_name: str
        :return: DataFrame
        '''
        df = pd.read_csv(file_name)

        if self.args.data != 'NASA':
            df.insert(df.shape[1]-1,'cycle index',np.arange(df.shape[0]))

        df = self.delete_3_sigma(df)

        if self.args.data != 'NASA' and nominal_capacity is not None:
            df['capacity'] = df['capacity'] / nominal_capacity
        
        f_df = df.iloc[:, :-1]
        if self.normalization_method == 'min-max':
            f_df = 2*(f_df - f_df.min())/(f_df.max() - f_df.min()) - 1
        elif self.normalization_method == 'z-score':
            f_df = (f_df - f_df.mean())/f_df.std()

        df.iloc[:, :-1] = f_df

        return df


    def load_one_battery(self, path, nominal_capacity=None):
    # def load_one_battery(self, path, i, nominal_capacity=None):
        '''
        Read a csv file and divide the data into x and y
        :param path:
        :param nominal_capacity:
        :return:
        '''
        df = self.read_one_csv(path,nominal_capacity)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]
        return (x1,y1),(x2,y2)


    def load_all_battery(self, path_list:list=None, nominal_capacity:float=None, specific_path:str=None):
        '''
        Read multiple csv files, divide the data into X and Y, and then package it into a dataloader
        :param path_list: list of file paths
        :param nominal_capacity: nominal capacity, used to calculate SOH
        :param batch_size: batch size
        :return: Dataloader
        '''
        if self.args.data != 'NASA':
            X1, X2, Y1, Y2 = [], [], [], []
    
            if self.args.run_mode == 'LAX':
                pass
            else:
                if self.args.log_dir is not None and self.args.save_folder is not None:
                    save_name = os.path.join(self.args.save_folder, self.args.log_dir)
                    write_to_file(save_name, 'data path:')
                    write_to_file(save_name, str(path_list))
    
            for path in path_list:
            # for i, path in enumerate(path_list):
                (x1, y1), (x2, y2) = self.load_one_battery(path, nominal_capacity)
                # (x1, y1), (x2, y2) = self.load_one_battery(path, i, nominal_capacity)
                X1.append(x1)
                X2.append(x2)
                Y1.append(y1)
                Y2.append(y2)
    
            X1 = np.concatenate(X1, axis=0)
            X2 = np.concatenate(X2, axis=0)
            Y1 = np.concatenate(Y1, axis=0)
            Y2 = np.concatenate(Y2, axis=0)
    
            tensor_X1 = torch.from_numpy(X1).float()
            tensor_X2 = torch.from_numpy(X2).float()
            tensor_Y1 = torch.from_numpy(Y1).float().view(-1,1)
            tensor_Y2 = torch.from_numpy(Y2).float().view(-1,1)
    
            drop_last = False 
    
            # Condition 1
            # 1.1 
            split = int(tensor_X1.shape[0] * 0.8)
            train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]
            train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
            train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
            train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]
            # 1.2 
            train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
                train_test_split(train_X1, train_X2, train_Y1, train_Y2, test_size=0.2, random_state=420)
    
            train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                      batch_size=self.args.batch_size,
                                      shuffle=True)
            valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                      batch_size=self.args.batch_size,
                                      shuffle=True)
            test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
                                     batch_size=self.args.batch_size,
                                     shuffle=False)
    
            # Condition 2
            train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
                train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)
            train_loader_2 = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                      batch_size=self.args.batch_size,
                                      drop_last=drop_last,
                                      shuffle=True)
            valid_loader_2 = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                      batch_size=self.args.batch_size,
                                      drop_last=drop_last,
                                      shuffle=True)
    
            # Condition 3
            test_loader_3 = DataLoader(TensorDataset(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2),
                                     batch_size=self.args.batch_size,
                                     drop_last=drop_last,
                                     shuffle=False)
    
            loader = {
                'train': train_loader, 
                'valid': valid_loader, 
                'test': test_loader,
                'train_2': train_loader_2,
                'valid_2': valid_loader_2,
                'test_3': test_loader_3
            }
            
        else:
            df = self.read_one_csv(specific_path)
            X = df.iloc[:, :-1].values
            Y = df.iloc[:, -1].values
            X1 = X[:-1]
            X2 = X[1:]
            Y1 = Y[:-1]
            Y2 = Y[1:]

            tensor_X1 = torch.from_numpy(X1).float()
            tensor_X2 = torch.from_numpy(X2).float()
            tensor_Y1 = torch.from_numpy(Y1).float().view(-1,1)
            tensor_Y2 = torch.from_numpy(Y2).float().view(-1,1)

            total_train_X1, test_X1, total_train_X2, test_X2, total_train_Y1, test_Y1, total_train_Y2, test_Y2 = \
            train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)
            train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
                train_test_split(total_train_X1, total_train_X2, total_train_Y1, total_train_Y2, test_size=0.2, random_state=420)
            
            train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                      batch_size=self.args.batch_size,
                                      shuffle=True)
            valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                      batch_size=self.args.batch_size,
                                      shuffle=True)
            test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
                                     batch_size=self.args.batch_size,
                                     shuffle=False)
            loader = {
                'train': train_loader, 
                'valid': valid_loader, 
                'test': test_loader,
            }
            
        return loader



class XJTUdata(DF):
    def __init__(self, root, args):
        super(XJTUdata, self).__init__(args)
        self.root = root
        self.file_list = os.listdir(root)
        self.variables = pd.read_csv(os.path.join(root, self.file_list[0])).columns
        self.num = len(self.file_list)
        self.batch_names = ['2C','3C','R2.5','R3','RW','satellite']
        self.batch_size = args.batch_size

        if self.normalization:
            self.nominal_capacity = 2.0
        else:
            self.nominal_capacity = None


    def read_all(self, specific_path_list=None):
        '''
        Read all csv files, divide the data into four parts: x1, y1, x2, y2, and encapsulate it into a dataloader
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            for file in self.file_list:
                path = os.path.join(self.root, file)
                file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)



class TJUdata(DF):
    def __init__(self, root, args):
        super(TJUdata, self).__init__(args)
        self.root = root
        self.batchs = ['Dataset_1_NCA_battery','Dataset_2_NCM_battery','Dataset_3_NCM_NCA_battery']
        if self.normalization:
            self.nominal_capacities = [3.5,3.5,2.5]
        else:
            self.nominal_capacities = [None,None,None]


    def read_all(self, specific_path_list):
        '''
        Read all csv files and encapsulate them into a dataloader
        :param self:
        :return: dict
        '''
        for i,batch in enumerate(self.batchs):
            if batch in specific_path_list[0]:
                normal_capacity = self.nominal_capacities[i]
                break
        return self.load_all_battery(path_list=specific_path_list, nominal_capacity=normal_capacity)
    


class MITdata(DF):
    def __init__(self, root, args):
        super(MITdata, self).__init__(args)
        self.root = root
        self.batchs = ['2017-05-12','2017-06-30','2018-04-12']
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None


    def read_all(self, specific_path_list=None):
        '''
        Read all csv files.
        If specific_path_list is not None, read the specified file; otherwise read all files;
        :param self:
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            for batch in self.batchs:
                root = os.path.join(self.root, batch)
                files = os.listdir(root)
                for file in files:
                    path = os.path.join(root, file)
                    file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)
        


class HUSTdata(DF):
    def __init__(self, root, args):
        super(HUSTdata, self).__init__(args)
        self.root = root
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None


    def read_all(self, specific_path_list=None):
        '''
        Read all csv files.
        If specific_path_list is not None, read the specified file;
        otherwise read all files;
        :param self:
        :param specific_path:
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            files = os.listdir(self.root)
            for file in files:
                path = os.path.join(self.root, file)
                file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)



class NASAdata(DF):
    def __init__(self, root, args):
        super(NASAdata, self).__init__(args)
        self.root = root
        

    def get_dataloaders(self, specific_path=None):
        '''
        Read all csv files.
        If specific_path_list is not None, read the specified file;
        otherwise read all files;
        :param self:
        :param specific_path:
        :return: dict
        '''
        return self.load_all_battery(specific_path=specific_path)

import math
from utils.math_utils import calculate_kpd, tst_cold_start_kpd
import json

class BatteryCycleDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, dataset_name, battery_ids, window_size=40, capacity_column_index=0, num_features=None, initial_knee_points=None):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.battery_ids = battery_ids
        self.window_size = window_size
        self.capacity_column_index = capacity_column_index
        self.num_features = num_features
        self.initial_knee_points = initial_knee_points or {}
        
        self.samples = []
        
        # Determine actual absolute data_dir based on dataset_name
        self.data_dir = os.path.join(self.data_root, f"{self.dataset_name} data")
        self._load_and_process_data()

    def _calculate_soh(self, raw_data_matrix, bat_id):
        # Calculate SOH as capacity / initial capacity
        # Assume capacity is the last column
        capacity = raw_data_matrix[:, -1]
        
        # Determine nominal capacity
        if self.dataset_name == 'XJTU':
            nominal_capacity = 2.0
        elif self.dataset_name == 'MIT':
            nominal_capacity = 1.1
        elif self.dataset_name == 'HUST':
            nominal_capacity = 1.1
        elif self.dataset_name == 'TJU':
            if 'NCM_NCA' in bat_id:
                nominal_capacity = 2.5
            elif 'NCA' in bat_id or 'NCM' in bat_id:
                nominal_capacity = 3.5
            else:
                nominal_capacity = 3.5 # Fallback for TJU
        else:
            # Fallback for unknown datasets
            nominal_capacity = capacity[0] if capacity[0] != 0 else 1e-6
            
        soh = capacity / nominal_capacity
        return soh
        
    def _load_and_process_data(self):
        for bat_id in self.battery_ids:
            # We already have relative path stored as bat_id
            file_path = os.path.join(self.data_dir, f"{bat_id}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CRITICAL: Real data file not found at -> {file_path}")
                
            try:
                df = pd.read_csv(file_path)
                
                # Replace inf and -inf
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna()
                df = df.reset_index(drop=True)
                
                raw_data_matrix = df.values
                soh = self._calculate_soh(raw_data_matrix, bat_id)
                
                num_cycles = raw_data_matrix.shape[0]
                current_num_features = self.num_features if self.num_features is not None else raw_data_matrix.shape[1] - 1
                
                # Get knee point cycle for this battery, fallback to 100 if missing
                # bat_id could be nested like "default/Bat_008" or "1-1", so we need to match appropriately
                def find_knee(d, key):
                    if isinstance(d, dict):
                        if key in d: return d[key]
                        for k, v in d.items():
                            res = find_knee(v, key)
                            if res is not None: return res
                    return None
                    
                bat_key = bat_id.split('/')[-1]
                found_knee = find_knee(self.initial_knee_points, bat_key)
                if found_knee is not None:
                    knee_point_cycle = found_knee
                else:
                    knee_point_cycle = 100
                
                for i in range(num_cycles):
                    current_cycle = i + 1
                    
                    if current_cycle < self.window_size:
                        # Cold start: Pad with initial cycle
                        pad_len = self.window_size - current_cycle
                        pad = np.tile(raw_data_matrix[0, :current_num_features], (pad_len, 1))
                        window_data = np.vstack([pad, raw_data_matrix[:current_cycle, :current_num_features]])
                        kpd = tst_cold_start_kpd(current_cycle, knee_point_cycle)
                    else:
                        window_data = raw_data_matrix[current_cycle - self.window_size:current_cycle, :current_num_features]
                        kpd = calculate_kpd(current_cycle, knee_point_cycle)
                        
                    target_soh = soh[i]
                    self.samples.append((window_data, kpd, target_soh))
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, target_kpd, target_soh = self.samples[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor([target_kpd], dtype=torch.float32), torch.tensor([target_soh], dtype=torch.float32)

def get_dataloader(data_root, dataset_name, battery_ids, batch_size, shuffle=True, capacity_column_index=0, initial_knee_points=None, window_size=40, num_features=None):
    dataset = BatteryCycleDataset(data_root, dataset_name, battery_ids, window_size=window_size, capacity_column_index=capacity_column_index, num_features=num_features, initial_knee_points=initial_knee_points)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
