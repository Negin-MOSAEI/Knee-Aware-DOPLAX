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
        from knee_point_detection import KneePointDetector

        df = pd.read_csv(file_name)

        if self.args.data != 'NASA':
            df.insert(df.shape[1]-1,'cycle index',np.arange(df.shape[0]))

        df = self.delete_3_sigma(df)

        has_knee_distance = 'knee_point_distance' in df.columns
        knee_distance = None
        if has_knee_distance:
            knee_distance = df['knee_point_distance'].values.copy()
        elif 'capacity' in df.columns:
            cap = df['capacity'].values.astype(float)
            nom = nominal_capacity if nominal_capacity is not None else (cap[0] if len(cap) > 0 and cap[0] > 0 else 1.0)
            soh = cap / nom
            detector = KneePointDetector(n_percent=2.0)
            knee_distance = detector.compute_knee_distances(soh)
            cap_col_idx = df.columns.get_loc('capacity')
            df.insert(cap_col_idx, 'knee_point_distance', knee_distance)
            has_knee_distance = True

        if self.args.data != 'NASA' and nominal_capacity is not None:
            df['capacity'] = df['capacity'] / nominal_capacity
        
        f_df = df.iloc[:, :-1]
        if self.normalization_method == 'min-max':
            f_df = 2*(f_df - f_df.min())/(f_df.max() - f_df.min()) - 1
        elif self.normalization_method == 'z-score':
            f_df = (f_df - f_df.mean())/f_df.std()

        f_df = f_df.fillna(0.0)
        df.iloc[:, :-1] = f_df

        if has_knee_distance:
            df['knee_point_distance'] = knee_distance

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

        has_knee_distance = 'knee_point_distance' in df.columns
        kd = None
        if has_knee_distance:
            kd = df['knee_point_distance'].values
            feat_cols = [c for c in df.columns if c not in ['capacity', 'knee_point_distance']]
            x = df[feat_cols].values
        else:
            x = df.iloc[:, :-1].values
        y = df['capacity'].values
        
        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]
        
        if has_knee_distance:
            kd1 = kd[:-1]
            kd2 = kd[1:]
            return (x1,y1,kd1),(x2,y2,kd2)
        return (x1,y1),(x2,y2)


    def load_all_battery(self, path_list:list=None, nominal_capacity:float=None, specific_path:str=None):
        '''
        Read multiple csv files, divide the data into X and Y, and then package it into a dataloader.
        If knee_point_distance is present in the CSV, it is collected and returned
        inside each TensorDataset as an extra target tensor.
        :param path_list: list of file paths
        :param nominal_capacity: nominal capacity, used to calculate SOH
        :return: Dataloader dict
        '''
        if self.args.data != 'NASA':
            X1, X2, Y1, Y2 = [], [], [], []
            KD1, KD2 = [], []
            has_knee_distance = False
    
            if self.args.run_mode == 'LAX':
                pass
            else:
                if self.args.log_dir is not None and self.args.save_folder is not None:
                    save_name = os.path.join(self.args.save_folder, self.args.log_dir)
                    write_to_file(save_name, 'data path:')
                    write_to_file(save_name, str(path_list))
    
            for path in path_list:
                result = self.load_one_battery(path, nominal_capacity)
                if len(result[0]) == 3:
                    (x1, y1, kd1), (x2, y2, kd2) = result
                    has_knee_distance = True
                else:
                    (x1, y1), (x2, y2) = result
                X1.append(x1)
                X2.append(x2)
                Y1.append(y1)
                Y2.append(y2)
                if has_knee_distance:
                    KD1.append(kd1)
                    KD2.append(kd2)
    
            X1 = np.concatenate(X1, axis=0)
            X2 = np.concatenate(X2, axis=0)
            Y1 = np.concatenate(Y1, axis=0)
            Y2 = np.concatenate(Y2, axis=0)
    
            tensor_X1 = torch.from_numpy(X1).float()
            tensor_X2 = torch.from_numpy(X2).float()
            tensor_Y1 = torch.from_numpy(Y1).float().view(-1,1)
            tensor_Y2 = torch.from_numpy(Y2).float().view(-1,1)
    
            if has_knee_distance:
                KD1 = np.concatenate(KD1, axis=0)
                KD2 = np.concatenate(KD2, axis=0)
                tensor_KD1 = torch.from_numpy(KD1).float().view(-1,1)
                tensor_KD2 = torch.from_numpy(KD2).float().view(-1,1)
    
            drop_last = False 
    
            def _make_loaders(tx1, tx2, ty1, ty2, tkd1=None, tkd2=None):
                if tkd1 is not None:
                    ds = TensorDataset(tx1, tx2, ty1, ty2, tkd1, tkd2)
                else:
                    ds = TensorDataset(tx1, tx2, ty1, ty2)
                return ds
    
            # Condition 1
            # 1.1
            split = int(tensor_X1.shape[0] * 0.8)
            train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]
            train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
            train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
            train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]
            if has_knee_distance:
                train_KD1, test_KD1 = tensor_KD1[:split], tensor_KD1[split:]
                train_KD2, test_KD2 = tensor_KD2[:split], tensor_KD2[split:]
            # 1.2
            if has_knee_distance:
                train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2, train_KD1, valid_KD1, train_KD2, valid_KD2 = \
                    train_test_split(train_X1, train_X2, train_Y1, train_Y2, train_KD1, train_KD2, test_size=0.2, random_state=420)
            else:
                train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
                    train_test_split(train_X1, train_X2, train_Y1, train_Y2, test_size=0.2, random_state=420)
    
            train_loader = DataLoader(_make_loaders(train_X1, train_X2, train_Y1, train_Y2,
                                                    train_KD1 if has_knee_distance else None,
                                                    train_KD2 if has_knee_distance else None),
                                      batch_size=self.args.batch_size,
                                      shuffle=True)
            valid_loader = DataLoader(_make_loaders(valid_X1, valid_X2, valid_Y1, valid_Y2,
                                                    valid_KD1 if has_knee_distance else None,
                                                    valid_KD2 if has_knee_distance else None),
                                      batch_size=self.args.batch_size,
                                      shuffle=False)
            test_loader = DataLoader(_make_loaders(test_X1, test_X2, test_Y1, test_Y2,
                                                   test_KD1 if has_knee_distance else None,
                                                   test_KD2 if has_knee_distance else None),
                                     batch_size=self.args.batch_size,
                                     shuffle=False)
    
            # Condition 2
            if has_knee_distance:
                train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2, train_KD1, valid_KD1, train_KD2, valid_KD2 = \
                    train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, tensor_KD1, tensor_KD2, test_size=0.2, random_state=420)
            else:
                train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
                    train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)
            train_loader_2 = DataLoader(_make_loaders(train_X1, train_X2, train_Y1, train_Y2,
                                                      train_KD1 if has_knee_distance else None,
                                                      train_KD2 if has_knee_distance else None),
                                      batch_size=self.args.batch_size,
                                      drop_last=drop_last,
                                      shuffle=True)
            valid_loader_2 = DataLoader(_make_loaders(valid_X1, valid_X2, valid_Y1, valid_Y2,
                                                      valid_KD1 if has_knee_distance else None,
                                                      valid_KD2 if has_knee_distance else None),
                                      batch_size=self.args.batch_size,
                                      drop_last=drop_last,
                                      shuffle=True)
    
            # Condition 3
            test_loader_3 = DataLoader(_make_loaders(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2,
                                                     tensor_KD1 if has_knee_distance else None,
                                                     tensor_KD2 if has_knee_distance else None),
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
