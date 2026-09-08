from dataloader.dataloader import XJTUdata, TJUdata, MITdata, HUSTdata, NASAdata
from natsort import natsorted
import os
import json

def _get_splits(dataset_name, batch_name):
    with open('config/train_test_split.json', 'r') as f:
        splits = json.load(f)
    if dataset_name not in splits or batch_name not in splits[dataset_name]:
        raise ValueError(f"Splits for {dataset_name} - {batch_name} not found in config.")
    return splits[dataset_name][batch_name]

def load_XJTU_data(args, small_sample=None, data_path='data/Full'):
    print(f'DEBUG load_XJTU_data called with data_path={data_path}')
    root = data_path + '/XJTU data'
    args.root = root
    data = XJTUdata(root=root, args=args)
    
    batch_name = 'Sim_satellite' if args.batch == 'satellite' else args.batch
    splits = _get_splits('XJTU', batch_name)
    train_list = [os.path.join(root, b + '.csv') for b in splits['train']]
    val_list = [os.path.join(root, b + '.csv') for b in splits['val']]
    test_list = [os.path.join(root, b + '.csv') for b in splits['test']]
    
    if small_sample is not None:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(specific_path_list=train_list)
    val_loader = data.read_all(specific_path_list=val_list)
    test_loader = data.read_all(specific_path_list=test_list)
    
    dataloader = {'train': train_loader['train_3'],
                  'valid': val_loader['test_3'],
                  'test': test_loader['test_3']}
    return dataloader


def load_TJU_data(args, small_sample=None, data_path='data/Full'):
    root = data_path + '/TJU data'
    args.root = root
    data = TJUdata(root=root, args=args)
    
    tju_map = {
        'NCA': 'Dataset_1_NCA_battery',
        'NCM': 'Dataset_2_NCM_battery',
        'NCM_NCA': 'Dataset_3_NCM_NCA_battery'
    }
    batch_name = tju_map.get(args.batch, args.batch)

    splits = _get_splits('TJU', batch_name)
    train_list = [os.path.join(root, b + '.csv') for b in splits['train']]
    val_list = [os.path.join(root, b + '.csv') for b in splits['val']]
    test_list = [os.path.join(root, b + '.csv') for b in splits['test']]

    if small_sample is not None:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(specific_path_list=train_list)
    val_loader = data.read_all(specific_path_list=val_list)
    test_loader = data.read_all(specific_path_list=test_list)
    
    dataloader = {'train': train_loader['train_3'],
                  'valid': val_loader['test_3'],
                  'test': test_loader['test_3']}
    return dataloader


def load_MIT_data(args, small_sample=None, data_path='data/Full'):
    root = data_path + '/MIT data'
    args.root = root
    data = MITdata(root=root, args=args)
    
    splits = _get_splits('MIT', args.batch)
    train_list = [os.path.join(root, b + '.csv') for b in splits['train']]
    val_list = [os.path.join(root, b + '.csv') for b in splits['val']]
    test_list = [os.path.join(root, b + '.csv') for b in splits['test']]

    if small_sample is not None:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(specific_path_list=train_list)
    val_loader = data.read_all(specific_path_list=val_list)
    test_loader = data.read_all(specific_path_list=test_list)
    
    dataloader = {'train': train_loader['train_3'],
                  'valid': val_loader['test_3'],
                  'test': test_loader['test_3']}
    return dataloader


def load_HUST_data(args, small_sample=None, data_path='data/Full'):
    root = data_path + '/HUST data'
    args.root = root
    data = HUSTdata(root=root, args=args)
    
    splits = _get_splits('HUST', args.batch)
    train_list = [os.path.join(root, b + '.csv') for b in splits['train']]
    val_list = [os.path.join(root, b + '.csv') for b in splits['val']]
    test_list = [os.path.join(root, b + '.csv') for b in splits['test']]

    if small_sample is not None:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(specific_path_list=train_list)
    val_loader = data.read_all(specific_path_list=val_list)
    test_loader = data.read_all(specific_path_list=test_list)
    
    dataloader = {'train': train_loader['train_3'],
                  'valid': val_loader['test_3'],
                  'test': test_loader['test_3']}
    return dataloader


def load_NASA_data(args, small_sample=None, data_path='data/Full'):
    root = data_path + '/NASA data'
    args.root = root
    data = NASAdata(root=root, args=args)
    file_path = os.path.join(args.root, 'NASA-original-nature.csv')
    dataloader = data.get_dataloaders(specific_path=file_path)

    return dataloader
