from dataloader.dataloader import XJTUdata, TJUdata, MITdata, HUSTdata, NASAdata
from natsort import natsorted
import os



def load_XJTU_data(args, small_sample=None, data_path='data/Full'):
    root = data_path + '/XJTU data'
    args.root = root
    data = XJTUdata(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)

    if args.run_optuna:
        batches = ['R2.5', 'R3', 'satellite'] # batch 3, 4, 6

    for file in files:
        if args.run_optuna:
            for args.batch in batches:
                if args.batch in file:
                    if '4' in file or '8' in file:
                        test_list.append(os.path.join(root, file))
                    else:
                        train_list.append(os.path.join(root, file))
        else:
            if args.batch == 'All' or args.batch in file:
                if '4' in file or '8' in file:
                    test_list.append(os.path.join(root, file))
                else:
                    train_list.append(os.path.join(root, file))
                
    if small_sample is not None:
        train_list = train_list[:small_sample]

    if args.run_optuna:
        args.batch = '_'.join(batches)

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': train_loader['train_2'],
                  'valid': train_loader['valid_2'],
                  'test': test_loader['test_3']}
    return dataloader


def load_TJU_data(args, small_sample=None, data_path='data/Full'):
    root = data_path + '/TJU data'
    args.root = root
    data = TJUdata(root=root, args=args)
    train_list = []
    test_list = []

    # The numbers whose units digit is 5 or 9 are test set, and the others are training set
    mod = [(5,9),(4,8),(5,9)]
    batchs = sorted(os.listdir(root))
    TJU_batches = {
        'NCA': 0,
        'NCM': 1, 
        'NCM_NCA': 2 
    }
    
    if args.batch == 'All':
        batch_ids = [0, 1, 2]
    else:
        batch_ids = [TJU_batches[args.batch]]
        
    for batch_id in batch_ids:
        batch = batchs[batch_id]
        batch_root = os.path.join(root, batch)
        files = os.listdir(batch_root)
        for i,f in enumerate(files):
            id = i + 1
            if id % 10 == mod[batch_id][0] or id % 10 == mod[batch_id][1]:
                test_list.append(os.path.join(batch_root, f))
                # breakpoint()
                print(f)
            else:
                train_list.append(os.path.join(batch_root, f))
    if small_sample is not None:
        train_list = train_list[:small_sample]
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': train_loader['train_2'],
                    'valid': train_loader['valid_2'],
                    'test': test_loader['test_3']}
    return dataloader


def load_MIT_data(args, small_sample=None, data_path='data/Full'):
    root = data_path + '/MIT data'
    args.root = root
    train_list = []
    test_list = []
    
    for batch in ['2017-05-12','2017-06-30','2018-04-12']:
        batch_root = os.path.join(root,batch)
        files = os.listdir(batch_root)
        for f in files:
            id = int(f.split('-')[-1].split('.')[0])
            if id % 5 == 0:
                test_list.append(os.path.join(batch_root,f))
            else:
                train_list.append(os.path.join(batch_root,f))

    if small_sample is not None:
        train_list = train_list[:small_sample]    
    data = MITdata(root=root, args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader


def load_HUST_data(args, small_sample=None, data_path='data/Full'):
    test_id = ['1-4','1-8','2-4','2-8',
               '3-4','3-8','4-4','4-8',
               '5-4','5-7','6-4','6-8',
               '7-4','7-8','8-4','8-8',
               '9-4','9-8','10-4','10-8']
    root = data_path + '/HUST data'
    args.root = root
    data = HUSTdata(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)
    for f in files:
        if f[:-4] in test_id:
            test_list.append(f'{root}/{f}')
        else:
            train_list.append(f'{root}/{f}')
    if small_sample is not None:
        train_list = train_list[:small_sample]

    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader


def load_NASA_data(args, small_sample=None, data_path='data/Full'):
    root = data_path + '/NASA data'
    args.root = root
    data = NASAdata(root=root, args=args)
    file_path = os.path.join(args.root, 'NASA-original-nature.csv')
    dataloader = data.get_dataloaders(specific_path=file_path)

    return dataloader
