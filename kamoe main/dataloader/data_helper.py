import os
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader

from .dataloader import BatteryTrajectoryDataset, create_battery_dataloader
from .knee_points import KneePointRegistry, SplitRegistry


# Canonical metadata and nominal capacities for all 13 batches across 4 datasets
DATASET_METADATA = {
    'XJTU': {
        'nominal_capacity': 2.0,
        'batches': ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite', 'Sim_satellite'],
        'default_dir': 'XJTU data'
    },
    'TJU': {
        'batches': {
            'Dataset_1_NCA_battery': {'nominal_capacity': 3.5},
            'Dataset_2_NCM_battery': {'nominal_capacity': 3.5},
            'Dataset_3_NCM_NCA_battery': {'nominal_capacity': 2.5},
            'NCA': {'nominal_capacity': 3.5, 'folder': 'Dataset_1_NCA_battery'},
            'NCM': {'nominal_capacity': 3.5, 'folder': 'Dataset_2_NCM_battery'},
            'NCM_NCA': {'nominal_capacity': 2.5, 'folder': 'Dataset_3_NCM_NCA_battery'},
        },
        'default_dir': 'TJU data'
    },
    'MIT': {
        'nominal_capacity': 1.1,
        'batches': ['2017-05-12', '2017-06-30', '2018-04-12'],
        'default_dir': 'MIT data'
    },
    'HUST': {
        'nominal_capacity': 1.1,
        'batches': ['default'],
        'default_dir': 'HUST data'
    }
}


def _resolve_data_root(data_root: str, subfolder: str) -> str:
    """Helper to locate the dataset folder across common project paths."""
    candidates = [
        os.path.join(data_root, subfolder),
        os.path.join(data_root, 'Processed', subfolder),
        os.path.join(data_root, 'Full', subfolder),
        os.path.join('data', 'Processed', subfolder),
        os.path.join('..', 'data', 'Processed', subfolder),
        os.path.join('..', 'DOPLAX-SOH-main 4', 'data', 'Processed', subfolder),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(data_root, subfolder)


def _partition_files_by_split(
    files: List[str],
    split_info: Dict[str, List[str]],
    batch_filter: Optional[str] = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Partitions files into train, val, and test lists using exact keys from train_test_split.json.
    """
    train_keys = set(split_info.get("train", []))
    val_keys = set(split_info.get("val", []))
    test_keys = set(split_info.get("test", []))

    train_files, val_files, test_files = [], [], []

    for f in files:
        if not f.endswith('.csv'):
            continue
        
        # If a batch filter is given, verify file belongs to that batch
        if batch_filter and batch_filter != 'default':
            b_norm = 'Sim_satellite' if batch_filter == 'satellite' else batch_filter
            if b_norm not in f and batch_filter not in f:
                continue

        base_name = os.path.basename(f)
        clean_name = base_name[:-4] if base_name.endswith('.csv') else base_name

        # Match against split keys
        def is_match(keys: set) -> bool:
            if clean_name in keys:
                return True
            for k in keys:
                if k.endswith(clean_name) or clean_name.endswith(k) or f.endswith(k) or f.endswith(k + '.csv'):
                    return True
            return False

        if is_match(train_keys):
            train_files.append(f)
        elif is_match(val_keys):
            val_files.append(f)
        elif is_match(test_keys):
            test_files.append(f)

    return train_files, val_files, test_files


def load_XJTU_battery_data(
    batch: str = '2C',
    data_root: str = 'data/Processed',
    normalization_method: str = 'min-max',
    shuffle_train_batteries: bool = False,
    knee_registry: Optional[KneePointRegistry] = None,
    split_registry: Optional[SplitRegistry] = None,
) -> Dict[str, Any]:
    """
    Loads XJTU battery data per battery trajectory using train_test_split.json.
    """
    root = _resolve_data_root(data_root, 'XJTU data')
    if not os.path.exists(root):
        raise FileNotFoundError(f"XJTU data directory not found at {root}")

    split_reg = split_registry or SplitRegistry()
    splits = split_reg.get_splits('XJTU', batch)

    all_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.csv')]
    train_files, val_files, test_files = _partition_files_by_split(all_files, splits, batch_filter=batch)

    nominal_cap = DATASET_METADATA['XJTU']['nominal_capacity']
    reg = knee_registry or KneePointRegistry()

    train_ds = BatteryTrajectoryDataset(train_files, 'XJTU', batch, nominal_cap, normalization_method, reg)
    val_ds = BatteryTrajectoryDataset(val_files, 'XJTU', batch, nominal_cap, normalization_method, reg)
    test_ds = BatteryTrajectoryDataset(test_files, 'XJTU', batch, nominal_cap, normalization_method, reg)

    return {
        'train_dataset': train_ds,
        'valid_dataset': val_ds,
        'test_dataset': test_ds,
        'train': create_battery_dataloader(train_ds, shuffle=shuffle_train_batteries),
        'valid': create_battery_dataloader(val_ds, shuffle=False),
        'test': create_battery_dataloader(test_ds, shuffle=False)
    }


def load_TJU_battery_data(
    batch: str = 'Dataset_1_NCA_battery',
    data_root: str = 'data/Processed',
    normalization_method: str = 'min-max',
    shuffle_train_batteries: bool = False,
    knee_registry: Optional[KneePointRegistry] = None,
    split_registry: Optional[SplitRegistry] = None,
) -> Dict[str, Any]:
    """
    Loads TJU battery data per battery trajectory using train_test_split.json.
    """
    root = _resolve_data_root(data_root, 'TJU data')
    
    batch_map = {
        'NCA': 'Dataset_1_NCA_battery',
        'NCM': 'Dataset_2_NCM_battery',
        'NCM_NCA': 'Dataset_3_NCM_NCA_battery'
    }
    actual_batch = batch_map.get(batch, batch)
    batch_root = os.path.join(root, actual_batch)

    if not os.path.exists(batch_root):
        raise FileNotFoundError(f"TJU batch directory not found at {batch_root}")

    split_reg = split_registry or SplitRegistry()
    splits = split_reg.get_splits('TJU', actual_batch)

    all_files = [os.path.join(batch_root, f) for f in os.listdir(batch_root) if f.endswith('.csv')]
    train_files, val_files, test_files = _partition_files_by_split(all_files, splits)

    nominal_cap = 2.5 if 'Dataset_3' in actual_batch or 'NCM_NCA' in actual_batch else 3.5
    reg = knee_registry or KneePointRegistry()

    train_ds = BatteryTrajectoryDataset(train_files, 'TJU', actual_batch, nominal_cap, normalization_method, reg)
    val_ds = BatteryTrajectoryDataset(val_files, 'TJU', actual_batch, nominal_cap, normalization_method, reg)
    test_ds = BatteryTrajectoryDataset(test_files, 'TJU', actual_batch, nominal_cap, normalization_method, reg)

    return {
        'train_dataset': train_ds,
        'valid_dataset': val_ds,
        'test_dataset': test_ds,
        'train': create_battery_dataloader(train_ds, shuffle=shuffle_train_batteries),
        'valid': create_battery_dataloader(val_ds, shuffle=False),
        'test': create_battery_dataloader(test_ds, shuffle=False)
    }


def load_MIT_battery_data(
    batch: str = '2017-05-12',
    data_root: str = 'data/Processed',
    normalization_method: str = 'min-max',
    shuffle_train_batteries: bool = False,
    knee_registry: Optional[KneePointRegistry] = None,
    split_registry: Optional[SplitRegistry] = None,
) -> Dict[str, Any]:
    """
    Loads MIT battery data per battery trajectory using train_test_split.json.
    """
    root = _resolve_data_root(data_root, 'MIT data')
    batch_root = os.path.join(root, batch)
    if not os.path.exists(batch_root):
        raise FileNotFoundError(f"MIT batch directory not found at {batch_root}")

    split_reg = split_registry or SplitRegistry()
    splits = split_reg.get_splits('MIT', batch)

    all_files = [os.path.join(batch_root, f) for f in os.listdir(batch_root) if f.endswith('.csv')]
    train_files, val_files, test_files = _partition_files_by_split(all_files, splits)

    nominal_cap = DATASET_METADATA['MIT']['nominal_capacity']
    reg = knee_registry or KneePointRegistry()

    train_ds = BatteryTrajectoryDataset(train_files, 'MIT', batch, nominal_cap, normalization_method, reg)
    val_ds = BatteryTrajectoryDataset(val_files, 'MIT', batch, nominal_cap, normalization_method, reg)
    test_ds = BatteryTrajectoryDataset(test_files, 'MIT', batch, nominal_cap, normalization_method, reg)

    return {
        'train_dataset': train_ds,
        'valid_dataset': val_ds,
        'test_dataset': test_ds,
        'train': create_battery_dataloader(train_ds, shuffle=shuffle_train_batteries),
        'valid': create_battery_dataloader(val_ds, shuffle=False),
        'test': create_battery_dataloader(test_ds, shuffle=False)
    }


def load_HUST_battery_data(
    data_root: str = 'data/Processed',
    normalization_method: str = 'min-max',
    shuffle_train_batteries: bool = False,
    knee_registry: Optional[KneePointRegistry] = None,
    split_registry: Optional[SplitRegistry] = None,
) -> Dict[str, Any]:
    """
    Loads HUST battery data per battery trajectory using train_test_split.json.
    """
    root = _resolve_data_root(data_root, 'HUST data')
    if not os.path.exists(root):
        raise FileNotFoundError(f"HUST data directory not found at {root}")

    split_reg = split_registry or SplitRegistry()
    splits = split_reg.get_splits('HUST', 'default')

    all_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.csv')]
    train_files, val_files, test_files = _partition_files_by_split(all_files, splits)

    nominal_cap = DATASET_METADATA['HUST']['nominal_capacity']
    reg = knee_registry or KneePointRegistry()

    train_ds = BatteryTrajectoryDataset(train_files, 'HUST', 'default', nominal_cap, normalization_method, reg)
    val_ds = BatteryTrajectoryDataset(val_files, 'HUST', 'default', nominal_cap, normalization_method, reg)
    test_ds = BatteryTrajectoryDataset(test_files, 'HUST', 'default', nominal_cap, normalization_method, reg)

    return {
        'train_dataset': train_ds,
        'valid_dataset': val_ds,
        'test_dataset': test_ds,
        'train': create_battery_dataloader(train_ds, shuffle=shuffle_train_batteries),
        'valid': create_battery_dataloader(val_ds, shuffle=False),
        'test': create_battery_dataloader(test_ds, shuffle=False)
    }


def load_battery_dataloader(
    dataset_name: str,
    batch_name: str = 'default',
    data_root: str = 'data/Processed',
    normalization_method: str = 'min-max',
    shuffle_train: bool = False,
) -> Dict[str, Any]:
    """
    Unified entrypoint to load data for any of the 13 batches across the 4 datasets.
    """
    d_name = dataset_name.upper()
    if d_name == 'XJTU':
        return load_XJTU_battery_data(
            batch=batch_name,
            data_root=data_root,
            normalization_method=normalization_method,
            shuffle_train_batteries=shuffle_train
        )
    elif d_name == 'TJU':
        return load_TJU_battery_data(
            batch=batch_name,
            data_root=data_root,
            normalization_method=normalization_method,
            shuffle_train_batteries=shuffle_train
        )
    elif d_name == 'MIT':
        return load_MIT_battery_data(
            batch=batch_name,
            data_root=data_root,
            normalization_method=normalization_method,
            shuffle_train_batteries=shuffle_train
        )
    elif d_name == 'HUST':
        return load_HUST_battery_data(
            data_root=data_root,
            normalization_method=normalization_method,
            shuffle_train_batteries=shuffle_train
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: 'XJTU', 'TJU', 'MIT', 'HUST'")
