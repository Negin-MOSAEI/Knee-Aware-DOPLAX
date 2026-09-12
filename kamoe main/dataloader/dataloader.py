import os
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .knee_points import KneePointRegistry

warnings.filterwarnings('ignore')


class BatteryPreprocessor:
    """
    Handles CSV reading, 3-sigma outlier cleaning, capacity/SOH normalization,
    and feature scaling (min-max or z-score).
    """

    def __init__(self, normalization_method: str = 'min-max', nominal_capacity: Optional[float] = None):
        self.normalization_method = normalization_method
        self.nominal_capacity = nominal_capacity

    @staticmethod
    def _3_sigma(series: pd.Series) -> np.ndarray:
        std = series.std()
        if std == 0 or np.isnan(std):
            return np.array([], dtype=int)
        mean = series.mean()
        rule = (series < mean - 3 * std) | (series > mean + 3 * std)
        return np.where(rule)[0]

    def delete_3_sigma(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        out_indices = set()
        for col in df.columns:
            out_indices.update(self._3_sigma(df[col]))
        if out_indices:
            df = df.drop(index=list(out_indices)).reset_index(drop=True)
        return df

    def process_csv(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        
        # Add cycle index column if not already present
        if 'cycle index' not in df.columns:
            df.insert(df.shape[1] - 1, 'cycle index', np.arange(df.shape[0]))

        # Remove 3-sigma outliers
        df = self.delete_3_sigma(df)

        # Normalize capacity (SOH)
        if self.nominal_capacity is not None and self.nominal_capacity > 0:
            df['capacity'] = df['capacity'] / self.nominal_capacity

        # Normalize features (all columns except the target 'capacity')
        f_df = df.iloc[:, :-1].copy()
        if self.normalization_method == 'min-max':
            f_min = f_df.min()
            f_max = f_df.max()
            f_denom = np.where(f_max - f_min == 0, 1.0, f_max - f_min)
            f_df = 2.0 * (f_df - f_min) / f_denom - 1.0
        elif self.normalization_method == 'z-score':
            f_mean = f_df.mean()
            f_std = f_df.std().replace(0, 1.0)
            f_df = (f_df - f_mean) / f_std

        df.iloc[:, :-1] = f_df
        return df


class BatteryTrajectoryDataset(Dataset):
    """
    PyTorch Dataset where each item represents ONE FULL BATTERY trajectory.
    Maintains exact cycle sequence order (unshuffled) for sequential models (like TCN)
    while also providing transition pairs (x1, x2, y1, y2) for physics/operator models (DeepOPINN, LAX).
    """

    def __init__(
        self,
        file_paths: List[str],
        dataset_name: str,
        batch_name: str,
        nominal_capacity: Optional[float],
        normalization_method: str = 'min-max',
        knee_registry: Optional[KneePointRegistry] = None,
    ):
        self.file_paths = file_paths
        self.dataset_name = dataset_name
        self.batch_name = batch_name
        self.nominal_capacity = nominal_capacity
        self.normalization_method = normalization_method
        self.preprocessor = BatteryPreprocessor(
            normalization_method=normalization_method,
            nominal_capacity=nominal_capacity
        )
        self.knee_registry = knee_registry or KneePointRegistry()
        self.battery_data_cache: List[Dict[str, Any]] = []

        self._preload_all()

    def _preload_all(self):
        for path in self.file_paths:
            battery_name = os.path.basename(path)
            clean_name = battery_name[:-4] if battery_name.endswith('.csv') else battery_name

            df = self.preprocessor.process_csv(path)
            x = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

            num_cycles = len(df)
            cycle_indices = np.arange(num_cycles, dtype=np.float32).reshape(-1, 1)

            # Query knee point from registry
            knee_point = self.knee_registry.get_knee_point(
                dataset=self.dataset_name,
                batch=self.batch_name,
                battery_name=clean_name
            )

            # If knee point not found in registry, default to last cycle or fallback
            if knee_point is None:
                knee_point = num_cycles

            # Knee distance formula: - arctan(C - C_knee)
            knee_distance = -np.arctan(cycle_indices - float(knee_point)).astype(np.float32)
            # Post-knee indicator: 1 if t >= knee_point, else 0
            is_post_knee = (cycle_indices >= knee_point).astype(np.float32)

            # Consecutive step pairs for physics-informed / operator learning (DeepOPINN, LAX)
            x1 = x[:-1]
            x2 = x[1:]
            y1 = y[:-1]
            y2 = y[1:]

            item = {
                'x': torch.from_numpy(x),                           # [L_i, D]
                'y': torch.from_numpy(y),                           # [L_i, 1]
                'cycle_indices': torch.from_numpy(cycle_indices),   # [L_i, 1]
                'knee_point': int(knee_point),                      # Scalar
                'knee_distance': torch.from_numpy(knee_distance),   # [L_i, 1]
                'is_post_knee': torch.from_numpy(is_post_knee),     # [L_i, 1]
                'x1': torch.from_numpy(x1),                         # [L_i - 1, D]
                'x2': torch.from_numpy(x2),                         # [L_i - 1, D]
                'y1': torch.from_numpy(y1),                         # [L_i - 1, 1]
                'y2': torch.from_numpy(y2),                         # [L_i - 1, 1]
                'battery_name': clean_name,
                'file_path': path,
                'dataset': self.dataset_name,
                'batch': self.batch_name,
                'length': num_cycles,
                'nominal_capacity': self.nominal_capacity,
            }
            self.battery_data_cache.append(item)

    def __len__(self) -> int:
        return len(self.battery_data_cache)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.battery_data_cache[idx]


def single_battery_collate_fn(batch: List[Dict[str, Any]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Collate function ensuring each battery trajectory is preserved intact with its full variable length.
    If batch_size=1 (default), returns the single battery dictionary directly.
    If batch_size > 1, returns the list of battery dictionaries.
    """
    if len(batch) == 1:
        return batch[0]
    return batch


def create_battery_dataloader(
    dataset: BatteryTrajectoryDataset,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Creates a DataLoader yielding one battery per batch (preserving variable cycle length L_i).
    """
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=single_battery_collate_fn,
        num_workers=num_workers
    )
