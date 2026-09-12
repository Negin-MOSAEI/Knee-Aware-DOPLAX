import json
import os
from typing import Optional, Dict, Any, List, Tuple


def _find_json_file(json_filename: str) -> str:
    search_paths = [
        json_filename,
        os.path.join("..", json_filename),
        os.path.join(os.path.dirname(__file__), "..", "..", json_filename),
        os.path.join(os.path.dirname(__file__), "..", json_filename),
        os.path.abspath(json_filename),
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find '{json_filename}'. Checked paths: {search_paths}")


class KneePointRegistry:
    """
    Registry for loading and querying battery knee cycle points from initial_knee_points.json.
    Supports all 13 batches across 4 datasets (HUST, MIT, TJU, XJTU).
    """

    def __init__(self, json_path: str = "initial_knee_points.json"):
        self.json_path = json_path
        self.knee_dict = self._load_json(json_path)

    def _load_json(self, json_path: str) -> Dict[str, Any]:
        resolved_path = _find_json_file(json_path)
        with open(resolved_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_knee_point(self, dataset: str, batch: str, battery_name: str) -> Optional[int]:
        """
        Retrieves the knee cycle point for a given battery.
        """
        clean_name = battery_name[:-4] if battery_name.endswith(".csv") else battery_name

        dataset_upper = dataset.upper()
        if dataset_upper not in self.knee_dict:
            for k in self.knee_dict:
                if k.upper() == dataset_upper:
                    dataset_upper = k
                    break

        dataset_data = self.knee_dict.get(dataset_upper, {})

        # 1. HUST lookup
        if dataset_upper == "HUST":
            hust_data = dataset_data.get("default", dataset_data)
            return hust_data.get(clean_name, None)

        # 2. MIT lookup
        elif dataset_upper == "MIT":
            batch_data = dataset_data.get(batch, {})
            if clean_name in batch_data:
                return batch_data[clean_name]
            prefixed = f"{batch}/{clean_name}"
            if prefixed in batch_data:
                return batch_data[prefixed]
            for k, v in batch_data.items():
                if k.endswith(clean_name):
                    return v

        # 3. TJU lookup
        elif dataset_upper == "TJU":
            batch_data = dataset_data.get(batch, {})
            if clean_name in batch_data:
                return batch_data[clean_name]
            prefixed = f"{batch}/{clean_name}"
            if prefixed in batch_data:
                return batch_data[prefixed]
            for k, v in batch_data.items():
                if k.endswith(clean_name):
                    return v

        # 4. XJTU lookup
        elif dataset_upper == "XJTU":
            batch_data = dataset_data.get(batch, {})
            if clean_name in batch_data:
                return batch_data[clean_name]
            if not batch_data and (batch == 'satellite' or batch == 'Sim_satellite'):
                batch_data = dataset_data.get('Sim_satellite', dataset_data.get('satellite', {}))
            if clean_name in batch_data:
                return batch_data[clean_name]
            for k, v in batch_data.items():
                if k.endswith(clean_name) or clean_name.endswith(k):
                    return v

        return None


class SplitRegistry:
    """
    Registry for loading and querying exact train/val/test splits from train_test_split.json.
    Supports all 13 batches across 4 datasets (HUST, MIT, TJU, XJTU).
    """

    def __init__(self, json_path: str = "train_test_split.json"):
        self.json_path = json_path
        self.split_dict = self._load_json(json_path)

    def _load_json(self, json_path: str) -> Dict[str, Any]:
        resolved_path = _find_json_file(json_path)
        with open(resolved_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_splits(self, dataset: str, batch: str) -> Dict[str, List[str]]:
        """
        Returns a dict with keys: 'train', 'val', 'test' containing battery ID lists.
        """
        dataset_upper = dataset.upper()
        if dataset_upper not in self.split_dict:
            for k in self.split_dict:
                if k.upper() == dataset_upper:
                    dataset_upper = k
                    break

        dataset_data = self.split_dict.get(dataset_upper, {})

        if dataset_upper == "HUST":
            return dataset_data.get("default", {"train": [], "val": [], "test": []})
        
        if batch in dataset_data:
            return dataset_data[batch]

        # Handle aliases like satellite vs Sim_satellite
        if (batch == 'satellite' or batch == 'Sim_satellite'):
            if 'Sim_satellite' in dataset_data:
                return dataset_data['Sim_satellite']
            if 'satellite' in dataset_data:
                return dataset_data['satellite']

        # Handle TJU aliases (e.g. NCA vs Dataset_1_NCA_battery)
        for k in dataset_data:
            if batch in k or k in batch:
                return dataset_data[k]

        return {"train": [], "val": [], "test": []}
