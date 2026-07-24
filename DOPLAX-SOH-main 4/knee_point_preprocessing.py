"""
Knee Point Detection and Data Preprocessing Module

This module provides functionality to:
1. Extract SOH trajectories from raw battery cycling data
2. Detect knee points in degradation curves
3. Compute knee_point_distance for each cycle
4. Preprocess and save datasets with the new target variable
"""

import numpy as np
import pandas as pd
import os
import ast
import re
from typing import List, Tuple, Optional, Dict
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


def parse_numpy_array_string(array_str: str) -> np.ndarray:
    """
    Parse a string representation of a numpy array into an actual numpy array.
    
    The strings are in format like:
    "[[0.]\n[0.021]\n[0.038]\n...]"
    or with array() and dtype specifications.
    """
    # Clean the string
    cleaned = array_str.strip()
    
    # Remove array() and dtype specifications
    cleaned = re.sub(r'array\(|\)', '', cleaned)
    cleaned = re.sub(r"dtype='<U19'", '', cleaned)
    cleaned = re.sub(r'dtype="<U19"', '', cleaned)
    cleaned = cleaned.replace('\\n', ',').replace('\n', ',')
    
    # Fix double brackets and commas
    cleaned = cleaned.replace('[[', '[').replace(']]', ']')
    cleaned = cleaned.replace('[,', '[').replace(',]', ']')
    cleaned = cleaned.replace(',,', ',')
    
    # Remove any remaining array( or ) 
    cleaned = cleaned.replace('array(', '').replace(')', '')
    
    try:
        # Parse as Python list
        data = ast.literal_eval(cleaned)
        return np.array(data).flatten()
    except Exception as e:
        # Fallback: try to extract numbers with regex
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', cleaned)
        if numbers:
            return np.array([float(n) for n in numbers])
        return np.array([])


def extract_capacity_trajectory(df: pd.DataFrame, capacity_col: str = 'capacity_Ah') -> np.ndarray:
    """
    Extract end-of-cycle capacity for each row (cycle) in the dataframe.
    
    Args:
        df: DataFrame with raw cycling data
        capacity_col: Name of the column containing capacity arrays
        
    Returns:
        Array of end-of-cycle capacities (one per cycle)
    """
    capacities = []
    for i in range(len(df)):
        cap_str = df.iloc[i][capacity_col]
        cap_array = parse_numpy_array_string(cap_str)
        if len(cap_array) > 0:
            # Last value is the end-of-cycle capacity
            capacities.append(cap_array[-1])
        else:
            capacities.append(np.nan)
    
    capacities = np.array(capacities)
    # Forward fill any NaN values
    capacities = pd.Series(capacities).ffill().bfill().values
    return capacities


def compute_soh(capacities: np.ndarray, nominal_capacity: float) -> np.ndarray:
    """Convert capacities to State of Health (SOH) values."""
    return capacities / nominal_capacity


class KneePointDetector:
    """
    Detects knee points in battery degradation curves using linear regression
    on the initial linear phase and a threshold-based criterion.
    """
    
    def __init__(self, n_percent: float = 2.0, min_linear_points: int = 10, 
                 max_linear_fraction: float = 0.5):
        """
        Initialize the detector.
        
        Args:
            n_percent: Threshold percentage n for knee detection.
                      Knee is where |actual - fitted| >= (n/100) * fitted
            min_linear_points: Minimum number of points to fit initial linear region
            max_linear_fraction: Maximum fraction of trajectory to consider as linear region
        """
        self.n_percent = n_percent
        self.min_linear_points = min_linear_points
        self.max_linear_fraction = max_linear_fraction
        
    def detect_knee_point(self, soh_values: np.ndarray) -> int:
        """
        Detect knee point index in a battery SOH trajectory.
        
        Args:
            soh_values: Array of SOH values from start to EOL
            
        Returns:
            knee_point_index: Index of the first cycle where deviation exceeds threshold
        """
        n_cycles = len(soh_values)
        if n_cycles < self.min_linear_points * 2:
            return n_cycles // 2
        
        max_linear_end = int(n_cycles * self.max_linear_fraction)
        max_linear_end = max(max_linear_end, self.min_linear_points)
        
        best_knee = n_cycles // 2
        best_score = float('inf')
        
        # Try different endpoints for the linear region
        for linear_end in range(self.min_linear_points, max_linear_end + 1):
            x_linear = np.arange(linear_end).reshape(-1, 1)
            y_linear = soh_values[:linear_end]
            
            # Fit linear regression on initial segment
            model = LinearRegression()
            model.fit(x_linear, y_linear)
            
            # Predict on entire trajectory
            x_all = np.arange(n_cycles).reshape(-1, 1)
            y_fitted = model.predict(x_all)
            
            # Compute residuals
            residuals = np.abs(soh_values - y_fitted)
            threshold = (self.n_percent / 100.0) * y_fitted
            
            # Find first point after linear region where residual exceeds threshold
            knee_candidates = np.where(residuals >= threshold)[0]
            knee_candidates = knee_candidates[knee_candidates >= linear_end]
            
            if len(knee_candidates) > 0:
                knee_idx = knee_candidates[0]
                score = self._evaluate_knee_quality(soh_values, knee_idx, model)
                
                if score < best_score:
                    best_score = score
                    best_knee = knee_idx
        
        return best_knee
    
    def _evaluate_knee_quality(self, soh_values: np.ndarray, knee_idx: int, 
                                linear_model: LinearRegression) -> float:
        """
        Evaluate the quality of a detected knee point.
        Lower score = better knee point.
        """
        n_cycles = len(soh_values)
        
        # R² before knee (should be high for good linear fit)
        x_before = np.arange(knee_idx).reshape(-1, 1)
        y_before = soh_values[:knee_idx]
        if len(y_before) > 1:
            model_before = LinearRegression()
            model_before.fit(x_before, y_before)
            r2_before = model_before.score(x_before, y_before)
        else:
            r2_before = 0.0
        
        # R² after knee (should be lower - non-linear degradation)
        x_after = np.arange(knee_idx, n_cycles).reshape(-1, 1)
        y_after = soh_values[knee_idx:]
        if len(y_after) > 1:
            model_after = LinearRegression()
            model_after.fit(x_after, y_after)
            r2_after = model_after.score(x_after, y_after)
            slope_change = abs(model_after.coef_[0] - linear_model.coef_[0])
        else:
            r2_after = 0.0
            slope_change = 0.0
        
        # Score: want high r2_before, low r2_after, significant slope change
        score = (1.0 - r2_before) + r2_after + 0.1 * slope_change
        return score
    
    def compute_knee_distances(self, soh_values: np.ndarray) -> np.ndarray:
        """
        Compute knee point distance for each cycle.

        Definition: 1.0 at start of life (cycle 0), decreases linearly
        to 0.0 at the knee point, stays 0.0 at and after the knee.
        
        Args:
            soh_values: Array of SOH values
            
        Returns:
            Array of knee point distances (normalized to [0, 1])
        """
        knee_idx = self.detect_knee_point(soh_values)
        n_cycles = len(soh_values)
        if knee_idx == 0:
            return np.zeros(n_cycles)
        distances = (knee_idx - np.arange(n_cycles, dtype=float)) / knee_idx
        return np.clip(distances, 0.0, 1.0)


def optimize_n_percent(all_soh_trajectories: List[np.ndarray], 
                       n_range: Tuple[float, float] = (0.5, 10.0),
                       n_steps: int = 20) -> float:
    """
    Automated optimization to find optimal n% threshold across all batteries.
    
    Uses a scoring function based on knee point consistency and quality.
    
    Args:
        all_soh_trajectories: List of SOH arrays from all battery cells
        n_range: (min_n, max_n) range to search
        n_steps: Number of steps in grid search
        
    Returns:
        Optimal n_percent value
    """
    n_values = np.linspace(n_range[0], n_range[1], n_steps)
    best_n = n_range[0]
    best_score = float('inf')
    
    for n in n_values:
        detector = KneePointDetector(n_percent=n)
        scores = []
        
        for soh in all_soh_trajectories:
            if len(soh) < 10:
                continue
            try:
                knee_idx = detector.detect_knee_point(soh)
                score = _evaluate_knee_consistency(soh, knee_idx)
                scores.append(score)
            except Exception:
                scores.append(100.0)
        
        if scores:
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_n = n
    
    return best_n


def _evaluate_knee_consistency(soh_values: np.ndarray, knee_idx: int) -> float:
    """Evaluate consistency of a detected knee point."""
    n = len(soh_values)
    if knee_idx <= 1 or knee_idx >= n - 1:
        return 100.0
    
    # Linear fit before knee
    x_before = np.arange(knee_idx).reshape(-1, 1)
    y_before = soh_values[:knee_idx]
    model_before = LinearRegression().fit(x_before, y_before)
    r2_before = model_before.score(x_before, y_before)
    
    # Linear fit after knee
    x_after = np.arange(knee_idx, n).reshape(-1, 1)
    y_after = soh_values[knee_idx:]
    model_after = LinearRegression().fit(x_after, y_after)
    r2_after = model_after.score(x_after, y_after)
    
    slope_before = model_before.coef_[0]
    slope_after = model_after.coef_[0]
    
    # Slope should change significantly at knee
    slope_ratio = abs(slope_after / slope_before) if abs(slope_before) > 1e-6 else 1.0
    
    # Good knee: high r2_before, lower r2_after, slope_ratio > 1 (steeper degradation)
    score = (1.0 - r2_before) + r2_after + abs(1.0 - slope_ratio)
    return score


def process_battery_file(input_path: str, output_path: str,
                         nominal_capacity: float = 2.0,
                         n_percent: Optional[float] = None,
                         auto_optimize: bool = False,
                         all_trajectories: Optional[List[np.ndarray]] = None) -> pd.DataFrame:
    """
    Process a single battery CSV file and add knee_point_distance column.
    
    Args:
        input_path: Path to raw battery CSV
        output_path: Path to save processed CSV
        nominal_capacity: Nominal capacity for SOH calculation
        n_percent: Fixed n% threshold (if None and auto_optimize=False, uses default 2.0)
        auto_optimize: Whether to auto-optimize n_percent
        all_trajectories: List of all SOH trajectories for joint optimization
        
    Returns:
        Processed DataFrame with knee_point_distance column
    """
    # Read raw data
    df = pd.read_csv(input_path)
    
    # Extract capacity trajectory (end-of-cycle capacity for each cycle)
    capacities = extract_capacity_trajectory(df, 'capacity_Ah')
    
    # Compute SOH
    soh = compute_soh(capacities, nominal_capacity)
    
    # Determine n_percent
    if auto_optimize and all_trajectories is not None:
        n_percent = optimize_n_percent(all_trajectories)
        print(f"  Auto-optimized n_percent = {n_percent:.2f}%")
    elif n_percent is None:
        n_percent = 2.0  # Default
    
    # Detect knee point and compute distances
    detector = KneePointDetector(n_percent=n_percent)
    knee_distances = detector.compute_knee_distances(soh)
    knee_idx = detector.detect_knee_point(soh)
    
    # Create processed DataFrame with one row per cycle
    # Features: we'll use summary statistics from each cycle's voltage, current, temp
    # Target: SOH (normalized capacity)
    # New target: knee_point_distance
    
    processed_rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Extract summary features from each cycle's arrays
        volt_str = row['voltage_V']
        curr_str = row['current_A']
        temp_str = row['temperature_C']
        cap_str = row['capacity_Ah']
        
        volt_arr = parse_numpy_array_string(volt_str)
        curr_arr = parse_numpy_array_string(curr_str)
        temp_arr = parse_numpy_array_string(temp_str)
        cap_arr = parse_numpy_array_string(cap_str)
        
        # Summary statistics as features
        features = {
            'cycle_index': i,
            'voltage_mean': np.mean(volt_arr) if len(volt_arr) > 0 else 0,
            'voltage_std': np.std(volt_arr) if len(volt_arr) > 0 else 0,
            'voltage_min': np.min(volt_arr) if len(volt_arr) > 0 else 0,
            'voltage_max': np.max(volt_arr) if len(volt_arr) > 0 else 0,
            'current_mean': np.mean(curr_arr) if len(curr_arr) > 0 else 0,
            'current_std': np.std(curr_arr) if len(curr_arr) > 0 else 0,
            'temperature_mean': np.mean(temp_arr) if len(temp_arr) > 0 else 0,
            'temperature_std': np.std(temp_arr) if len(temp_arr) > 0 else 0,
            'capacity_start': cap_arr[0] if len(cap_arr) > 0 else 0,
            'capacity_end': cap_arr[-1] if len(cap_arr) > 0 else 0,
            'capacity_delta': cap_arr[-1] - cap_arr[0] if len(cap_arr) > 0 else 0,
            'soh': soh[i],
            'knee_point_distance': knee_distances[i],
        }
        processed_rows.append(features)
    
    processed_df = pd.DataFrame(processed_rows)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"  Processed {len(processed_df)} cycles, knee_idx={knee_idx}, saved to {output_path}")
    
    return processed_df


def process_dataset_directory(input_dir: str, output_dir: str,
                              nominal_capacity: float = 2.0,
                              dataset_name: str = "XJTU") -> Dict:
    """
    Process all battery files in a dataset directory.
    
    Args:
        input_dir: Directory containing raw battery CSV files
        output_dir: Directory to save processed CSV files
        nominal_capacity: Nominal capacity for SOH calculation
        dataset_name: Name of dataset (for logging)
        
    Returns:
        Dictionary with processing results and optimal n_percent
    """
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    print(f"\nProcessing {dataset_name} dataset: {len(csv_files)} battery files")
    
    # First pass: extract all SOH trajectories for joint optimization
    all_soh = []
    file_paths = []
    
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(input_path)
        capacities = extract_capacity_trajectory(df)
        soh = compute_soh(capacities, nominal_capacity)
        all_soh.append(soh)
        file_paths.append(input_path)
    
    # Optimize n_percent globally
    optimal_n = optimize_n_percent(all_soh)
    print(f"  Optimal n_percent for {dataset_name}: {optimal_n:.2f}%")
    
    # Second pass: process each file with optimal n_percent
    processed_dfs = []
    for csv_file, input_path, soh in zip(csv_files, file_paths, all_soh):
        output_path = os.path.join(output_dir, csv_file)
        processed_df = process_battery_file(
            input_path, output_path,
            nominal_capacity=nominal_capacity,
            n_percent=optimal_n,
            auto_optimize=False,
            all_trajectories=all_soh
        )
        processed_dfs.append(processed_df)
    
    return {
        'dataset': dataset_name,
        'optimal_n_percent': optimal_n,
        'num_batteries': len(csv_files),
        'processed_files': [os.path.join(output_dir, f) for f in csv_files]
    }


def process_all_datasets(base_input_dir: str, base_output_dir: str,
                         dataset_configs: Dict[str, Dict]) -> Dict:
    """
    Process all datasets with their respective configurations.
    
    Args:
        base_input_dir: Base directory containing dataset subdirectories
        base_output_dir: Base directory for output
        dataset_configs: Dict mapping dataset names to config dicts with
                        'folder' and 'nominal_capacity' keys
        
    Returns:
        Summary of processing results
    """
    results = {}
    
    for dataset_name, config in dataset_configs.items():
        input_dir = os.path.join(base_input_dir, config['folder'])
        output_dir = os.path.join(base_output_dir, config['folder'])
        nominal_capacity = config.get('nominal_capacity', 2.0)
        
        if not os.path.exists(input_dir):
            print(f"Warning: {input_dir} does not exist, skipping {dataset_name}")
            continue
            
        result = process_dataset_directory(
            input_dir, output_dir, nominal_capacity, dataset_name
        )
        results[dataset_name] = result
    
    return results


if __name__ == '__main__':
    # Configuration for each dataset
    DATASET_CONFIGS = {
        'XJTU': {'folder': 'XJTU data', 'nominal_capacity': 2.0},
        'TJU': {'folder': 'TJU data', 'nominal_capacity': 3.5},
        'MIT': {'folder': 'MIT data', 'nominal_capacity': 1.1},
        'HUST': {'folder': 'HUST data', 'nominal_capacity': 1.1},
    }
    
    BASE_INPUT = "data/Full"
    BASE_OUTPUT = "data/Processed"
    
    print("=" * 60)
    print("KNEE POINT DISTANCE PREPROCESSING PIPELINE")
    print("=" * 60)
    
    if not os.path.exists(BASE_INPUT):
        print(f"\nInput directory not found: {BASE_INPUT}")
        print("Please ensure the raw data is downloaded to data/Full/")
        print("Expected structure:")
        for name, cfg in DATASET_CONFIGS.items():
            print(f"  {BASE_INPUT}/{cfg['folder']}/*.csv")
        exit(1)
    
    results = process_all_datasets(BASE_INPUT, BASE_OUTPUT, DATASET_CONFIGS)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    for dataset, result in results.items():
        print(f"\n{dataset}:")
        print(f"  Optimal n%: {result['optimal_n_percent']:.2f}%")
        print(f"  Batteries processed: {result['num_batteries']}")
        print(f"  Output dir: {os.path.dirname(result['processed_files'][0])}")
    
    # Save summary
    import json
    summary_path = os.path.join(BASE_OUTPUT, "preprocessing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")