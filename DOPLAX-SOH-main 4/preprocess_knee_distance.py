"""
Knee Point Distance Preprocessing Script

This script processes raw battery cycling data to:
1. Extract per-cycle SOH trajectories from array-format data
2. Detect knee points using automated threshold optimization
3. Compute knee_point_distance for each cycle
4. Save processed CSV files compatible with existing data loaders
"""

import numpy as np
import pandas as pd
import os
import ast
import re
import json
import argparse
from typing import List, Tuple, Optional, Dict
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


def parse_numpy_array_string(array_str: str) -> np.ndarray:
    """Parse string representation of numpy array into actual array."""
    cleaned = array_str.strip()
    cleaned = re.sub(r'array\(|\)', '', cleaned)
    cleaned = re.sub(r"dtype='<U19'", '', cleaned)
    cleaned = re.sub(r'dtype="<U19"', '', cleaned)
    cleaned = cleaned.replace('\n', ',').replace('\r', ',')
    cleaned = cleaned.replace('[[', '[').replace(']]', ']')
    cleaned = cleaned.replace('[,', '[').replace(',]', ']')
    cleaned = cleaned.replace(',,', ',')
    cleaned = cleaned.replace('array(', '').replace(')', '')
    
    try:
        data = ast.literal_eval(cleaned)
        return np.array(data).flatten()
    except Exception:
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', cleaned)
        if numbers:
            return np.array([float(n) for n in numbers])
        return np.array([])


def extract_cycle_capacity(capacity_str: str) -> float:
    """Extract end-of-cycle capacity from capacity_Ah array string."""
    cap_array = parse_numpy_array_string(capacity_str)
    if len(cap_array) == 0:
        return np.nan
    # Use maximum value as cycle capacity (end of discharge)
    return float(np.max(cap_array))


def extract_soh_trajectory(df: pd.DataFrame, capacity_col: str = 'capacity_Ah', 
                           nominal_capacity: float = 2.0) -> np.ndarray:
    """Extract SOH trajectory from DataFrame."""
    capacities = []
    for i in range(len(df)):
        cap_str = df.iloc[i][capacity_col]
        cap = extract_cycle_capacity(cap_str)
        capacities.append(cap)
    
    capacities = np.array(capacities)
    # Forward/backward fill NaN values
    capacities = pd.Series(capacities).ffill().bfill().values
    
    # Normalize to get SOH
    soh = capacities / nominal_capacity
    return soh


class KneePointDetector:
    """Professional knee point detection for battery degradation trajectories."""
    
    def __init__(self, n_percent: float = 2.0, min_linear_points: int = 10, 
                 max_linear_fraction: float = 0.5):
        self.n_percent = n_percent
        self.min_linear_points = min_linear_points
        self.max_linear_fraction = max_linear_fraction
        
    def detect_knee_point(self, soh_values: np.ndarray) -> int:
        """Detect knee point index in SOH trajectory."""
        n_cycles = len(soh_values)
        if n_cycles < self.min_linear_points * 2:
            return n_cycles // 2
        
        max_linear_end = int(n_cycles * self.max_linear_fraction)
        max_linear_end = max(max_linear_end, self.min_linear_points)
        
        best_knee = n_cycles // 2
        best_score = float('inf')
        
        for linear_end in range(self.min_linear_points, max_linear_end + 1):
            x_linear = np.arange(linear_end).reshape(-1, 1)
            y_linear = soh_values[:linear_end]
            
            model = LinearRegression()
            model.fit(x_linear, y_linear)
            
            y_fitted_all = model.predict(np.arange(n_cycles).reshape(-1, 1))
            
            residuals = np.abs(soh_values - y_fitted_all)
            threshold = (self.n_percent / 100.0) * y_fitted_all
            
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
        """Evaluate quality of detected knee point. Lower = better."""
        n_cycles = len(soh_values)
        
        x_before = np.arange(knee_idx).reshape(-1, 1)
        y_before = soh_values[:knee_idx]
        if len(y_before) > 1:
            model_before = LinearRegression()
            model_before.fit(x_before, y_before)
            r2_before = model_before.score(x_before, y_before)
        else:
            r2_before = 0.0
        
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
        
        score = (1.0 - r2_before) + r2_after + 0.1 * slope_change
        return score
    
    def compute_knee_distances(self, soh_values: np.ndarray) -> np.ndarray:
        """Compute knee point distance: 1.0 at start -> 0.0 at knee -> 0.0 after."""
        knee_idx = self.detect_knee_point(soh_values)
        n_cycles = len(soh_values)
        if knee_idx == 0:
            return np.zeros(n_cycles)
        distances = (knee_idx - np.arange(n_cycles, dtype=float)) / knee_idx
        return np.clip(distances, 0.0, 1.0)
    
    def get_knee_index(self, soh_values: np.ndarray) -> int:
        return self.detect_knee_point(soh_values)


def optimize_n_percent(all_soh_trajectories: List[np.ndarray], 
                       n_range: Tuple[float, float] = (0.5, 10.0),
                       n_steps: int = 20) -> float:
    """Automated optimization to find optimal n% threshold."""
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
    """Evaluate consistency of detected knee point."""
    n = len(soh_values)
    if knee_idx <= 1 or knee_idx >= n - 1:
        return 100.0
    
    x_before = np.arange(knee_idx).reshape(-1, 1)
    y_before = soh_values[:knee_idx]
    model_before = LinearRegression().fit(x_before, y_before)
    r2_before = model_before.score(x_before, y_before)
    
    x_after = np.arange(knee_idx, n).reshape(-1, 1)
    y_after = soh_values[knee_idx:]
    model_after = LinearRegression().fit(x_after, y_after)
    r2_after = model_after.score(x_after, y_after)
    
    slope_before = model_before.coef_[0]
    slope_after = model_after.coef_[0]
    
    slope_ratio = abs(slope_after / slope_before) if abs(slope_before) > 1e-6 else 1.0
    
    score = (1.0 - r2_before) + r2_after + abs(1.0 - slope_ratio)
    return score


def process_battery_file(input_path: str, output_path: str,
                         nominal_capacity: float = 2.0,
                         n_percent: Optional[float] = None,
                         auto_optimize: bool = False,
                         all_trajectories: Optional[List[np.ndarray]] = None) -> Dict:
    """Process a single battery CSV file and add knee_point_distance."""
    df = pd.read_csv(input_path)
    
    if 'capacity_Ah' not in df.columns:
        raise ValueError(f"CSV must contain 'capacity_Ah' column. Found: {df.columns.tolist()}")
    
    # Extract SOH trajectory
    soh = extract_soh_trajectory(df, 'capacity_Ah', nominal_capacity)
    
    # Determine n_percent
    if auto_optimize and all_trajectories is not None:
        n_percent = optimize_n_percent(all_trajectories)
        print(f"  Auto-optimized n_percent = {n_percent:.2f}%")
    elif n_percent is None:
        n_percent = 2.0
    
    # Detect knee point and compute distances
    detector = KneePointDetector(n_percent=n_percent)
    knee_distances = detector.compute_knee_distances(soh)
    knee_idx = detector.get_knee_index(soh)
    
    # Add new columns to DataFrame
    df['soh'] = soh
    df['knee_point_distance'] = knee_distances
    df['knee_point_index'] = knee_idx
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed DataFrame
    df.to_csv(output_path, index=False)
    
    return {
        'input_file': input_path,
        'output_file': output_path,
        'num_cycles': len(df),
        'knee_index': int(knee_idx),
        'knee_soh': float(soh[knee_idx]) if knee_idx < len(soh) else None,
        'distance_range': [float(knee_distances.min()), float(knee_distances.max())],
        'n_percent_used': float(n_percent)
    }


def process_xjtu_dataset(input_base: str, output_base: str,
                         nominal_capacity: float = 2.0) -> Dict:
    """Process XJTU dataset from csv_converted structure."""
    print(f"\n{'='*60}")
    print(f"Processing XJTU dataset")
    print(f"{'='*60}")
    
    # XJTU files are in csv_converted/ with batch names in filenames
    input_dir = os.path.join(input_base, 'csv_converted')
    if not os.path.exists(input_dir):
        input_dir = input_base  # fallback
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} battery files")
    
    # Extract all SOH trajectories first for global optimization
    all_soh = []
    file_paths = []
    
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        try:
            df = pd.read_csv(input_path)
            if 'capacity_Ah' in df.columns:
                soh = extract_soh_trajectory(df, 'capacity_Ah', nominal_capacity)
                all_soh.append(soh)
                file_paths.append(input_path)
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")
    
    # Optimize n_percent globally
    optimal_n = optimize_n_percent(all_soh)
    print(f"Optimal n_percent for XJTU: {optimal_n:.2f}%")
    
    # Process each file with optimal n_percent
    output_dir = os.path.join(output_base, 'XJTU data')
    os.makedirs(output_dir, exist_ok=True)
    
    processed_files = []
    results = []
    
    for csv_file, input_path, soh in zip(csv_files, file_paths, all_soh):
        output_path = os.path.join(output_dir, csv_file)
        result = process_battery_file(
            input_path, output_path,
            nominal_capacity=nominal_capacity,
            n_percent=optimal_n,
            auto_optimize=False
        )
        processed_files.append(output_path)
        results.append(result)
        print(f"  Processed: {csv_file} -> knee_idx={result['knee_index']}, "
              f"knee_SOH={result['knee_soh']:.4f}, "
              f"dist_range=[{result['distance_range'][0]:.4f}, {result['distance_range'][1]:.4f}]")
    
    return {
        'dataset': 'XJTU',
        'optimal_n_percent': float(optimal_n),
        'num_batteries': len(processed_files),
        'processed_files': processed_files,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocess knee point distance for battery data')
    parser.add_argument('--input_dir', type=str, default='.',
                        help='Base input directory containing csv_converted/')
    parser.add_argument('--output_dir', type=str, default='data/Processed',
                        help='Base output directory for processed data')
    parser.add_argument('--dataset', type=str, default='XJTU',
                        choices=['XJTU', 'ALL'],
                        help='Dataset to process')
    parser.add_argument('--nominal_capacity', type=float, default=2.0,
                        help='Nominal capacity for SOH normalization')
    
    args = parser.parse_args()
    
    print("="*60)
    print("KNEE POINT DISTANCE PREPROCESSING PIPELINE")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset: {args.dataset}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    if args.dataset in ['XJTU', 'ALL']:
        result = process_xjtu_dataset(
            args.input_dir, 
            args.output_dir, 
            args.nominal_capacity
        )
        all_results['XJTU'] = result
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'knee_distance_preprocessing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    for dataset, result in all_results.items():
        print(f"\n{dataset}:")
        print(f"  Optimal n%: {result['optimal_n_percent']:.2f}%")
        print(f"  Batteries processed: {result['num_batteries']}")
        print(f"  Output directory: {os.path.dirname(result['processed_files'][0]) if result['processed_files'] else 'N/A'}")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()