"""
Test script for knee point preprocessing on sample data
"""
import sys
sys.path.insert(0, r'G:\ISDS\Knee-Aware-DOPLAX\DOPLAX-SOH-main 4')

from knee_point_preprocessing import (
    parse_numpy_array_string,
    extract_capacity_trajectory,
    compute_soh,
    KneePointDetector,
    optimize_n_percent
)
import pandas as pd
import numpy as np

# Test on one file
test_file = r'G:\ISDS\Knee-Aware-DOPLAX\DOPLAX-SOH-main 4\csv_converted\2C_battery-1.csv'
print(f"Testing on: {test_file}")

df = pd.read_csv(test_file)
print(f"Raw data shape: {df.shape}")

# Extract capacity trajectory
capacities = extract_capacity_trajectory(df, 'capacity_Ah')
print(f"Capacities shape: {capacities.shape}")
print(f"First 10: {capacities[:10]}")
print(f"Last 10: {capacities[-10:]}")

# Compute SOH
nominal_cap = 2.0
soh = compute_soh(capacities, nominal_cap)
print(f"\nSOH range: {soh.min():.4f} to {soh.max():.4f}")
print(f"SOH first 10: {soh[:10]}")
print(f"SOH last 10: {soh[-10:]}")

# Test knee point detection
detector = KneePointDetector(n_percent=2.0)
knee_idx = detector.detect_knee_point(soh)
print(f"\nKnee point index: {knee_idx}")
print(f"Knee point SOH: {soh[knee_idx]:.4f}")

# Compute distances
distances = detector.compute_knee_distances(soh)
print(f"Distance range: {distances.min():.4f} to {distances.max():.4f}")
print(f"Distances at knee_idx: {distances[knee_idx]:.4f}")

# Test optimization
print("\nTesting optimization...")
all_soh = [soh]
optimal_n = optimize_n_percent(all_soh)
print(f"Optimal n_percent: {optimal_n:.2f}")

print("\nTest passed!")