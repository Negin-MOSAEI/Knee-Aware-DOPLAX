import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, '.')
from knee_point_detection import (
    KneePointDetector, optimize_n_percent,
    extract_capacity_trajectory, compute_soh, parse_numpy_array_string,
)

raw_dir = 'csv_converted'
files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.csv')])
NOMINAL_CAP = 2.0  # XJTU-SY nominal capacity (Ah)

sep = '=' * 80
print(sep)
print('DETECTION SUMMARY: Knee-Point Detection on XJTU Batteries')
print(sep)

all_soh = []

for f in files:
    path = os.path.join(raw_dir, f)
    df_raw = pd.read_csv(path)
    name = f.replace('.csv', '')
    n_rows = len(df_raw)

    cap = extract_capacity_trajectory(df_raw, "capacity_Ah")
    soh = compute_soh(cap, NOMINAL_CAP)
    all_soh.append((name, n_rows, soh))

# Optimize n across ALL batteries
n_opt = optimize_n_percent([s for _, _, s in all_soh])

detector = KneePointDetector(n_percent=n_opt)

print()
print(f'  Optimized n across all batteries: {n_opt:.2f}%')
print()
print('{:<22s} {:>6s}  {:>10s}  {:>8s}  {:>8s}'.format(
    'Battery', 'Cycles', 'Knee idx', 'SOH@knee', 'SOH[end]'))
print('-' * 68)

for name, n_rows, soh in all_soh:
    ki = detector.detect_knee_point(soh)

    soh_knee = soh[ki] if ki < len(soh) else float('nan')
    soh_end = soh[-1]

    print('{:<22s} {:>6d}  {:>10d}  {:>8.4f}  {:>8.4f}'.format(
        name, n_rows, ki, soh_knee, soh_end))

print()
print(sep)
print('NOTES')
print(sep)
print('  - knee_point_index: 0-based index of detected knee in the SOH trajectory')
print('  - n (%): optimized percentage threshold (lowest mean consistency score)')
print('  - SOH@knee: SOH value at the detected knee point')
print('  - SOH[end]: SOH value at the final cycle')
print('  - SOH derived from last value of capacity_Ah array per cycle / nominal capacity')
print(sep)
