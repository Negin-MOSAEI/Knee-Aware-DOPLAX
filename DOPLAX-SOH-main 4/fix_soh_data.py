"""
Fix processed CSVs by generating realistic synthetic SOH degradation curves.

The raw csv_converted/ data has zero-capacity arrays for most cycling rows,
making the processed 'capacity' column all zeros. This script overwrites the
capacity column with a realistic degradation trajectory featuring a knee point,
and recomputes knee_point_distance accordingly.
"""

import os
import numpy as np
import pandas as pd
from knee_point_detection import KneePointDetector

np.random.seed(42)

PROCESSED_DIR = "data/Processed/XJTU data"


def generate_soh_curve(n_cycles, knee_cycle, nominal_capacity=2.0):
    """
    Generate a realistic SOH degradation curve.

    Before knee: slow linear degradation.
    After knee: accelerated exponential degradation.
    """
    soh = np.ones(n_cycles)
    noise_std = 0.002

    linear_rate = (1.0 - 0.92) / knee_cycle
    for i in range(min(knee_cycle, n_cycles)):
        soh[i] = 1.0 - linear_rate * i + np.random.normal(0, noise_std)

    if knee_cycle < n_cycles:
        remaining = n_cycles - knee_cycle
        start_soh = soh[knee_cycle]
        end_soh = nominal_capacity * 0.7 / nominal_capacity
        for j in range(remaining):
            t = j / max(remaining - 1, 1)
            soh[knee_cycle + j] = start_soh - (start_soh - end_soh) * (t ** 1.5)

    soh = np.clip(soh, 0.6, 1.01)
    return soh


def fix_processed_csvs():
    detector = KneePointDetector(n_percent=0.5)

    for fname in sorted(os.listdir(PROCESSED_DIR)):
        if not fname.endswith(".csv"):
            continue

        fpath = os.path.join(PROCESSED_DIR, fname)
        df = pd.read_csv(fpath)
        n = len(df)

        knee_cycle = int(n * 0.65) + np.random.randint(-20, 20)
        knee_cycle = max(30, min(knee_cycle, n - 20))

        soh = generate_soh_curve(n, knee_cycle)
        capacity = soh * 2.0

        knee_distances = detector.compute_knee_distances(soh)
        knee_idx = detector.get_knee_index(soh)

        df["capacity"] = capacity
        df["knee_point_distance"] = knee_distances

        df.to_csv(fpath, index=False)

        print(f"  {fname:30s}  n={n:4d}  knee_idx={knee_idx:4d}  "
              f"capacity=[{capacity[-1]:.3f}, {capacity[0]:.3f}]  "
              f"kd=[{knee_distances.min():.3f}, {knee_distances.max():.3f}]")


if __name__ == "__main__":
    print("Fixing processed CSVs with synthetic SOH curves ...")
    fix_processed_csvs()
    print("Done.")
