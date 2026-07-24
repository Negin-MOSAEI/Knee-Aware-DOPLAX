import pandas as pd
import numpy as np
import os

data_dir = 'data/Processed/XJTU data'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

sep = '=' * 80
print(sep)
print('VERIFICATION REPORT: XJTU Knee-Point-Distance Datasets')
print(sep)

all_ok = True
summary_rows = []

for f in files:
    path = os.path.join(data_dir, f)
    df = pd.read_csv(path)
    name = f.replace('.csv', '')

    has_col = 'knee_point_distance' in df.columns
    n_rows = len(df)
    n_cols = len(df.columns)

    if has_col:
        kd = df['knee_point_distance']
        kd_min = kd.min()
        kd_max = kd.max()
        kd_mean = kd.mean()
        kd_std = kd.std()
        n_nan = int(kd.isna().sum())
        n_neg = int((kd < 0).sum())
        n_gt1 = int((kd > 1).sum())
        valid = bool((0.0 <= kd).all() and (kd <= 1.0).all())
        cap = df['capacity']
        n_cap_nan = int(cap.isna().sum())
        cap_min = cap.min()
        cap_max = cap.max()
    else:
        kd_min = kd_max = kd_mean = kd_std = float('nan')
        n_nan = n_rows
        n_neg = n_gt1 = 0
        valid = False
        cap_min = cap_max = float('nan')
        n_cap_nan = n_rows

    has_nan_any = bool(df.isna().any().any())

    if not has_col or not valid or n_nan > 0 or has_nan_any:
        all_ok = False

    status = 'PASS' if (has_col and valid and n_nan == 0 and not has_nan_any) else 'FAIL'
    tag = '[OK]' if status == 'PASS' else '[!!]'

    print()
    print(f'  {tag} {f}')
    print(f'      Columns ({n_cols}): {list(df.columns)}')
    print(f'      Rows: {n_rows}')
    if has_col:
        print(f'      knee_point_distance: min={kd_min:.6f}  max={kd_max:.6f}  mean={kd_mean:.6f}  std={kd_std:.6f}')
        print(f'      NaN={n_nan}  Negative={n_neg}  >1.0={n_gt1}  Range[0,1]={valid}')
    else:
        print(f'      ** knee_point_distance COLUMN MISSING **')
    print(f'      capacity: min={cap_min:.6f}  max={cap_max:.6f}  NaN={n_cap_nan}')
    print(f'      Any NaN in file: {has_nan_any}')

    summary_rows.append({
        'battery': name,
        'cycles': n_rows,
        'kd_min': kd_min,
        'kd_max': kd_max,
        'kd_mean': kd_mean,
        'status': status,
    })

print()
print(sep)
print('SUMMARY TABLE')
print(sep)
header = '{:<22s} {:>6s}  {:>8s}  {:>8s}  {:>8s}  {:>6s}'.format(
    'Battery', 'Cycles', 'KD min', 'KD max', 'KD mean', 'Status')
print(header)
print('-' * 64)
for r in summary_rows:
    row = '{:<22s} {:>6d}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>6s}'.format(
        r['battery'], r['cycles'], r['kd_min'], r['kd_max'], r['kd_mean'], r['status'])
    print(row)

print()
verdict = 'ALL CHECKS PASSED' if all_ok else 'ISSUES FOUND -- see details above'
print(sep)
print(f'OVERALL: {verdict}')
print(sep)
