import pandas as pd
import numpy as np
import re
import ast

def parse_array(s):
    cleaned = s.strip()
    cleaned = re.sub(r'array\(|\)', '', cleaned)
    cleaned = re.sub(r"dtype='<U19'", '', cleaned)
    cleaned = re.sub(r'dtype="<U19"', '', cleaned)
    cleaned = cleaned.replace('\n', ',').replace('\r', ',')
    cleaned = cleaned.replace('[[', '[').replace(']]', ']')
    cleaned = cleaned.replace('[,', '[').replace(',]', ']')
    cleaned = cleaned.replace(',,', ',')
    cleaned = cleaned.replace('array(', '').replace(')', '')
    try:
        return np.array(ast.literal_eval(cleaned)).flatten()
    except:
        nums = re.findall(r'[-+]?\d*\.\d+|\d+', cleaned)
        return np.array([float(n) for n in nums]) if nums else np.array([])

df = pd.read_csv('csv_converted/2C_battery-1.csv', nrows=10)
for i in range(10):
    cap = parse_array(df['capacity_Ah'].iloc[i])
    curr = parse_array(df['current_A'].iloc[i])
    volt = parse_array(df['voltage_V'].iloc[i])
    print(f'Cycle {i}:')
    print(f'  cap: min={cap.min():.4f}, max={cap.max():.4f}, non-zero={np.sum(cap>0)}, len={len(cap)}')
    print(f'  curr: min={curr.min():.4f}, max={curr.max():.4f}')
    print(f'  volt: min={volt.min():.4f}, max={volt.max():.4f}')
    print(f'  desc: {df["description"].iloc[i]}')
    print()