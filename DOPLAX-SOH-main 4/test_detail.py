import pandas as pd
import numpy as np
import re
import ast

def parse_numpy_array_string(array_str):
    cleaned = array_str.strip()
    cleaned = re.sub(r'array\(|\)', '', cleaned)
    cleaned = re.sub(r"dtype='<U19'", '', cleaned)
    cleaned = re.sub(r'dtype="<U19"', '', cleaned)
    cleaned = cleaned.replace('\\n', ',').replace('\n', ',')
    cleaned = cleaned.replace('[[', '[').replace(']]', ']')
    cleaned = cleaned.replace('[,', '[').replace(',]', ']')
    cleaned = cleaned.replace(',,', ',')
    cleaned = cleaned.replace('array(', '').replace(')', '')
    
    try:
        data = ast.literal_eval(cleaned)
        return np.array(data).flatten()
    except Exception as e:
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', cleaned)
        if numbers:
            return np.array([float(n) for n in numbers])
        return np.array([])

df = pd.read_csv(r'G:\ISDS\Knee-Aware-DOPLAX\DOPLAX-SOH-main 4\csv_converted\2C_battery-1.csv')

# Check a few cycles in detail
for idx in [0, 50, 100, 200, 300, 380, 385, 389]:
    cap_str = df['capacity_Ah'].iloc[idx]
    cap_array = parse_numpy_array_string(cap_str)
    volt_str = df['voltage_V'].iloc[idx]
    volt_array = parse_numpy_array_string(volt_str)
    curr_str = df['current_A'].iloc[idx]
    curr_array = parse_numpy_array_string(curr_str)
    power_str = df['power_Wh'].iloc[idx]
    power_array = parse_numpy_array_string(power_str)
    
    print(f'\n=== Cycle {idx} ===')
    print(f'  Description: {df["description"].iloc[idx]}')
    print(f'  Capacity: min={cap_array.min():.4f}, max={cap_array.max():.4f}, mean={cap_array.mean():.4f}')
    print(f'  Voltage: min={volt_array.min():.4f}, max={volt_array.max():.4f}')
    print(f'  Current: min={curr_array.min():.4f}, max={curr_array.max():.4f}')
    print(f'  Power: min={power_array.min():.4f}, max={power_array.max():.4f}')
    print(f'  Cap array sample: {cap_array[:5]} ... {cap_array[-5:]}')
    print(f'  Non-zero cap count: {np.sum(cap_array > 0)}')