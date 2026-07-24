import pandas as pd
import numpy as np
import re
import ast

def parse_numpy_array_string(array_str):
    """Parse a string representation of a numpy array into an actual numpy array."""
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

capacities = []
max_caps = []
for i in range(len(df)):
    cap_str = df['capacity_Ah'].iloc[i]
    cap_array = parse_numpy_array_string(cap_str)
    if len(cap_array) > 0:
        # Get max capacity (end of discharge)
        max_cap = np.max(cap_array)
        max_caps.append(max_cap)
        # Last non-zero value might be end of cycle
        non_zero = cap_array[cap_array > 0]
        if len(non_zero) > 0:
            capacities.append(non_zero[-1])
        else:
            capacities.append(0)
    else:
        max_caps.append(0)
        capacities.append(0)

capacities = np.array(capacities)
max_caps = np.array(max_caps)

print(f'Number of cycles: {len(capacities)}')
print(f'First 10 last non-zero: {capacities[:10]}')
print(f'Last 10 last non-zero: {capacities[-10:]}')
print(f'First 10 max: {max_caps[:10]}')
print(f'Last 10 max: {max_caps[-10:]}')
print(f'Max range: {max_caps.min():.4f} to {max_caps.max():.4f}')

# Normalize to get SOH
nominal = 2.0
soh = max_caps / nominal
print(f'\nSOH range: {soh.min():.4f} to {soh.max():.4f}')
print(f'SOH first 10: {soh[:10]}')
print(f'SOH last 10: {soh[-10:]}')