import pandas as pd
import ast
import numpy as np

df = pd.read_csv(r'G:\ISDS\Knee-Aware-DOPLAX\DOPLAX-SOH-main 4\csv_converted\2C_battery-1.csv')

# Get the last value of capacity_Ah for each cycle (row)
capacities = []
for i in range(len(df)):
    cap_str = df.iloc[i]['capacity_Ah']
    # Parse the array string to get the last value
    try:
        # The format is like: [[0.]\n[0.021]\n[0.038]\n...]
        # Replace array() and dtype parts
        cleaned = cap_str.replace('array(', '').replace("dtype='<U19'", '').replace('dtype="<U19"', '')
        # Now it's like [[0.]\n[0.021]\n...] which is valid Python list
        cap_array = ast.literal_eval(cleaned)
        if len(cap_array) > 0:
            last_cap = cap_array[-1]
            capacities.append(last_cap)
    except Exception as e:
        print(f'Row {i} error: {e}')

capacities = np.array(capacities)
print(f'Number of cycles: {len(capacities)}')
print(f'First 5 capacities: {capacities[:5]}')
print(f'Last 5 capacities: {capacities[-5:]}')
print(f'Min capacity: {capacities.min():.4f}')
print(f'Max capacity: {capacities.max():.4f}')

# Normalize to get SOH
nominal_cap = 2.0  # XJTU nominal capacity
soh = capacities / nominal_cap
print(f'\nSOH range: {soh.min():.4f} to {soh.max():.4f}')