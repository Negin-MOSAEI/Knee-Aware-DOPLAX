import pandas as pd
df = pd.read_csv(r'G:\ISDS\Knee-Aware-DOPLAX\DOPLAX-SOH-main 4\csv_converted\2C_battery-1.csv', nrows=5)
for i in range(5):
    print(f'Row {i}:')
    desc = df['description'].iloc[i]
    print(f'  description: {desc}')
    curr = df['current_A'].iloc[i]
    print(f'  current_A: {curr[:50]}...')
    cap = df['capacity_Ah'].iloc[i]
    print(f'  capacity_Ah: {cap[:50]}...')
    print()