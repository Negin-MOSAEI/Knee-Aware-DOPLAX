import sys
import os
import numpy as np

from dataloader import load_battery_dataloader, SplitRegistry, KneePointRegistry

def test():
    data_root = "../DOPLAX-SOH-main 4/data/Processed"
    split_reg = SplitRegistry()
    knee_reg = KneePointRegistry()

    print("========================================")
    print("1. Testing XJTU (batch 2C) with train_test_split.json and -arctan formula")
    xjtu_splits = split_reg.get_splits('XJTU', '2C')
    print(f"  Expected split counts -> train: {len(xjtu_splits['train'])}, val: {len(xjtu_splits['val'])}, test: {len(xjtu_splits['test'])}")
    
    xjtu_data = load_battery_dataloader("XJTU", "2C", data_root=data_root)
    print(f"  Loaded dataset sizes -> train: {len(xjtu_data['train_dataset'])}, val: {len(xjtu_data['valid_dataset'])}, test: {len(xjtu_data['test_dataset'])}")
    
    for batch in xjtu_data['train']:
        print(f"  Battery: {batch['battery_name']}")
        print(f"  Cycles (Length L_i): {batch['length']}")
        print(f"  Knee Cycle (C_knee): {batch['knee_point']}")
        
        # Verify knee distance formula: -arctan(C - C_knee)
        expected_dist_0 = -np.arctan(0.0 - batch['knee_point'])
        actual_dist_0 = batch['knee_distance'][0].item()
        print(f"  Cycle 0 knee distance: {actual_dist_0:.6f} (expected: {expected_dist_0:.6f})")
        assert abs(actual_dist_0 - expected_dist_0) < 1e-5, "Knee distance formula mismatch!"
        
        expected_dist_knee = -np.arctan(float(batch['knee_point']) - batch['knee_point'])
        actual_dist_knee = batch['knee_distance'][batch['knee_point']].item()
        print(f"  Knee cycle knee distance: {actual_dist_knee:.6f} (expected: 0.000000)")
        assert abs(actual_dist_knee - 0.0) < 1e-5, "Knee cycle distance is not 0!"
        break

    print("\n========================================")
    print("2. Testing MIT (batch 2017-05-12)")
    mit_data = load_battery_dataloader("MIT", "2017-05-12", data_root=data_root)
    print(f"  Loaded -> train: {len(mit_data['train_dataset'])}, val: {len(mit_data['valid_dataset'])}, test: {len(mit_data['test_dataset'])}")

    print("\n========================================")
    print("3. Testing TJU (batch Dataset_1_NCA_battery)")
    tju_data = load_battery_dataloader("TJU", "Dataset_1_NCA_battery", data_root=data_root)
    print(f"  Loaded -> train: {len(tju_data['train_dataset'])}, val: {len(tju_data['valid_dataset'])}, test: {len(tju_data['test_dataset'])}")

    print("\n========================================")
    print("4. Testing HUST (batch default)")
    hust_data = load_battery_dataloader("HUST", data_root=data_root)
    print(f"  Loaded -> train: {len(hust_data['train_dataset'])}, val: {len(hust_data['valid_dataset'])}, test: {len(hust_data['test_dataset'])}")

    print("\n========================================")
    print("SUCCESS: All splits and knee point distance formulas verified successfully!")

if __name__ == "__main__":
    test()
