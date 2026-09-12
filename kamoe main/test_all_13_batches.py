import sys
import os
import numpy as np
import torch

from dataloader import load_battery_dataloader, SplitRegistry, KneePointRegistry
from configs import get_config
from Model import (
    TemporalConvNet,
    DeepOPINN,
    OptimizationNetwork,
    FusionMLP
)

ALL_13_BATCHES = [
    ("HUST", "default"),
    ("MIT", "2017-05-12"),
    ("MIT", "2017-06-30"),
    ("MIT", "2018-04-12"),
    ("TJU", "Dataset_1_NCA_battery"),
    ("TJU", "Dataset_2_NCM_battery"),
    ("TJU", "Dataset_3_NCM_NCA_battery"),
    ("XJTU", "2C"),
    ("XJTU", "3C"),
    ("XJTU", "R2.5"),
    ("XJTU", "R3"),
    ("XJTU", "RW"),
    ("XJTU", "Sim_satellite"),
]

def run_test():
    data_root = "../DOPLAX-SOH-main 4/data/Processed"

    print("=" * 115)
    print(f"{'#':<3} | {'Dataset':<6} | {'Batch Name':<28} | {'DeepOPINN Params (hid, lr, alpha, beta)':<40} | {'Status':<8}")
    print("=" * 115)

    for idx, (dataset, batch) in enumerate(ALL_13_BATCHES, 1):
        # 1. Instantiate models with exact dataset parameters from last project
        tcn_model = TemporalConvNet(num_inputs=17, num_channels=[32, 64, 128, 64, 32], output_dim=1)
        deepopinn_model = DeepOPINN(config=dataset)
        lax_model = OptimizationNetwork(config=dataset)
        fusion_model = FusionMLP(config=dataset)

        tcn_model.eval()
        deepopinn_model.eval()
        lax_model.eval()
        fusion_model.eval()

        loaders = load_battery_dataloader(dataset, batch, data_root=data_root)
        
        # Test on sample battery
        for sample in loaders['train']:
            x = sample['x']
            y = sample['y']
            x1 = sample['x1']
            x2 = sample['x2']
            y1 = sample['y1']
            y2 = sample['y2']

            pred_knee = tcn_model(x)
            _, u_deepopinn = deepopinn_model(x)
            loss_dict = deepopinn_model.compute_loss(x1, x2, y1, y2)
            u_lax = lax_model(x=x[:, :-1], t=x[:, -1:])
            u_final = fusion_model(u_deepopinn=u_deepopinn, u_lax=u_lax, knee_distance=pred_knee)

            assert u_final.shape == y.shape
            break

        param_str = f"hid={deepopinn_model.hidden_dim}, lr={deepopinn_model.lr:.1e}, a={deepopinn_model.pde_weight:.2f}, b={deepopinn_model.monotone_weight:.2f}"
        print(f"{idx:<3} | {dataset:<6} | {batch:<28} | {param_str:<40} | PASSED")

    print("=" * 115)
    print("SUCCESS: All 13 batches tested with dataset-specific parameters from the last project!")

if __name__ == "__main__":
    run_test()
