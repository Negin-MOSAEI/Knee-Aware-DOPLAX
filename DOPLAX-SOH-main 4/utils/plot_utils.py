import matplotlib.pyplot as plt
import os
from typing import List

def plot_soh_with_knee(
    cycles: List[int],
    soh: List[float],
    knee_point: int,
    battery_id: str,
    dataset_name: str,
    batch_name: str,
    save_dir: str,
    is_train: bool = True
):
    """
    Plots the raw SOH degradation alongside the theoretical KPD curve 
    and the identified knee point.
    
    Args:
        cycles (List[int]): List of cycle numbers.
        soh (List[float]): List of State of Health values.
        knee_point (int): The algorithmically detected knee point cycle.
        battery_id (str): Identifier for the battery.
        dataset_name (str): Name of the dataset (e.g., HUST, MIT).
        batch_name (str): Name of the batch (e.g., NCA, 2C).
        save_dir (str): Directory to save the plot.
        is_train (bool): Indicates if the battery is part of the train set.
    """
    import numpy as np
    
    # Calculate theoretical KPD curve for plotting
    kpd_curve = [float(np.arctan(c - knee_point)) for c in cycles]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Set background color based on Train/Test split
    bg_color = '#f0f8ff' if is_train else '#fff0f5'  # AliceBlue for train, LavenderBlush for test
    ax1.set_facecolor(bg_color)
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Cycles')
    ax1.set_ylabel('State of Health (SOH)', color=color1)
    ax1.plot(cycles, soh, color=color1, label='Raw SOH')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Mark the knee point
    ax1.axvline(x=knee_point, color='red', linestyle='--', label=f'Knee Point ({knee_point})')
    
    # Secondary axis for KPD
    ax2 = ax1.twinx()  
    color2 = 'tab:green'
    ax2.set_ylabel('Theoretical KPD (arctan)', color=color2)  
    ax2.plot(cycles, kpd_curve, color=color2, linestyle=':', label='KPD Curve')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title and legends
    set_type = "Train" if is_train else "Test"
    plt.title(f'SOH and Knee Point - {dataset_name} ({batch_name}) - {battery_id} [{set_type}]')
    
    fig.tight_layout() 
    
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{battery_id}_soh_knee.png")
    plt.savefig(save_path)
    plt.close()
