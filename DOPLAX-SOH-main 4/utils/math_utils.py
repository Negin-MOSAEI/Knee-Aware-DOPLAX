import numpy as np
import torch

def calculate_kpd(current_cycle: int, knee_point_cycle: int) -> float:
    """
    Calculates KPD using the negative arctan formulation: -arctan(current_cycle - knee_point_cycle).
    This ensures values are positive pre-knee and negative post-knee.
    """
    return float(-np.arctan(current_cycle - knee_point_cycle))

def tst_cold_start_kpd(current_cycle: int, knee_point_cycle: int = 10000000) -> float:
    """
    Mathematical fallback logic for cycles 1-39 when the TST cannot operate.
    Linearly interpolates from 1.5 down to the exact KPD at cycle 40.
    """
    if current_cycle >= 40:
        return calculate_kpd(current_cycle, knee_point_cycle)
        
    start_val = np.pi / 2
    end_val = np.pi / 2 - 0.1
    
    # Linear interpolation over 39 steps (cycle 1 to 39)
    # Cycle 1 -> fraction = 0 -> start_val
    # Cycle 39 -> fraction = 1 -> end_val
    # So fraction = (current_cycle - 1) / 38.0
    fraction = (current_cycle - 1) / 38.0
    val = start_val - fraction * (start_val - end_val)
    return float(val)
