import numpy as np
import torch

def calculate_kpd(current_cycle: int, knee_point_cycle: int) -> float:
    """
    Calculates KPD using the negative arctan formulation: -arctan(current_cycle - knee_point_cycle).
    This ensures values are positive pre-knee and negative post-knee.
    """
    return float(-np.arctan(current_cycle - knee_point_cycle))

def tst_cold_start_kpd(current_cycle: int) -> float:
    """
    Mathematical fallback logic for cycles 1-39 when the TST cannot operate.
    """
    return 1.5 
