import numpy as np
import torch

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))) * 100.0

def calculate_monotonicity_loss(u_pred: torch.Tensor, u_tar: torch.Tensor = None) -> torch.Tensor:
    """
    Corrected Monotonicity Loss for capacity fade trajectories.
    Option 1: Direct non-increasing penalty over sequence dimension (N-1).
    Penalizes any positive predicted capacity change (diff_pred > 0).
    """
    relu = torch.nn.ReLU()
    if u_pred.dim() >= 2 and u_pred.shape[1] > 1:
        # Slicing over sequence dimension (N-1)
        diff_pred = u_pred[:, 1:] - u_pred[:, :-1]
        if u_tar is not None and u_tar.dim() >= 2 and u_tar.shape[1] > 1:
            diff_tar = u_tar[:, 1:] - u_tar[:, :-1]
            return torch.mean(relu(-1.0 * diff_pred * diff_tar))
        return torch.mean(relu(diff_pred))
    elif u_pred.dim() == 1 and u_pred.shape[0] > 1:
        diff_pred = u_pred[1:] - u_pred[:-1]
        return torch.mean(relu(diff_pred))
    else:
        return torch.tensor(0.0, device=u_pred.device)

