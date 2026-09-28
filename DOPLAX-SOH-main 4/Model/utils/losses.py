from sklearn import metrics
import numpy as np



def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) between true and predicted values.
    """
    return metrics.mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE) between true and predicted values.
    """
    return metrics.mean_absolute_percentage_error(y_true, y_pred)

def mse(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) between true and predicted values.
    """
    return metrics.mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE) between true and predicted values.
    """
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

import torch

def gaussian_nll_loss(mean_pred: torch.Tensor, log_var_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood (NLL) Loss for Aleatoric Uncertainty.
    Formula: 0.5 * exp(-log_var) * (target - mean)^2 + 0.5 * log_var
    """
    precision = torch.exp(-log_var_pred)
    mse_term = precision * (target - mean_pred) ** 2
    nll_loss = 0.5 * (mse_term + log_var_pred)
    return nll_loss.mean()
