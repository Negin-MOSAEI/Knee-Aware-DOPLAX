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
