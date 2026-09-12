import os
import torch
import numpy as np
import logging


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def eval_metrix(pred, true, eps=1e-8):
    """
    Computes standard regression metrics: MAE, MAPE, MSE, RMSE.
    """
    pred = np.array(pred).flatten()
    true = np.array(true).flatten()

    mae = np.mean(np.abs(pred - true))
    mape = np.mean(np.abs((true - pred) / (true + eps))) * 100.0
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)

    return [float(mae), float(mape), float(mse), float(rmse)]


def get_logger(log_dir, filename="train.log"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(log_dir)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, filename))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
