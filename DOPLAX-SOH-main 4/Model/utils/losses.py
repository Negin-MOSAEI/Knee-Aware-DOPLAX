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

def evaluate_new_architecture(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device, dataset_name: str = "Test"):
    """
    Evaluates the newly upgraded KaDOPLAX architecture.
    Calculates SOH Accuracy (RMSE, MAE) and Uncertainty Coverage (95% CI).
    """
    model.eval()
    
    all_true_soh = []
    all_pred_soh = []
    all_total_std = []
    
    print(f"Evaluating model on {len(test_loader.dataset)} samples...")
    
    with torch.no_grad():
        for features, kpd, cycle_t, target_soh in test_loader:
            features = features.to(device)
            kpd = kpd.to(device)
            cycle_t = cycle_t.to(device)
            target_soh_np = target_soh.cpu().numpy().flatten()
            
            # Use the new predict_with_uq method
            mean_pred, total_std = model.predict_with_uq(features, kpd, cycle_t, n_samples=30)
            
            mean_pred_np = mean_pred.cpu().numpy().flatten()
            total_std_np = total_std.cpu().numpy().flatten()
            
            all_true_soh.extend(target_soh_np)
            all_pred_soh.extend(mean_pred_np)
            all_total_std.extend(total_std_np)
            
    all_true_soh = np.array(all_true_soh)
    all_pred_soh = np.array(all_pred_soh)
    all_total_std = np.array(all_total_std)
    
    # 1. SOH Accuracy
    rmse_val = rmse(all_true_soh, all_pred_soh)
    mae_val = mae(all_true_soh, all_pred_soh)
    
    # 2. Uncertainty Coverage (95% Confidence Interval)
    lower_bound = all_pred_soh - 1.96 * all_total_std
    upper_bound = all_pred_soh + 1.96 * all_total_std
    
    covered = (all_true_soh >= lower_bound) & (all_true_soh <= upper_bound)
    coverage_pct = np.mean(covered) * 100.0
    
    # Note: Knee Point Error (Absolute Cycle Difference) requires the outputs from the Phase 2 KneeTCN
    # For fusion evaluation, we focus on the downstream SOH reconstruction and UQ.
    
    # Print clean table
    print("\n" + "=" * 50)
    print(f"📊 ARCHITECTURAL EVALUATION ({dataset_name})")
    print("=" * 50)
    print(f" 1. SOH RMSE                  : {rmse_val:.5f}")
    print(f" 2. SOH MAE                   : {mae_val:.5f}")
    print(f" 3. 95% CI UQ Coverage        : {coverage_pct:.2f}%")
    print("=" * 50 + "\n")
    
    return rmse_val, mae_val, coverage_pct
