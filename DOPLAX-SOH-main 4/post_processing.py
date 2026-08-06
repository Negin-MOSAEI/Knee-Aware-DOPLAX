import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression
from scipy.signal import savgol_filter
import os

# ==== 1. Original Functions (mostly unchanged) ====

def apply_filter(y, window_size, mode="average"):
    """Applies a simple average or median filter."""
    y_filt = y.copy()
    for i in range(window_size - 1, len(y)):
        window = y[i - window_size + 1 : i + 1]
        if mode == "average":
            y_filt[i] = np.mean(window)
        elif mode == "median":
            y_filt[i] = np.median(window)
        else:
            raise ValueError("mode must be 'average' or 'median'")
    return y_filt

def hampel_filter(x, window=21, n_sigmas=3.0):
    """Median-based de-spike filter."""
    k = (window - 1) // 2
    x_pad = np.pad(x, (k, k), mode='edge')
    y = x.astype(float).copy()
    for i in range(len(x)):
        w = x_pad[i:i+window]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad == 0:
            continue
        thr = n_sigmas * 1.4826 * mad
        if abs(x[i] - med) > thr:
            y[i] = med
    return y

def find_segments(y, jump_thresh=0.01):
    """Detect upward jumps (segment boundaries)."""
    cuts = np.where(np.diff(y) > jump_thresh)[0] + 1
    return np.r_[0, cuts, len(y)]

def postprocess_capacity(y_pred,
                         jump_thresh=0.01,
                         hampel_window=21,
                         savgol_window=101,
                         polyorder=3,
                         isotonic=True,
                         clip=None):
    """
    Hampel de-spike → Isotonic Regression → Savitzky-Golay smoothing.

    Order matters: isotonic enforces monotonic degradation (piecewise constant),
    then SavGol smooths the staircase artifacts into a clean curve.

    isotonic : bool, default=True
        If True, applies IsotonicRegression(increasing=False) per segment
        to enforce non-increasing (monotonic degradation) constraint.

    If clip is not None, it should be a tuple (low, high) to clip final output.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    seg_bounds = find_segments(y_pred, jump_thresh)
    y_out = np.zeros_like(y_pred)

    for s, e in zip(seg_bounds[:-1], seg_bounds[1:]):
        seg = y_pred[s:e]

        # 1) de-spike
        seg = hampel_filter(seg, window=hampel_window)

        # 2) isotonic regression → enforce monotonic non-increasing
        if isotonic and len(seg) > 2:
            seg = IsotonicRegression(increasing=False).fit_transform(
                np.arange(len(seg)), seg
            )

        # 3) smooth → removes staircase artifacts from isotonic
        w = min(savgol_window, (len(seg)//2)*2 - 1)
        if w >= 5:
            if w % 2 == 0:
                w -= 1
            seg = savgol_filter(seg, window_length=w, polyorder=min(polyorder, w-1))

        y_out[s:e] = seg

    if clip is not None:
        lo, hi = clip
        return np.clip(y_out, lo, hi)
    return y_out

# ==== 2. Function to Process a Single CSV File and Return Metrics ====

def _mape(y_true, y_pred):
    """MAPE as a fraction (not %) with safe denominator."""
    denom = np.clip(np.abs(y_true), 1e-12, None)
    return np.mean(np.abs((y_true - y_pred) / denom))

def process_csv_file(csv_file_path):
    """
    Loads a single CSV, applies filtering, saves filtered data and plots,
    and returns a dictionary of performance metrics, including the group type.
    """
    # Parameters
    window_size = 15
    filter_type = "advanced"  # "average", "median", "both", or "advanced"

    # Infer group from parent folder
    parent_dir_name = os.path.basename(os.path.dirname(csv_file_path))
    group_type = "Other"
    if parent_dir_name.startswith("HUST-0"):
        group_type = "HUST-0"
    elif parent_dir_name.startswith("MIT-0"):
        group_type = "MIT-0"
    elif parent_dir_name.startswith("TJU-0"):
        group_type = "TJU-0"
    elif parent_dir_name.startswith("TJU-1"):
        group_type = "TJU-1"
    elif parent_dir_name.startswith("TJU-2"):
        group_type = "TJU-2"
    elif parent_dir_name.startswith("XJTU-2C"):
        group_type = "XJTU-2C"
    elif parent_dir_name.startswith("XJTU-3C"):
        group_type = "XJTU-3C"
    elif parent_dir_name.startswith("XJTU-R2.5"):
        group_type = "XJTU-R2.5"
    elif parent_dir_name.startswith("XJTU-R3"):
        group_type = "XJTU-R3"
    elif parent_dir_name.startswith("XJTU-RW"):
        group_type = "XJTU-RW"
    elif parent_dir_name.startswith("XJTU-satellite"):
        group_type = "XJTU-satellite"

    # Load CSV
    try:
        df = pd.read_csv(csv_file_path)
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
    except Exception as e:
        print(f"Error loading {csv_file_path}: {e}")
        return None

    results = {}

    # Raw metrics (MAPE is fractional, not percent)
    metrics = {
        "file_path": os.path.normpath(csv_file_path),
        "group": group_type,
        "rmse_raw": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape_raw": _mape(y_true, y_pred),
    }

    # Filtering branches
    if filter_type in ["average", "both"]:
        y_avg = apply_filter(y_pred, window_size, mode="average")
        results["average"] = {
            "filtered": y_avg,
            "rmse": np.sqrt(mean_squared_error(y_true, y_avg)),
            "mape": _mape(y_true, y_avg),
        }

    if filter_type in ["median", "both"]:
        y_med = apply_filter(y_pred, window_size, mode="median")
        results["median"] = {
            "filtered": y_med,
            "rmse": np.sqrt(mean_squared_error(y_true, y_med)),
            "mape": _mape(y_true, y_med),
        }

    if filter_type == "advanced":
        # IMPORTANT: no upper clipping; supports SOH > 1
        y_adv = postprocess_capacity(
            y_pred,
            jump_thresh=0.02,
            hampel_window=100,
            savgol_window=50,
            polyorder=3,
            monotonic=False,
            clip=None,  # keep unbounded; change to (0.0, 1.2) if you want a soft cap
        )
        results["advanced"] = {
            "filtered": y_adv,
            "rmse": np.sqrt(mean_squared_error(y_true, y_adv)),
            "mape": _mape(y_true, y_adv),
        }

    # Append filtered metrics to the dict
    for key, val in results.items():
        metrics[f"rmse_{key}"] = val['rmse']
        metrics[f"mape_{key}"] = val['mape']

    # Console report (no % sign)
    print(f"Processing: {csv_file_path}")
    print("===== Error Metrics =====")
    print(f"Group: {group_type}")
    print(f"RMSE before filtering : {metrics['rmse_raw']:.6f}")
    print(f"MAPE before filtering : {metrics['mape_raw']:.6f}")
    for key in results:
        print(f"\n-- {key.upper()} filter --")
        print(f"RMSE after filtering : {results[key]['rmse']:.6f}")
        print(f"MAPE after filtering : {results[key]['mape']:.6f}")

    # Save filtered data CSV
    output_df_dict = {"y_true": y_true, "y_pred": y_pred}
    for key, val in results.items():
        output_df_dict[f"y_pred_{key}"] = val["filtered"]
    output_df = pd.DataFrame(output_df_dict)

    output_folder = os.path.dirname(csv_file_path)
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_csv_path = os.path.join(output_folder, f"{base_name}_filtered.csv")
    output_df.to_csv(output_csv_path, index=False)
    print(f"Saved filtered data to: {output_csv_path}")

    # === High-Resolution Plotting ===
    plt.figure(figsize=(12, 6), dpi=300)  # High DPI for sharper plots
    plt.plot(y_true, label="Actual SOH", color="blue", linewidth=1.2)
    plt.plot(y_pred, label="Predicted SOH", color="red", alpha=0.7, linewidth=1.2)

    for key, val in results.items():
        if key == "advanced":
            plt.plot(val["filtered"],
                     label="Predicted SOH with post processing",
                     color="green", linewidth=1.2)
        elif key == "average":
            plt.plot(val["filtered"],
                     label="Predicted SOH (moving average)",
                     color="green", linewidth=1.2, alpha=0.7)
        elif key == "median":
            plt.plot(val["filtered"],
                     label="Predicted SOH (moving median)",
                     color="green", linewidth=1.2, alpha=0.7)

    plt.xlabel("Cycle Index")
    plt.ylabel("Normalized SOH")
    plt.legend()
    plt.tight_layout()

    output_plot_path = os.path.join(output_folder, f"{base_name}_plot.png")
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')  # save high-res
    print(f"Saved high-resolution plot to: {output_plot_path}")
    plt.close()

    return metrics

# ==== 3. Main Script to Iterate, Collect, Group, and Save Metrics ====

if __name__ == "__main__":
    root_dir = "."
    all_metrics = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith("test_outputs.csv"):
                file_path = os.path.join(dirpath, file)
                metrics = process_csv_file(file_path)
                if metrics:
                    all_metrics.append(metrics)
                print("-" * 50)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv("summary_of_all_metrics.csv", index=False)
        print("\n" + "=" * 50)
        print("Completed processing all files.")
        print("Summary of all metrics saved to summary_of_all_metrics.csv")
        print("=" * 50)

        cols_to_avg = ['rmse_raw', 'mape_raw', 'rmse_advanced', 'mape_advanced']
        available = [c for c in cols_to_avg if c in metrics_df.columns]
        grouped_results = metrics_df.groupby('group')[available].mean()

        print("\n" + " Averaged Losses by Group ".center(50, "=") + "\n")
        print(grouped_results)
        print("\n" + "=" * 50)

        grouped_results.to_csv("averaged_losses_by_group.csv")
        print("Averaged losses by group saved to averaged_losses_by_group.csv")
        print("=" * 50)
