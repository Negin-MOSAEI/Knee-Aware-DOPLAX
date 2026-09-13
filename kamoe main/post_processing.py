import os
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def apply_filter(y: np.ndarray, window_size: int = 15, mode: str = "average") -> np.ndarray:
    """
    Applies a moving average or median filter.
    """
    y_filt = np.asarray(y, dtype=float).copy()
    for i in range(window_size - 1, len(y_filt)):
        window = y_filt[i - window_size + 1 : i + 1]
        if mode == "average":
            y_filt[i] = np.mean(window)
        elif mode == "median":
            y_filt[i] = np.median(window)
        else:
            raise ValueError("mode must be 'average' or 'median'")
    return y_filt


def hampel_filter(x: np.ndarray, window: int = 21, n_sigmas: float = 3.0) -> np.ndarray:
    """
    Median-based de-spiking filter (Hampel identifier).
    Replaces outlier spikes beyond n_sigmas * 1.4826 * MAD with the local median.
    """
    k = (window - 1) // 2
    x_arr = np.asarray(x, dtype=float)
    x_pad = np.pad(x_arr, (k, k), mode='edge')
    y = x_arr.copy()

    for i in range(len(x_arr)):
        w = x_pad[i : i + window]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad == 0:
            continue
        thr = n_sigmas * 1.4826 * mad
        if abs(x_arr[i] - med) > thr:
            y[i] = med
    return y


def find_segments(y: np.ndarray, jump_thresh: float = 0.02) -> np.ndarray:
    """
    Detects sudden jump transitions (e.g. resting / capacity regeneration steps).
    """
    cuts = np.where(np.diff(y) > jump_thresh)[0] + 1
    return np.r_[0, cuts, len(y)]


def postprocess_capacity(
    y_pred: np.ndarray,
    jump_thresh: float = 0.02,
    hampel_window: int = 21,
    savgol_window: int = 51,
    polyorder: int = 2,
    monotonic: bool = True,
    clip: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Advanced capacity/SOH post-processing:
    1. Segment-wise Hampel de-spiking to eliminate stochastic measurement noise/spikes.
    2. Savitzky-Golay polynomial smoothing.
    3. Monotonic decreasing constraint via IsotonicRegression within each degradation segment.
    4. Supports values > 1.0 without hard upper clipping.
    """
    from sklearn.isotonic import IsotonicRegression

    y_pred = np.asarray(y_pred, dtype=float).flatten()
    seg_bounds = find_segments(y_pred, jump_thresh=jump_thresh)
    y_out = np.zeros_like(y_pred)

    for s, e in zip(seg_bounds[:-1], seg_bounds[1:]):
        seg = y_pred[s:e].copy()

        # 1. Hampel De-spike
        if len(seg) >= hampel_window:
            seg = hampel_filter(seg, window=hampel_window)
        elif len(seg) >= 5:
            w_hampel = (len(seg) // 2) * 2 + 1
            seg = hampel_filter(seg, window=w_hampel)

        # 2. Savitzky-Golay Smoothing
        w_sg = min(savgol_window, (len(seg) // 2) * 2 - 1)
        if w_sg >= 5:
            p_order = min(polyorder, w_sg - 1)
            seg = savgol_filter(seg, window_length=w_sg, polyorder=p_order)

        # 3. Monotonic Decreasing Constraint (allow > 1.0 by y_max=None)
        if monotonic and len(seg) > 1:
            ir = IsotonicRegression(increasing=False, y_min=0.0, y_max=None)
            seg = ir.fit_transform(np.arange(len(seg)), seg)

        y_out[s:e] = seg

    if clip is not None:
        lo, hi = clip
        return np.clip(y_out, lo, hi)

    return y_out

