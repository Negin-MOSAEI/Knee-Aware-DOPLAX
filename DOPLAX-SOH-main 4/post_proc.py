"""
post_proc.py – Battery SOH Post-Processing Utilities
=====================================================
Smooths noisy SOH predictions from the KaDOPLAX fusion model using a
three-stage pipeline: jump-threshold clipping, Hampel filtering, and
Savitzky-Golay smoothing.
"""

import numpy as np
from scipy.signal import savgol_filter


def _hampel_filter(x: np.ndarray, window_size: int = 10, n_sigma: float = 3.0) -> np.ndarray:
    """Replace outliers with the local median (Hampel identifier)."""
    out = x.copy()
    half = window_size // 2
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        window = x[lo:hi]
        median = np.median(window)
        mad = 1.4826 * np.median(np.abs(window - median))
        if abs(x[i] - median) > n_sigma * mad:
            out[i] = median
    return out


def postprocess_capacity(
    capacity: np.ndarray,
    jump_thresh: float = 0.02,
    hampel_window: int = 100,
    savgol_window: int = 50,
    polyorder: int = 2,
    monotonic: bool = False,
    clip: tuple = None,
) -> np.ndarray:
    """Apply post-processing pipeline to a predicted SOH / capacity sequence.

    Parameters
    ----------
    capacity : np.ndarray
        Raw predicted capacity or SOH values.
    jump_thresh : float
        Maximum allowed step-to-step change (clipped if exceeded).
    hampel_window : int
        Window size for the Hampel outlier filter.
    savgol_window : int
        Window size for the Savitzky-Golay smoother.
    polyorder : int
        Polynomial order for the Savitzky-Golay filter.
    monotonic : bool
        If True, enforce monotonic (non-increasing) trend.
    clip : tuple of (lo, hi) or None
        Clip values to this range after smoothing.

    Returns
    -------
    np.ndarray – smoothed capacity / SOH.
    """
    x = np.array(capacity, dtype=np.float64)

    # Stage 1: jump-threshold clipping
    for i in range(1, len(x)):
        diff = x[i] - x[i - 1]
        if abs(diff) > jump_thresh:
            x[i] = x[i - 1] + np.sign(diff) * jump_thresh

    # Stage 2: Hampel filter (outlier removal)
    if hampel_window > 1:
        x = _hampel_filter(x, window_size=hampel_window)

    # Stage 3: Savitzky-Golay smoothing
    if savgol_window > polyorder and len(x) >= savgol_window:
        x = savgol_filter(x, savgol_window, polyorder)

    # Optional: enforce monotonic decrease
    if monotonic:
        for i in range(1, len(x)):
            if x[i] > x[i - 1]:
                x[i] = x[i - 1]

    # Optional: clip to range
    if clip is not None:
        x = np.clip(x, clip[0], clip[1])

    return x.astype(np.float32)
