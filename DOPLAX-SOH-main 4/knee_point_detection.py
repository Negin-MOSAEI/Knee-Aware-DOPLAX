"""
Knee Point Detection & Distance Calculation Module
===================================================

Single source-of-truth for knee point detection in battery SOH degradation curves.

Algorithm
---------
1. For each battery cell's SOH curve (cycle 0 → EOL):
   a. Isolate the initial linear degradation phase.
   b. Fit a linear regression model to that phase.
2. Detect knee_point_index as the first cycle where:
       |actual_SOH − fitted_SOH| >= (n / 100) × fitted_SOH
3. For every cycle i, compute the normalised distance:
       knee_point_distance[i] = (knee_index − i) / knee_index   if i < knee_index
                                 0.0                              if i >= knee_index
   Semantics: 1.0 at cycle 0 (start of life), decreases linearly to
   0.0 at the knee, then stays 0.0 at and after the knee.
4. Optimise n across all batteries via grid search + consistency scoring.

Usage
-----
    from knee_point_detection import KneePointDetector, optimize_n_percent

    detector = KneePointDetector(n_percent=2.0)
    distances = detector.compute_knee_distances(soh_values)

    optimal_n = optimize_n_percent(all_soh_trajectories)
"""

import numpy as np
import pandas as pd
import os
import json
import re
import ast
from typing import List, Tuple, Optional, Dict
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
#  Utility: parse array-encoded CSV cells
# ---------------------------------------------------------------------------

def parse_numpy_array_string(array_str: str) -> np.ndarray:
    """
    Parse a string-encoded numpy array (as stored in CSV cells) into a flat
    ``np.ndarray``.

    Handles formats such as::

        "[[0.]\n[0.021]\n[0.038]\n...]"
        "array([0., 0.021, 0.038], dtype='<U19')"
    """
    cleaned = array_str.strip()
    # Strip wrapping artefacts
    cleaned = re.sub(r"array\(|\)", "", cleaned)
    cleaned = re.sub(r"dtype=['\"]<U19['\"]", "", cleaned)
    # Normalise whitespace / newlines → commas
    cleaned = cleaned.replace("\n", ",").replace("\r", ",")
    # Collapse nested brackets to flat list
    cleaned = cleaned.replace("[,", "[").replace(",]", "]")
    cleaned = cleaned.replace(",,", ",")
    try:
        data = ast.literal_eval(cleaned)
        return np.array(data).flatten()
    except Exception:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", cleaned)
        if numbers:
            return np.array([float(n) for n in numbers])
        return np.array([])


# ---------------------------------------------------------------------------
#  Utility: extract SOH trajectory from a DataFrame
# ---------------------------------------------------------------------------

def extract_capacity_trajectory(
    df: pd.DataFrame,
    capacity_col: str = "capacity_Ah",
) -> np.ndarray:
    """
    Extract the **end-of-cycle** capacity for every row (cycle) in *df*.

    Each cell in *capacity_col* contains a numpy-array string representing the
    capacity profile over one cycle.  The *last* value of that array is taken
    as the end-of-cycle (discharge) capacity.

    Returns a 1-D array of length ``len(df)`` with NaN filled forward/backward
    where parsing fails.
    """
    capacities: List[float] = []
    for i in range(len(df)):
        cap_str = df.iloc[i][capacity_col]
        cap_array = parse_numpy_array_string(cap_str)
        if len(cap_array) > 0:
            capacities.append(float(cap_array[-1]))
        else:
            capacities.append(np.nan)
    capacities = np.array(capacities)
    capacities = pd.Series(capacities).ffill().bfill().values
    return capacities


def compute_soh(capacities: np.ndarray, nominal_capacity: float) -> np.ndarray:
    """Normalise raw capacities to State-of-Health ∈ [0, 1]."""
    return capacities / nominal_capacity


def knee_aware_branch_weights(kd):
    """
    Compute branch weights from knee_point_distance.

    kd[i] = 1.0 at cycle 0 (start of life), decreases to 0.0 at knee,
    stays 0.0 at and after the knee.

    Weighting rules:
        w_lax  = kd        (LAX dominates early, drops to zero at knee)
        w_pinn = 1.0 - kd  (PINN compensates as LAX fades)

    Args:
        kd: tensor or array of shape (B,) or (B, 1), values in [0, 1]

    Returns:
        w_lax, w_pinn: same shape as kd, sum to 1.0
    """
    import torch
    w_lax = torch.as_tensor(kd, dtype=torch.float32)
    w_pinn = 1.0 - w_lax
    return w_lax, w_pinn


def extract_soh_trajectory(
    df: pd.DataFrame,
    capacity_col: str = "capacity_Ah",
    nominal_capacity: float = 2.0,
) -> np.ndarray:
    """
    Convenience wrapper: extract capacities *and* convert to SOH in one call.
    """
    capacities = extract_capacity_trajectory(df, capacity_col)
    return compute_soh(capacities, nominal_capacity)


# ---------------------------------------------------------------------------
#  Core: KneePointDetector
# ---------------------------------------------------------------------------

class KneePointDetector:
    """
    Detect the knee point in a battery SOH degradation curve.

    Parameters
    ----------
    n_percent : float
        Threshold *n* (in %).  Knee is the first cycle where
        ``|actual − fitted| >= (n/100) × fitted``.
    min_linear_points : int
        Fewest cycles used to fit the initial linear region.
    max_linear_fraction : float
        Largest fraction of the trajectory eligible as the linear region
        (e.g. 0.5 = first half of cycles).
    """

    def __init__(
        self,
        n_percent: float = 2.0,
        min_linear_points: int = 10,
        max_linear_fraction: float = 0.5,
    ):
        self.n_percent = n_percent
        self.min_linear_points = min_linear_points
        self.max_linear_fraction = max_linear_fraction

    # ----- public API -----------------------------------------------------

    def detect_knee_point(self, soh_values: np.ndarray) -> int:
        """
        Return the index of the knee point in *soh_values*.

        The method tries every candidate endpoint for the linear region
        (from ``min_linear_points`` up to ``max_linear_fraction × N``), fits a
        linear model, and picks the (linear_end, knee) pair that yields the
        best quality score.
        """
        n = len(soh_values)
        if n < self.min_linear_points * 2:
            return n // 2

        max_le = max(int(n * self.max_linear_fraction), self.min_linear_points)

        best_knee = n // 2
        best_score = float("inf")

        for le in range(self.min_linear_points, max_le + 1):
            x_le = np.arange(le).reshape(-1, 1)
            y_le = soh_values[:le]

            model = LinearRegression().fit(x_le, y_le)
            y_fitted = model.predict(np.arange(n).reshape(-1, 1))

            residuals = np.abs(soh_values - y_fitted)
            threshold = (self.n_percent / 100.0) * y_fitted

            candidates = np.where(residuals >= threshold)[0]
            candidates = candidates[candidates >= le]

            if len(candidates) > 0:
                knee_idx = int(candidates[0])
                score = self._score_knee(soh_values, knee_idx, model)
                if score < best_score:
                    best_score = score
                    best_knee = knee_idx

        return best_knee

    def compute_knee_distances(self, soh_values: np.ndarray) -> np.ndarray:
        """
        Compute normalised knee-point distance for every cycle.

        Definition
        ----------
        Cycle 0 (start of life) → 1.0.
        Approaching knee point  → decreases smoothly to 0.0.
        At and after knee point → 0.0.

        Returns
        -------
        distances : np.ndarray, shape (len(soh_values),)
        """
        knee_idx = self.detect_knee_point(soh_values)
        n = len(soh_values)
        if knee_idx == 0:
            return np.zeros(n)
        distances = (knee_idx - np.arange(n, dtype=float)) / knee_idx
        return np.clip(distances, 0.0, 1.0)

    def get_knee_index(self, soh_values: np.ndarray) -> int:
        """Alias for :pymethod:`detect_knee_point`."""
        return self.detect_knee_point(soh_values)

    # ----- internal scoring -----------------------------------------------

    @staticmethod
    def _score_knee(
        soh_values: np.ndarray,
        knee_idx: int,
        linear_model: LinearRegression,
    ) -> float:
        """
        Rate a candidate knee point (lower is better).

        Components
        ----------
        * **(1 − R²_before)** — penalises poor linear fit before the knee.
        * **R²_after** — penalises a post-knee segment that is still well
          described by a single line (we *want* accelerated / non-linear
          degradation).
        * **0.1 × |slope_change|** — small regulariser that mildly penalises
          extremely abrupt slope jumps (noise guard).
        """
        n = len(soh_values)

        # Before knee
        x_b = np.arange(knee_idx).reshape(-1, 1)
        y_b = soh_values[:knee_idx]
        r2_b = (
            LinearRegression().fit(x_b, y_b).score(x_b, y_b)
            if len(y_b) > 1
            else 0.0
        )

        # After knee
        x_a = np.arange(knee_idx, n).reshape(-1, 1)
        y_a = soh_values[knee_idx:]
        if len(y_a) > 1:
            m_a = LinearRegression().fit(x_a, y_a)
            r2_a = m_a.score(x_a, y_a)
            slope_change = abs(m_a.coef_[0] - linear_model.coef_[0])
        else:
            r2_a = 0.0
            slope_change = 0.0

        return (1.0 - r2_b) + r2_a + 0.1 * slope_change


# ---------------------------------------------------------------------------
#  n-percent optimisation
# ---------------------------------------------------------------------------

def _evaluate_knee_consistency(soh_values: np.ndarray, knee_idx: int) -> float:
    """
    Score the quality of a detected knee point for use in the grid search
    over *n*.  Lower is better.

    Components
    ----------
    * **(1 − R²_before)** — high R² before the knee means the linear phase
      is well captured.
    * **R²_after** — low R² after the knee means the post-knee phase is
      non-linear (accelerated degradation).
    * **|1 − slope_ratio|** — slope_ratio = |slope_after / slope_before|;
      values ≫ 1 indicate a pronounced knee while values ≈ 1 indicate a
      smooth, gradual transition.
    """
    n = len(soh_values)
    if knee_idx <= 1 or knee_idx >= n - 1:
        return 100.0

    x_b = np.arange(knee_idx).reshape(-1, 1)
    y_b = soh_values[:knee_idx]
    m_b = LinearRegression().fit(x_b, y_b)
    r2_b = m_b.score(x_b, y_b)

    x_a = np.arange(knee_idx, n).reshape(-1, 1)
    y_a = soh_values[knee_idx:]
    m_a = LinearRegression().fit(x_a, y_a)
    r2_a = m_a.score(x_a, y_a)

    slope_b = m_b.coef_[0]
    slope_a = m_a.coef_[0]
    slope_ratio = abs(slope_a / slope_b) if abs(slope_b) > 1e-6 else 1.0

    return (1.0 - r2_b) + r2_a + abs(1.0 - slope_ratio)


def optimize_n_percent(
    all_soh_trajectories: List[np.ndarray],
    n_range: Tuple[float, float] = (0.5, 10.0),
    n_steps: int = 20,
) -> float:
    """
    Grid-search for the optimal threshold *n* (%) across a set of SOH curves.

    For each candidate *n*, every trajectory is scored with
    :func:`_evaluate_knee_consistency` and the *n* with the lowest mean score
    is returned.

    Parameters
    ----------
    all_soh_trajectories : list of np.ndarray
        SOH arrays (one per battery cell).
    n_range : (float, float)
        Search interval ``(min_n, max_n)``.
    n_steps : int
        Number of uniformly spaced candidates in *n_range*.

    Returns
    -------
    float
        The optimal *n* value.
    """
    n_values = np.linspace(n_range[0], n_range[1], n_steps)
    best_n = n_range[0]
    best_score = float("inf")

    for n in n_values:
        detector = KneePointDetector(n_percent=n)
        scores: List[float] = []

        for soh in all_soh_trajectories:
            if len(soh) < 10:
                continue
            try:
                knee_idx = detector.detect_knee_point(soh)
                scores.append(_evaluate_knee_consistency(soh, knee_idx))
            except Exception:
                scores.append(100.0)

        if scores:
            avg = float(np.mean(scores))
            if avg < best_score:
                best_score = avg
                best_n = float(n)

    return best_n


# ---------------------------------------------------------------------------
#  Processing pipeline helpers
# ---------------------------------------------------------------------------

def process_battery_csv(
    input_path: str,
    output_path: str,
    nominal_capacity: float = 2.0,
    n_percent: Optional[float] = None,
    auto_optimize: bool = False,
    all_trajectories: Optional[List[np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Read a raw battery CSV, compute SOH + knee distances, and save the result.

    Adds three new columns: ``soh``, ``knee_point_distance``, ``knee_point_index``.
    """
    df = pd.read_csv(input_path)
    if "capacity_Ah" not in df.columns:
        raise ValueError(
            f"CSV must contain 'capacity_Ah'. Found: {df.columns.tolist()}"
        )

    soh = extract_soh_trajectory(df, "capacity_Ah", nominal_capacity)

    if auto_optimize and all_trajectories is not None:
        n_percent = optimize_n_percent(all_trajectories)
        print(f"  Auto-optimised n_percent = {n_percent:.2f}%")
    elif n_percent is None:
        n_percent = 2.0

    detector = KneePointDetector(n_percent=n_percent)
    knee_distances = detector.compute_knee_distances(soh)
    knee_idx = detector.get_knee_index(soh)

    df["soh"] = soh
    df["knee_point_distance"] = knee_distances
    df["knee_point_index"] = knee_idx

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        f"  {os.path.basename(input_path):30s}  knee_idx={knee_idx:4d}  "
        f"knee_SOH={soh[knee_idx]:.4f}  "
        f"dist=[{knee_distances.min():.4f}, {knee_distances.max():.4f}]"
    )
    return df


def process_dataset_directory(
    input_dir: str,
    output_dir: str,
    nominal_capacity: float = 2.0,
    dataset_name: str = "XJTU",
) -> float:
    """
    Process every battery CSV in *input_dir* and write augmented files to
    *output_dir*.  Returns the optimal *n* found for this dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".csv"))
    print(f"[{dataset_name}] Found {len(csv_files)} battery files")

    # --- Pass 1: collect all SOH trajectories for joint n optimisation ---
    all_soh: List[np.ndarray] = []
    file_paths: List[str] = []
    for csv_file in csv_files:
        path = os.path.join(input_dir, csv_file)
        try:
            df = pd.read_csv(path)
            if "capacity_Ah" in df.columns:
                soh = extract_soh_trajectory(df, "capacity_Ah", nominal_capacity)
                all_soh.append(soh)
                file_paths.append(path)
        except Exception as exc:
            print(f"  Warning: skipping {csv_file}: {exc}")

    optimal_n = optimize_n_percent(all_soh)
    print(f"[{dataset_name}] Optimal n_percent = {optimal_n:.2f}%\n")

    # --- Pass 2: apply optimal n to every file ---
    for path, soh in zip(file_paths, all_soh):
        out_path = os.path.join(output_dir, os.path.basename(path))
        process_battery_csv(
            path,
            out_path,
            nominal_capacity=nominal_capacity,
            n_percent=optimal_n,
            auto_optimize=False,
        )

    return optimal_n


def process_all_datasets(
    base_input_dir: str,
    base_output_dir: str,
    dataset_configs: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Process multiple dataset folders in one call.

    Parameters
    ----------
    dataset_configs : dict
        Mapping ``{name: {'folder': str, 'nominal_capacity': float}}``.

    Returns
    -------
    dict
        ``{dataset_name: optimal_n}`` for each processed dataset.
    """
    results: Dict[str, float] = {}

    for name, cfg in dataset_configs.items():
        in_dir = os.path.join(base_input_dir, cfg["folder"])
        out_dir = os.path.join(base_output_dir, cfg["folder"])
        cap = cfg.get("nominal_capacity", 2.0)

        if not os.path.isdir(in_dir):
            print(f"  Warning: {in_dir} not found — skipping {name}")
            continue

        print(f"\n{'=' * 60}\n  Processing {name}\n{'=' * 60}")
        results[name] = process_dataset_directory(in_dir, out_dir, cap, name)

    return results


# ---------------------------------------------------------------------------
#  Raw CSV → Scalar conversion  (csv_converted/ → dataloader-ready)
# ---------------------------------------------------------------------------

def _extract_scalar_features(
    df: pd.DataFrame,
    soh: np.ndarray,
    knee_distances: np.ndarray,
) -> pd.DataFrame:
    """
    Convert one raw battery DataFrame (array-encoded strings) into a
    scalar-per-cycle DataFrame ready for the dataloader.

    Each row's numpy-array cells are reduced to summary statistics:
    mean / std / min / max for voltage, current, temperature, power;
    start / delta for capacity.

    Output column order (last = target):
        voltage_mean, voltage_std, voltage_min, voltage_max,
        current_mean, current_std,
        temperature_mean, temperature_std,
        power_mean, power_std,
        capacity_start, capacity_delta,
        knee_point_distance, capacity
    """
    rows: List[Dict] = []

    for i in range(len(df)):
        row = df.iloc[i]

        volt = parse_numpy_array_string(str(row.get("voltage_V", "[]")))
        curr = parse_numpy_array_string(str(row.get("current_A", "[]")))
        temp = parse_numpy_array_string(str(row.get("temperature_C", "[]")))
        cap = parse_numpy_array_string(str(row.get("capacity_Ah", "[]")))
        pwr = parse_numpy_array_string(str(row.get("power_Wh", "[]")))

        def _sts(a: np.ndarray):
            if len(a) == 0:
                return 0.0, 0.0, 0.0, 0.0
            return (
                float(np.mean(a)),
                float(np.std(a)),
                float(np.min(a)),
                float(np.max(a)),
            )

        v_m, v_s, v_n, v_x = _sts(volt)
        c_m, c_s, _, _ = _sts(curr)
        t_m, t_s, _, _ = _sts(temp)
        p_m, p_s, _, _ = _sts(pwr)

        cap_start = float(cap[0]) if len(cap) > 0 else 0.0
        cap_end = float(cap[-1]) if len(cap) > 0 else 0.0

        rows.append(
            {
                "voltage_mean": v_m,
                "voltage_std": v_s,
                "voltage_min": v_n,
                "voltage_max": v_x,
                "current_mean": c_m,
                "current_std": c_s,
                "temperature_mean": t_m,
                "temperature_std": t_s,
                "power_mean": p_m,
                "power_std": p_s,
                "capacity_start": cap_start,
                "capacity_delta": cap_end - cap_start,
                "knee_point_distance": float(knee_distances[i]),
                "capacity": cap_end,
            }
        )

    return pd.DataFrame(rows)


def process_raw_battery_csv(
    input_path: str,
    output_path: str,
    nominal_capacity: float = 2.0,
    n_percent: Optional[float] = None,
    all_trajectories: Optional[List[np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Convert a single raw (array-encoded) battery CSV to scalar format with
    ``knee_point_distance`` and save it.

    The output is directly consumable by the dataloader:
    last column = ``capacity`` (raw Ah), second-to-last = ``knee_point_distance``.
    """
    df = pd.read_csv(input_path)
    if "capacity_Ah" not in df.columns:
        raise ValueError(
            f"CSV must contain 'capacity_Ah'. Found: {df.columns.tolist()}"
        )

    soh = extract_soh_trajectory(df, "capacity_Ah", nominal_capacity)

    if n_percent is None:
        if all_trajectories is not None:
            n_percent = optimize_n_percent(all_trajectories)
            print(f"  Auto-optimised n_percent = {n_percent:.2f}%")
        else:
            n_percent = 2.0

    detector = KneePointDetector(n_percent=n_percent)
    knee_distances = detector.compute_knee_distances(soh)
    knee_idx = detector.get_knee_index(soh)

    out_df = _extract_scalar_features(df, soh, knee_distances)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(
        f"  {os.path.basename(input_path):30s}  knee_idx={knee_idx:4d}  "
        f"knee_SOH={soh[knee_idx]:.4f}  "
        f"dist=[{knee_distances.min():.4f}, {knee_distances.max():.4f}]  "
        f"rows={len(out_df)}"
    )
    return out_df


def process_raw_dataset_directory(
    input_dir: str,
    output_dir: str,
    nominal_capacity: float = 2.0,
    dataset_name: str = "XJTU",
) -> float:
    """
    Two-pass processing of an entire raw dataset directory:

    1. **Pass 1** — extract all SOH trajectories and optimise *n* jointly.
    2. **Pass 2** — convert every CSV to scalar format with the optimal *n*
       and save to *output_dir*.

    Returns the optimal *n* value.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".csv"))
    print(f"[{dataset_name}] Found {len(csv_files)} raw battery files")

    all_soh: List[np.ndarray] = []
    file_paths: List[str] = []
    for csv_file in csv_files:
        path = os.path.join(input_dir, csv_file)
        try:
            df = pd.read_csv(path)
            if "capacity_Ah" in df.columns:
                soh = extract_soh_trajectory(df, "capacity_Ah", nominal_capacity)
                all_soh.append(soh)
                file_paths.append(path)
        except Exception as exc:
            print(f"  Warning: skipping {csv_file}: {exc}")

    optimal_n = optimize_n_percent(all_soh)
    print(f"[{dataset_name}] Optimal n_percent = {optimal_n:.2f}%\n")

    for path in file_paths:
        out_path = os.path.join(output_dir, os.path.basename(path))
        process_raw_battery_csv(
            path,
            out_path,
            nominal_capacity=nominal_capacity,
            n_percent=optimal_n,
        )

    return optimal_n


# ---------------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Knee Point Distance — Preprocessing Pipeline")
    print("=" * 60)

    RAW_DIR = "csv_converted"
    PROCESSED_DIR = "data/Processed/XJTU data"

    if not os.path.isdir(RAW_DIR):
        print(f"\nRaw data directory not found: {RAW_DIR}")
        raise SystemExit(1)

    optimal_n = process_raw_dataset_directory(
        input_dir=RAW_DIR,
        output_dir=PROCESSED_DIR,
        nominal_capacity=2.0,
        dataset_name="XJTU",
    )

    print("\n" + "=" * 60)
    print(f"Optimal n_percent: {optimal_n:.2f}%")
    print(f"Output directory:  {PROCESSED_DIR}")
    print("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    params_path = os.path.join(PROCESSED_DIR, "..", "knee_detection_params.json")
    with open(params_path, "w") as fh:
        json.dump({"XJTU": optimal_n}, fh, indent=2)
    print(f"\nSaved → {params_path}")
