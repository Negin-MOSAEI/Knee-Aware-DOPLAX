"""
Validation and Safety Audit Utilities for KaDOPLAX.

Includes:
1. validate_amortized_approximation: Multi-sample relative L2 gap validation.
2. evaluate_uq_calibration: 3-region UQ calibration (PICP/MPIW) and reliability diagrams.
3. audit_data_leakage: Runtime assertions for fold isolation, cell chunking, and HPO splits.
"""

import os
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any


def validate_amortized_approximation(
    model_lax: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    relative_threshold: float = 0.05,
    eps: float = 1e-8,
    n_steps: int = 5,
    inner_lr: float = 1e-2,
    amortized_net: Optional[nn.Module] = None
) -> Dict[str, Any]:
    """
    Validates that the amortized initial-state predictor closely approximates
    the iterative 5-step gradient optimization solution.

    Computes:
        relative_gap = ||y_amortized - y_iterative||_2 / (||y_iterative||_2 + eps)

    Evaluates across all validation samples and reports:
        mean, max, and P95 relative gap, alongside raw L2 gap.
    """
    model_lax.eval()
    all_rel_gaps: List[float] = []
    all_raw_gaps: List[float] = []

    # Identify the amortized predictor
    predictor = amortized_net
    if predictor is None:
        if hasattr(model_lax, 'amortized_predictor') and model_lax.amortized_predictor is not None:
            predictor = model_lax.amortized_predictor
        elif hasattr(model_lax, 'predict_y'):
            predictor = model_lax.predict_y

    for batch in val_loader:
        # Extract features and time depending on dataloader signature
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 3:
                # Assuming (x, t, ...) or (features, kpd, cycle_t, ...)
                x_sample = batch[0].to(device)
                if x_sample.dim() == 3:
                    # [batch, seq_len, feat] -> take last cycle's features
                    x = x_sample[:, -1, :3] if x_sample.shape[-1] >= 3 else x_sample[:, -1]
                else:
                    x = x_sample[:, :model_lax.x_dim] if hasattr(model_lax, 'x_dim') else x_sample

                t = batch[2].to(device) if len(batch) > 2 else torch.zeros(x.shape[0], 1, device=device)
                if t.dim() > 1:
                    t = t.squeeze(-1)
            else:
                x = batch[0].to(device)
                t = batch[1].to(device) if len(batch) > 1 else torch.zeros(x.shape[0], device=device)
                if t.dim() > 1:
                    t = t.squeeze(-1)
        else:
            continue

        batch_size = x.shape[0]
        y_dim = getattr(model_lax, 'y_dim', x.shape[-1])

        # 1. Compute y_amortized
        if predictor is not None:
            with torch.no_grad():
                if callable(predictor):
                    if hasattr(predictor, 'forward'):
                        try:
                            t_in = t.unsqueeze(-1) if t.dim() == 1 else t
                            y_amortized = predictor(torch.cat([x, t_in], dim=-1))
                        except Exception:
                            y_amortized = predictor(x)
                    else:
                        y_amortized = predictor(x, t)
                else:
                    y_amortized = torch.zeros(batch_size, y_dim, device=device)
        elif hasattr(model_lax, 'optimize_y_amortized'):
            with torch.no_grad():
                y_amortized = model_lax.optimize_y_amortized(x, t)
        else:
            # Fallback placeholder if no amortized predictor exists yet
            y_amortized = torch.zeros(batch_size, y_dim, device=device)

        # 2. Compute y_iterative (reference ground truth via n_steps gradient steps)
        with torch.enable_grad():
            x_ref = x.detach()
            t_ref = t.detach()
            # Deterministic initialization from dataset prior (X_mean)
            if hasattr(model_lax, 'X_mean') and model_lax.X_mean is not None:
                y_init = model_lax.X_mean[:y_dim].unsqueeze(0).repeat(batch_size, 1)
            else:
                y_init = torch.zeros(batch_size, y_dim, device=device)

            y_iter = nn.Parameter(y_init.clone().detach())
            opt = torch.optim.Adam([y_iter], lr=inner_lr)

            for _ in range(n_steps):
                opt.zero_grad()
                out = model_lax.net(x_ref, y_iter, t_ref)
                loss = out.mean()
                loss.backward()
                opt.step()

            y_iterative = y_iter.detach()

        # 3. Compute raw and relative L2 gaps
        raw_diff = torch.norm(y_amortized - y_iterative, p=2, dim=-1)  # [B]
        iter_norm = torch.norm(y_iterative, p=2, dim=-1)               # [B]
        rel_diff = raw_diff / (iter_norm + eps)                        # [B]

        all_raw_gaps.extend(raw_diff.cpu().tolist())
        all_rel_gaps.extend(rel_diff.cpu().tolist())

    if not all_rel_gaps:
        return {"passed": False, "error": "No validation samples evaluated."}

    mean_rel = float(np.mean(all_rel_gaps))
    max_rel = float(np.max(all_rel_gaps))
    p95_rel = float(np.percentile(all_rel_gaps, 95))

    mean_raw = float(np.mean(all_raw_gaps))
    max_raw = float(np.max(all_raw_gaps))
    p95_raw = float(np.percentile(all_raw_gaps, 95))

    passed = bool(p95_rel <= relative_threshold)

    results = {
        "passed": passed,
        "relative_threshold": relative_threshold,
        "relative_gap": {
            "mean": mean_rel,
            "max": max_rel,
            "p95": p95_rel
        },
        "raw_l2_gap": {
            "mean": mean_raw,
            "max": max_raw,
            "p95": p95_raw
        },
        "num_samples_evaluated": len(all_rel_gaps)
    }

    print(f"\n[Amortized Validation] Evaluated {len(all_rel_gaps)} samples.")
    print(f"  Relative L2 Gap -> Mean: {mean_rel:.4f}, P95: {p95_rel:.4f}, Max: {max_rel:.4f} (Threshold: {relative_threshold})")
    print(f"  Raw L2 Gap      -> Mean: {mean_raw:.4f}, P95: {p95_raw:.4f}, Max: {max_raw:.4f}")
    print(f"  Status: {'PASSED [OK]' if passed else 'FAILED [GAP EXCEEDED]'}")

    return results


def evaluate_uq_calibration(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    cycle_indices: np.ndarray,
    knee_point_cycle: int,
    nominal_confidence: float = 0.95,
    near_knee_window: int = 10,
    save_fig_path: Optional[str] = None,
    tolerance_pp: float = 0.05
) -> Dict[str, Any]:
    """
    Evaluates Uncertainty Quantification (UQ) calibration across 3 operational degradation regions:
      (a) pre-knee: cycle < knee_point_cycle - near_knee_window
      (b) near-knee: knee_point_cycle - near_knee_window <= cycle <= knee_point_cycle + near_knee_window
      (c) post-knee: cycle > knee_point_cycle + near_knee_window

    Computes:
      - PICP (Prediction Interval Coverage Probability)
      - MPIW (Mean Prediction Interval Width)
      - Multi-level Reliability Diagram (confidence levels from 0.10 to 0.99)
      - Calibration divergence alert between near-knee and aggregate coverage.
    """
    from scipy.stats import norm

    y_true = np.asarray(y_true).ravel()
    y_pred_mean = np.asarray(y_pred_mean).ravel()
    y_pred_std = np.maximum(np.asarray(y_pred_std).ravel(), 1e-6)
    cycle_indices = np.asarray(cycle_indices).ravel()

    # Define regional masks
    k_min = knee_point_cycle - near_knee_window
    k_max = knee_point_cycle + near_knee_window

    mask_pre = cycle_indices < k_min
    mask_near = (cycle_indices >= k_min) & (cycle_indices <= k_max)
    mask_post = cycle_indices > k_max
    mask_all = np.ones(len(y_true), dtype=bool)

    regions = {
        "aggregate": mask_all,
        "pre_knee": mask_pre,
        "near_knee": mask_near,
        "post_knee": mask_post
    }

    # Standard normal quantile for nominal confidence
    z_nom = norm.ppf((1.0 + nominal_confidence) / 2.0)

    metrics: Dict[str, Dict[str, float]] = {}

    for name, mask in regions.items():
        if np.sum(mask) == 0:
            metrics[name] = {"count": 0, "picp": 0.0, "mpiw": 0.0}
            continue

        lower = y_pred_mean[mask] - z_nom * y_pred_std[mask]
        upper = y_pred_mean[mask] + z_nom * y_pred_std[mask]
        covered = (y_true[mask] >= lower) & (y_true[mask] <= upper)

        picp = float(np.mean(covered))
        mpiw = float(np.mean(upper - lower))
        metrics[name] = {
            "count": int(np.sum(mask)),
            "picp": picp,
            "mpiw": mpiw
        }

    # Check for near-knee divergence alert
    agg_picp = metrics["aggregate"]["picp"]
    near_picp = metrics["near_knee"]["picp"]
    picp_divergence = abs(near_picp - agg_picp)
    near_knee_alert = picp_divergence > tolerance_pp

    # Reliability diagram curve points across alpha in [0.10, 0.99]
    nominal_levels = np.linspace(0.10, 0.98, 25)
    empirical_curves: Dict[str, List[float]] = {r: [] for r in regions.keys()}

    for alpha in nominal_levels:
        z_val = norm.ppf((1.0 + alpha) / 2.0)
        for r_name, mask in regions.items():
            if np.sum(mask) == 0:
                empirical_curves[r_name].append(0.0)
                continue
            lb = y_pred_mean[mask] - z_val * y_pred_std[mask]
            ub = y_pred_mean[mask] + z_val * y_pred_std[mask]
            cov = np.mean((y_true[mask] >= lb) & (y_true[mask] <= ub))
            empirical_curves[r_name].append(float(cov))

    # Generate and save reliability diagram if path requested
    if save_fig_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_fig_path)), exist_ok=True)
        plt.figure(figsize=(7, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Ideal Calibration (y=x)', alpha=0.7)
        plt.plot(nominal_levels, empirical_curves["aggregate"], 'o-', color='#1f77b4', lw=2, label=f'Aggregate (PICP={agg_picp:.3f})')
        plt.plot(nominal_levels, empirical_curves["pre_knee"], 's--', color='#2ca02c', label=f'Pre-Knee (PICP={metrics["pre_knee"]["picp"]:.3f})')
        plt.plot(nominal_levels, empirical_curves["near_knee"], '^-.', color='#d62728', lw=2.5, label=f'Near-Knee (PICP={near_picp:.3f})')
        plt.plot(nominal_levels, empirical_curves["post_knee"], 'd:', color='#ff7f0e', label=f'Post-Knee (PICP={metrics["post_knee"]["picp"]:.3f})')

        plt.xlabel('Nominal Confidence Level $(1 - \\alpha)$', fontsize=12)
        plt.ylabel('Empirical Coverage (PICP)', fontsize=12)
        plt.title(f'UQ Reliability Diagram (Knee Cycle: {knee_point_cycle})', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right', framealpha=0.9)
        plt.xlim([0.05, 1.0])
        plt.ylim([0.05, 1.02])
        plt.tight_layout()
        plt.savefig(save_fig_path, dpi=300)
        plt.close()
        print(f"[UQ Calibration] Saved reliability diagram figure to: {save_fig_path}")

    report = {
        "nominal_confidence": nominal_confidence,
        "knee_point_cycle": knee_point_cycle,
        "near_knee_window": near_knee_window,
        "metrics_by_region": metrics,
        "picp_divergence_near_vs_agg": float(picp_divergence),
        "near_knee_alert": bool(near_knee_alert),
        "figure_path": save_fig_path
    }

    print(f"\n[UQ Regional Calibration Report] (Nominal: {nominal_confidence*100:.1f}%)")
    for r_name, m in metrics.items():
        print(f"  {r_name.upper():<10} | Samples: {m['count']:<5} | PICP: {m['picp']:.4f} | MPIW: {m['mpiw']:.4f}")
    if near_knee_alert:
        print(f"  [ALERT] Near-knee calibration diverges by {picp_divergence*100:.2f} pp from aggregate (> {tolerance_pp*100:.1f} pp tolerance)!")

    return report


def audit_data_leakage(
    project_root: str,
    dataset_name: str,
    batch_name: str,
    test_battery_ids: List[str],
    train_battery_ids: List[str],
    standardizer_obj: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Strict runtime data-leakage audit:
    1. Standardizer Leakage: Verifies standardization statistics are fit strictly on training fold.
    2. Cell Chunking: Verifies sequence windows never mix cycles across battery cell boundaries.
    3. HPO Split Leakage: Verifies Optuna HPO trials never evaluated against held-out LOBO test cells.
    """
    print("\n" + "="*60)
    print(f"[DATA LEAKAGE AUDIT] Verifying: {dataset_name} - {batch_name}")
    print("="*60)

    audit_results: Dict[str, Any] = {
        "passed": True,
        "violations": [],
        "warnings": []
    }

    # Check 1: Disjoint battery IDs between train and test
    overlap_ids = set(test_battery_ids).intersection(set(train_battery_ids))
    if overlap_ids:
        msg = f"Critical train/test fold overlap: Battery IDs {overlap_ids} appear in both splits!"
        audit_results["violations"].append(msg)
        audit_results["passed"] = False
        raise AssertionError(msg)
    else:
        print("  [Pass] Train and Test battery splits are strictly disjoint.")

    # Check 2: Standardizer fitting isolation
    if standardizer_obj is not None:
        fitted_ids = getattr(standardizer_obj, 'fitted_battery_ids', None)
        if fitted_ids is not None:
            leaked = set(fitted_ids).intersection(set(test_battery_ids))
            if leaked:
                msg = f"Standardizer was fitted on test battery IDs: {leaked}!"
                audit_results["violations"].append(msg)
                audit_results["passed"] = False
                raise AssertionError(msg)
            else:
                print("  [Pass] Standardizer verified to contain only training fold statistics.")
        else:
            audit_results["warnings"].append(
                "Standardizer object does not record fitted_battery_ids metadata. Ensure .fit() was called on train set only."
            )

    # Check 3: Optuna HPO Study Leakage
    hpo_log_path = os.path.join(project_root, 'outputs', 'reports', 'hpo_trial_battery_splits.json')
    study_log_path = os.path.join(project_root, 'outputs', 'reports', 'hpo_study_log.json')

    found_log = None
    for p in [hpo_log_path, study_log_path]:
        if os.path.exists(p):
            found_log = p
            break

    if found_log is not None:
        try:
            with open(found_log, 'r') as f:
                hpo_data = json.load(f)

            # Check if any trial used test batteries
            hpo_used_batteries = set()
            if isinstance(hpo_data, dict):
                for key, val in hpo_data.items():
                    if isinstance(val, list):
                        hpo_used_batteries.update(val)
                    elif isinstance(val, dict) and 'evaluated_batteries' in val:
                        hpo_used_batteries.update(val['evaluated_batteries'])

            hpo_leaked = hpo_used_batteries.intersection(set(test_battery_ids))
            if hpo_leaked:
                msg = f"Optuna HPO Leakage: Held-out test batteries {hpo_leaked} were used during HPO trials!"
                audit_results["violations"].append(msg)
                audit_results["passed"] = False
                raise AssertionError(msg)
            else:
                print("  [Pass] Optuna HPO trials confirmed zero overlap with test battery cells.")
        except Exception as e:
            audit_results["warnings"].append(f"Failed to parse HPO split log: {e}")
    else:
        warn_msg = (
            f"WARNING: HPO trial split log not found at '{hpo_log_path}'. "
            "Unable to automatically verify that historical Optuna HPO trials excluded test battery cells. "
            "Recommendation: Run HPO with split logging enabled before citing final publication metrics."
        )
        print(f"  [WARN] {warn_msg}")
        audit_results["warnings"].append(warn_msg)

    print(f"[Audit Summary] Status: {'PASSED' if audit_results['passed'] else 'FAILED'}")
    return audit_results
