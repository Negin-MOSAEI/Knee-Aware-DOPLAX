"""
run_phases3_7.py – Runner for DOPLAX Pipeline Phases 3-7
==========================================================
Forces CPU-only execution on macOS and runs with 10 epochs.

Usage:
    python run_phases3_7.py
"""

import os
import sys

# ---- Force CPU-only before any torch import ----
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["DDE_BACKEND"] = "pytorch"

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import pipeline.phase3_train_deepopinn as phase3
import pipeline.phase4_train_lax as phase4
import pipeline.phase5_train_fusion as phase5
import pipeline.phase6_final_soh as phase6
import pipeline.phase7_infer_kpis as phase7

# Patch AFTER all imports: deepxde resets torch defaults, so we monkey-patch
# torch.randperm to always produce CPU tensors. This prevents the MPS/CPU
# generator-device mismatch inside DataLoader's RandomSampler.
import torch as _torch

_orig_randperm = _torch.randperm


def _cpu_randperm(*args, **kwargs):
    kwargs["device"] = "cpu"
    return _orig_randperm(*args, **kwargs)


_torch.randperm = _cpu_randperm

NUM_EPOCHS = 10


def main():
    print("=" * 60)
    print(f"  DOPLAX Phases 3-7  |  epochs={NUM_EPOCHS}  |  device=cpu")
    print("=" * 60)

    print("\n>>> Phase 3: Training DeepOPINN ...")
    phase3.run_phase3(project_root, num_epochs=NUM_EPOCHS)
    print("-" * 60)

    print("\n>>> Phase 4: Training LAX ...")
    phase4.run_phase4(project_root, num_epochs=NUM_EPOCHS)
    print("-" * 60)

    print("\n>>> Phase 5: Training Fusion MLP ...")
    phase5.run_phase5(project_root, num_epochs=NUM_EPOCHS)
    print("-" * 60)

    print("\n>>> Phase 6: Final SOH Estimation ...")
    phase6.run_phase6(project_root)
    print("-" * 60)

    print("\n>>> Phase 7: Inference KPI Reporting ...")
    phase7.run_phase7(project_root)

    print("\n" + "=" * 60)
    print("  PIPELINE PHASES 3-7 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
