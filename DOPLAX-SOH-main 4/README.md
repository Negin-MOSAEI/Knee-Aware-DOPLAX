# KaDOPLAX: Knee-Aware Deep Operator and Physics-Informed Lax solver for Battery Prognostics

KaDOPLAX is a state-of-the-art, 8-phase pipeline and deep learning architecture for robust, early-stage Lithium-Ion Battery (LIB) State of Health (SOH) estimation and Knee Point Discovery (KPD). 

By intrinsically coupling Physics-Informed Neural Networks (PINNs) via the DeepOPINN framework, the Hopf-Lax mathematical theory for solving Non-Linear PDEs, and a Bagging MLP for predictive fusion, this repository guarantees a strict **Zero Data Leakage** policy while establishing new benchmarks for reliable early-cycle battery health forecasting.

## 🌟 Key Features
- **Strict Two-Phase Fusion Training Strategy:** Completely distinct training phases for independent feature extraction pipelines (DeepOPINN + LAX) and their subsequent amalgamation (Bagging MLP).
- **Zero Data Leakage Inference:** Predicts missing Knee Point Distribution (KPD) parameters for unseen test batteries purely from a robust Transformer (TST) predictor using early cycles (e.g. 1-40). No target SOH or KPD leaks into any phase of inference.
- **Physical Capacity Nominalization:** Accurately anchors relative SOH calculations against dataset-specific physical battery chemistries (XJTU: 2.0Ah, MIT: 1.1Ah, HUST: 1.1Ah, TJU: 3.5Ah/2.5Ah).
- **Automated Bayesian HPO:** Native integration with `Optuna` for extensive hyperparameter exploration, tuning architecture dimensionality specific to the volatility of different datasets.

## 🏗️ Architecture

The KaDOPLAX architecture consists of three interconnected modules acting together under a unified pipeline constraint.

- **Module 1 (DeepOPINN Pipeline):** A physics-informed continuous branch/trunk network (PI-Net) that predicts the hidden physics states ($u_1$) of the battery given input features and cycle index $t$.
- **Module 2 (LAX Pipeline):** A highly specialized Hopf-Lax solver utilizing a learnable block. Generates independent condition approximations ($u_2$).
- **Module 3 (Fusion Module):** An integrated Bagging Multi-Layer Perceptron (MLP) mapping the composite tensor `[u_1, u_2, KPD]` to the final scalar SOH estimation.

## 🚀 The 8-Phase Prognostics Pipeline

Execution is entirely automated through the `run_all.py` master orchestrator script. 

1. **Phase 1: Train TST (Time Series Transformer)**
   - Learns to map raw early-cycle features directly to an explicit continuous Knee Point Distribution (KPD).
2. **Phase 2: Infer KPD**
   - Applies the pre-trained TST on *all* dataset variants to predict their inferred KPDs. This inferred parameter replaces Ground Truth entirely for later fusion to prevent leakage.
3. **Phase 3: Train DeepOPINN**
   - Optimizes the physics-informed BranchNet + TrunkNet modules individually for each battery cross-section.
4. **Phase 4: Train LAX Module**
   - Optimizes the Hopf-Lax sub-network mapping to discover secondary topological degradation pathways.
5. **Phase 5: Train Fusion MLP**
   - Locks the gradients of both DeepOPINN and LAX. Feeds combinations of $u_1$, $u_2$, and the *Inferred KPD (from Phase 2)* to the Bagging MLP.
6. **Phase 6: Final SOH Estimation & Plotting**
   - Inference-only extraction. SOH is estimated for test-set batteries. Smooths final outputs using dynamic Savitzky-Golay filters. 
7. **Phase 7: Inference KPI Reporting**
   - Calculates comprehensive metric evaluations (RMSE, MAE, R², MAPE, Max Error) and aggregates them into `outputs/final_results/aggregated_metrics.csv`.
8. **Phase 8: Project Cleanup**
   - Archives raw epoch plots and clears intermediate caches for memory efficiency.

## 🛠️ Usage & Execution

### 1. Requirements & Setup
All requisite modules exist under standard machine-learning Python distributions:
```bash
pip install torch numpy pandas scikit-learn scipy matplotlib deepxde optuna
```

### 2. Standard Training & Evaluation
To execute the comprehensive 8-Phase training pipeline for all configured datasets (XJTU, MIT, HUST, TJU):
```bash
python run_all.py
```
Outputs, models, generated SOH plots, and aggregated evaluation KPIs will automatically be structured in the `/outputs` directory.

### 3. Hyperparameter Optimization (HPO)
To run automated Bayesian Architecture searches via Optuna (optimizing hidden layers, depths, and learning rates):
```bash
python run_hpo.py
```
This saves optimized layer topologies internally to `config/optimized_architectures.json`, which the 8-Phase pipeline automatically detects and scales to dynamically.

## 📂 Repository Structure

- `Model/` - All KaDOPLAX network layers (`Backbones/`, `Auxiliary_nets/`, `Combination_nets/`).
- `pipeline/` - Scripts dividing execution into logical phase segments (1 through 8).
- `dataloader/` - Contains parsing scripts, specialized normalizers, and the crucial capacity-anchored `BatteryCycleDataset`.
- `config/` - Houses JSON profiles dictating train-test splits and optimized network geometries.
- `outputs/` - Generated metrics, figures, un-normalized KPD predictions, and model states (`.pt`).
- `run_all.py` - Master orchestrator for the KaDOPLAX end-to-end framework.
- `run_hpo.py` - Standalone manager for iterative architecture search loops.
