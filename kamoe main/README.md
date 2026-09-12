# Knee-Aware DOPLAX: Multi-Modal Physics-Informed Operator Learning & Hamilton-Jacobi Solver for Battery SOH Estimation

A comprehensive, modular framework for lithium-ion battery **State-of-Health (SOH)** estimation and **Knee-Point Identification** across 13 benchmark batches and 4 public datasets (**XJTU**, **MIT**, **TJU**, **HUST**).

---

## 🌟 Key Architecture & Methodology

```
                       ┌────────────────────────────────────────────────────────┐
                       │               Raw Cycle Trajectory Input               │
                       │             (16 Features + Cycle Index)                │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                  ┌───────────────────────────────┼───────────────────────────────┐
                  ▼                               ▼                               ▼
       ┌────────────────────┐          ┌────────────────────┐          ┌────────────────────┐
       │     TCN Model      │          │     DeepOPINN      │          │     LAX Solver     │
       │ (Causal Dilated)   │          │ (Operator Learning)│          │  (Hopf-Lax HJ PDE) │
       └──────────┬─────────┘          └──────────┬─────────┘          └──────────┬─────────┘
                  │                               │                               │
                  │ Continuous KPD                │ SOH Prediction                │ SOH Prediction
                  │ d = -arctan(C - C_knee)       │ u_deepopinn                   │ u_lax
                  │                               │                               │
                  └───────────────────────┬───────┴───────────────────────────────┘
                                          │
                                          ▼
                               ┌────────────────────┐
                               │     FusionMLP      │
                               │(Knee-Aware Fusion) │
                               └──────────┬─────────┘
                                          │
                                          ▼ Raw SOH
                               ┌────────────────────┐
                               │  Post-Processing   │
                               │ (Hampel + SavGol)  │
                               └──────────┬─────────┘
                                          │
                                          ▼ Final SOH
                               ┌────────────────────┐
                               │ Benchmark Metrics  │
                               │ (RMSE, MAE, ΔCknee)│
                               └────────────────────┘
```

1. **Temporal Convolutional Network (`TCN`)**:
   - Predicts continuous **Knee Point Distance (KPD)**:
     $$d(C) = -\arctan(C - C_{\text{knee\_point}})$$
   - Estimates zero-crossing cycle $\hat{C}_{\text{knee}} = \arg\min_C |\hat{d}(C)|$.

2. **Physics-Informed Deep Operator Network (`DeepOPINN`)**:
   - Integrates DeepONet Cartesian product feature extraction with dynamical residual PDE networks.
   - **Knee-Aware PDE Loss**:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \alpha \cdot \frac{1}{N}\sum_{c=1}^N \Big( f(c)^2 \cdot \max\big(0, \widehat{\text{KPD}}(c)\big) \Big) + \beta \cdot \mathcal{L}_{\text{mono}}$$

3. **Hopf–Lax Hamilton–Jacobi Solver (`LAX`)**:
   - Solves the Hamilton–Jacobi PDE $u(x, t) = \min_y [g(y) + t H^*((x-y)/t)]$ with learnable Hamiltonian $H^*$ and symmetric distance blocks.

4. **Multi-Modal Fusion (`FusionMLP`)**:
   - Adaptively combines $u_{\text{deepopinn}}$, $u_{\text{lax}}$, and predicted $\widehat{\text{KPD}}$ to output final capacity estimates.

5. **Advanced Post-Processing (`post_processing.py`)**:
   - Applies segment-wise Hampel outlier filtering and Savitzky–Golay smoothing without monotonic-only constraints, allowing natural capacity regeneration and relaxation.

---

## 📊 Benchmark Datasets & Batches (13 Canonical Batches)

All dataset splits and ground truth knee points are configured via `train_test_split.json` and `initial_knee_points.json`:

| Dataset | Batches Supported | Normalization | Nominal Capacity |
| :--- | :--- | :--- | :--- |
| **XJTU** | `2C`, `3C`, `R2.5`, `R3`, `RW`, `Sim_satellite` | Min-Max | 2.0 Ah |
| **MIT** | `2017-05-12`, `2017-06-30`, `2018-04-12` | Min-Max | 1.1 Ah |
| **TJU** | `Dataset_1_NCA_battery`, `Dataset_2_NCM_battery`, `Dataset_3_NCM_NCA_battery` | Min-Max | 3.5 Ah / 2.5 Ah |
| **HUST** | `default` | Min-Max | 1.1 Ah |

---

## 🚀 Quickstart & Pipeline Execution

### 1. Train Knee Point Distance Model (TCN)
```bash
# Train on all 13 batches
python train_tcn.py --dataset all --batch all

# Train on specific batch
python train_tcn.py --dataset XJTU --batch 2C
```

### 2. Predict and Save KPD Trajectories
```bash
# Predict KPD and estimate knee cycles for all batteries
python predict_kpd.py --dataset all --batch all
```

### 3. Train DeepOPINN Operator Network
```bash
# Train DeepOPINN with knee-aware PDE residual weighting
python train_deepopinn.py --dataset all --batch all
```

### 4. Train LAX Hopf-Lax Solver
```bash
# Train LAX solver network
python train_lax.py --dataset all --batch all
```

### 5. Train Multi-Modal Fusion Network (FusionMLP)
```bash
# Train FusionMLP combining DeepOPINN + LAX + KPD
python train_fusion.py --dataset all --batch all
```

### 6. End-to-End Inference
```bash
# Generate full cycle trajectories (CSV + NPY) for all 13 batches
python inference.py --dataset all --batch all
```

### 7. Evaluate KPIs & Benchmark Table
```bash
# Generate CSV, Markdown, and JSON KPI benchmark reports
python evaluate_kpis.py --dataset all --batch all
```

### 8. Generate Publication Plots
```bash
# Generate high-resolution (300 DPI PNG + vector SVG) degradation curves
python plot_results.py --dataset all --batch all --split test
```

---

## 📁 Repository Structure

```
kamoe main/
├── configs.py                  # Dataset-specific hyperparameter dataclasses
├── post_processing.py          # Hampel de-spiking + Savitzky-Golay smoothing
├── train_tcn.py                # TCN training script with cosine LR scheduler
├── predict_kpd.py              # KPD inference & zero-crossing knee estimation
├── train_deepopinn.py          # DeepOPINN training with knee-weighted PDE loss
├── train_lax.py                # LAX Hopf-Lax Hamilton-Jacobi solver training
├── train_fusion.py             # FusionMLP multi-modal training script
├── inference.py                # End-to-end full evaluation pipeline
├── evaluate_kpis.py            # Automated KPI benchmarking table generator
├── plot_results.py             # Publication degradation curves & KPD dynamics
│
├── dataloader/                 # Data loading & preprocessing modules
│   ├── knee_points.py          # JSON registries for splits and knee points
│   ├── dataloader.py           # BatteryTrajectoryDataset & Preprocessor
│   └── data_helper.py          # Partitioned loader helpers for 4 datasets
│
├── Model/                      # Neural network architectures
│   ├── Backbones/              # TCN, DeepONet, MLP backbones
│   ├── Auxiliary_nets/         # Solution_u SOH prediction network
│   ├── PI_nets/                # DeepOPINN and LAX solver implementations
│   ├── Combination_nets/       # FusionMLP and Bagging networks
│   └── utils/                  # Losses, LR schedulers, and evaluation metrics
│
├── checkpoints/                # Saved model weights per dataset and batch
├── predictions/                # Output CSV and NPY trajectories
└── results/                    # KPI benchmark tables and publication figures
```

---

## 📈 Evaluation Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error ($\%$)
- **$\Delta C_{\text{knee}}$**: Absolute Knee Cycle Prediction Error in cycles ($|\hat{C}_{\text{knee}} - C_{\text{knee\_point}}|$)
