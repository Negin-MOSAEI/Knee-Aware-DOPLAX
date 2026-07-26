
<h1 align="center">Knee-Aware DOPLAX: Unified Physics-Informed Framework via Operator Learning and Hopf–Lax Solver for Battery Health Estimation</h1>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>
<p align="center">
  <b>Anonymous code repository for submission</b>
</p>



---

## 📋 Table of Contents
- [Overview](#-overview)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training Models](#-training-models)
- [Inference](#-inference)
- [Visualization](#-visualization)
- [Knee Point Detection](#-knee-point-detection)
- [Loss Functions](#-loss-functions)
- [References](#-references)

---

## 🔬 Overview

**Knee-Aware DOPLAX** is a physics-integrated framework for accurate State-of-Health (SOH) estimation of lithium-ion batteries. It addresses the challenges of nonlinear degradation and dataset shift across cycling protocols by unifying three physics-informed approaches:

1. **DeepOPINN**: A DeepONet-backboned Physics-Informed Neural Network (PINN) enforcing PDE residuals for operator-level feature extraction.
2. **LAX Network**: A novel solver-style network derived from the Hopf–Lax representation of Hamilton–Jacobi (HJ) equations with learned Hamiltonian and dual-convexity regularization.
3. **Knee-Aware Bagging Fusion Module**: A learned fusion mechanism that optimally combines predictions from both pathways with knee-point distance awareness for degradation-state-aware SOH estimation.

**Key Innovation**: The inclusion of knee-point distance as a third input to the fusion MLP gives the model awareness of where the battery sits relative to its degradation knee point — the inflection where capacity fade accelerates.

**Comprehensive Evaluation**: Validated on four public benchmarks (XJTU, TJU, MIT, HUST).

---

## 🏗️ Architecture

### DOPLAX Framework

<p align="center">
  <img src="./figures/DOPLAX_refined_v5.2_art.svg" alt="DOPLAX Architecture" width="90%"/>
</p>

### Knee-Aware DOPLAX Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Knee-Aware DOPLAX                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Raw Features (x) ──┬──→ DeepONet (extractor + Solution_u)  │
│                      │           ↓                           │
│                      │       u_pinn                          │
│                      │           ↓                           │
│  Raw Features (x) ───┼──→ LAX (OptimizationNetwork)         │
│  Time (t) ───────────┘           ↓                           │
│                            u_lax                             │
│                      ↓           ↓                           │
│  Knee Distance ──────┼───────────┼─────────────────────────  │
│  (kd)                ↓           ↓                           │
│              ┌─────────────────────────┐                     │
│              │  MLP_Bagging_NN([3→1])  │                     │
│              │  Input: [u_pinn, u_lax,  │                     │
│              │          knee_distance]  │                     │
│              └────────────┬────────────┘                     │
│                           ↓                                  │
│                    Final SOH Prediction                       │
└─────────────────────────────────────────────────────────────┘
```

### Two-Phase Training Strategy

<p align="center">
  <img src="./figures/DOPLAX_refined_v5.2_strategy.svg" alt="Two-Phase Training Architecture" width="90%"/>
</p>



---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (recommended for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/DOPLAX-SOH-4765/
cd DOPLAX-SOH

# Create conda environment
conda create -n doplax python=3.10
conda activate doplax

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r ./requirements.txt
```

### Requirements

```txt
matplotlib
numpy
pandas
regex
scikit-learn
seaborn
scipy
natsort
deepxde
optuna
openpyxl
plotly
h5py
tqdm
mpld3
umap-learn
openTSNE
```

---

## 📊 Dataset Preparation

DOPLAX supports four public battery datasets. Download and organize them as follows:

### Supported Datasets

| Dataset | Batteries | Source |
|---------|-----------|--------|
| **XJTU** | 55 | [Link](https://github.com/wang-fujin/PINN4SOH/tree/main/data) |
| **TJU** | 130 | [Link](https://github.com/wang-fujin/PINN4SOH/tree/main/data) |
| **MIT** | 125 | [Link](https://github.com/wang-fujin/PINN4SOH/tree/main/data) |
| **HUST** | 77 | [Link](https://github.com/wang-fujin/PINN4SOH/tree/main/data) |

 **Raw data can be obtained from the following links:**

https://github.com/wang-fujin/PINN4SOH?tab=readme-ov-file#4--additional-information

### Directory Structure

```
data/
├── Full/
│   ├── XJTU data/
│   │   ├── 2C_battery-1.csv
│   │   ├── 2C_battery-2.csv
│   │   └── ...
│   ├── TJU data/
│   │   ├── NCA/
│   │   ├── NCM/
│   │   └── NCM_NCA/
│   ├── MIT data/
│   │   ├── 2017-05-12/
│   │   ├── 2017-06-30/
│   │   └── 2018-04-12/
│   └── HUST data/
│       ├── 1-1.csv
│       └── ...
└── Processed/
    ├── XJTU data/
    ├── TJU data/
    ├── MIT data/
    └── HUST data/
```

### Batch Information

<details>
<summary><b>XJTU Dataset Batches</b></summary>

| Batch | Name | Batteries |
|-------|------|-----------|
| 1 | 2C | 8 |
| 2 | 3C | 15 |
| 3 | R2.5 | 8 |
| 4 | R3 | 8 |
| 5 | RW | 8 |
| 6 | satellite | 8 |

<details>
<summary><b>TJU Dataset Batches</b></summary>

| Batch | Chemistry | Batteries |
|-------|-----------|-----------|
| 1 | NCA | 66 |
| 2 | NCM | 55 |
| 3 | NCM-NCA | 9 |

## 🚀 Training Models


First, open one of the main files designed for each dataset—such as main_XJTU.py, main_TJU.py, or others—then follow the steps below to train your chosen model.

### STEP1: Setting Main parameters in each main files

| **Main Parameter**            | **Description** |
|-------------------------------|-----------------|
| **run_main** | Set to **True** if you want to recreate tables (instead of Table 3 from the paper); otherwise set to **False**. |
| **run_optimization** | Set to **True** to run hyperparameter optimization for your chosen model; otherwise set to **False**. |
| **run_samll_sample** | Set to **True** to limit the number of training samples; otherwise set to **False**. |
| **train_with_target_cells** | Set to **True** to restrict training to one or two specific batteries; otherwise set to **False**. |
| **run_info** | First decide whether you want to run a single model or run models through a pipeline by setting `run_for_a_model_explicitly`. Then, assign the model name to `model_name`. The `model_name` value is required when running a model explicitly. |
| **run_for_LAX** | Controls weight initialization for LAX is matter when you want to use an integrated model like DOPLAX: **True** = unfreeze, **False** = freeze, **None** = random initial weights. |
| **run_for_DeepOPINN** | Controls weight initialization for DeepOPINN is matter when you want to use an integrated model like DOPLAX: **True** = unfreeze, **False** = freeze, **None** = random initial weights. |
| **finetuning_mode** | Set to **True** if you want to calculate and save losses in an Excel file (unlike Table 3 in the paper); otherwise set to **False**. |
| **path_to_weights** | Path to the pretrained weight files. |
| **run_name** | Directory or path where experiment results will be saved. |
| **data_path** | Path to the dataset storage location. |
| **n_experiments** | Number of times each model should be trained per batch. |


### STEP2: Training Individual Models

#### DeepOPINN Only

```python
run_info = {
    'run_for_a_model_explicitly': True,
    'model_name': 'DeepOPINN'
}
```

#### LAX Network Only

```python
run_info = {
    'run_for_a_model_explicitly': True,
    'model_name': 'LAX'
}
```

#### Knee-Aware DOPLAX (Combined)

```python
run_info = {
    'run_for_a_model_explicitly': False,
    'model_name': 'Bagging_u'
}
```

### STEP3: Running the desired main file

You can run the scripts using the default settings, or you can run them with your own parameter values:
```python
# Train on XJTU dataset
python ./main_XJTU.py

# Train on TJU dataset
python ./main_TJU.py

# Train on MIT dataset
python ./main_MIT.py

# Train on HUST dataset
python ./main_HUST.py
```

### STEP4: Training Bagging MLP (Knee-Aware Fusion)

After training DeepOPINN and LAX individually, train the Bagging MLP fusion module:

```bash
# Train all models across all datasets
python run_all_bagging.py
```

This script:
1. Trains DeepOPINN for all 4 datasets (skips if checkpoint exists)
2. Trains LAX for all 4 datasets (skips if checkpoint exists)
3. Trains Bagging MLP for all 4 datasets (skips if checkpoint exists)

**Key Bagging MLP Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `bag_hidden_dim` | Hidden layer dimensions for MLP | `[10, 10]` (HUST: `[50, 50]`) |
| `bagging_NN_lr` | Learning rate for Bagging MLP | Dataset-specific |
| `mono_bag` | Monotonicity loss weight | Dataset-specific |

---

## 🚀 Inference
To run inference with the models, first download the pretrained weights from [Zenodo](https://doi.org/10.5281/zenodo.17883282), then load them using the provided load_model method in each module provided for each model.

---

## 📈 Visualization

### Generating Knee-Aware DOPLAX Plots

Generate comprehensive plots showing all three sub-model predictions:

```bash
python plot_all_datasets.py
```

This generates 17 plots in `outputs/plots/`:

| Plot | Description |
|------|-------------|
| `kneaware_capacity_per_battery_{ds}.png` (×4) | Per-battery capacity curves: True vs DeepOPINN vs LAX vs Knee-Aware DOPLAX |
| `kneaware_scatter_per_battery_{ds}.png` (×4) | Per-battery scatter: 3 columns (DeepOPINN, LAX, Knee-Aware DOPLAX) |
| `kneaware_combined_overview.png` | 2×2 grid: best battery from each dataset, all 3 models overlaid |
| `kneaware_predicted_vs_actual_all.png` | 3×4 grid: all batteries overlaid per dataset per model |
| `kneaware_model_comparison_bars.png` | Cross-dataset RMSE/MAE/MAPE/R bar charts for all 3 models |
| `kneaware_metrics_table.png` | Color-coded metrics table figure |
| `kneaware_metrics_summary.csv` | Raw metrics data |
| `training_loss_{ds}.png` (×4) | Training/validation loss curves |

### Sample Results

<p align="center">
  <img src="./figures/f_comparison_plots.svg" alt="SOH Degradation Curves" width="90%"/>
</p>
*SOH degradation curves comparing PINN (left) and DOPLAX (right) across four datasets.*

---

## 🔍 Knee Point Detection

The knee point is the inflection in capacity degradation where fade accelerates. Knee-Aware DOPLAX uses this as a feature input.

### Knee Point Distance Formula

```python
knee_point_distance[i] = (knee_index - i) / knee_index   # if i < knee_index
                         0                                  # if i >= knee_index
```

- **1.0** at cycle 0 (far from knee)
- **0.0** at/after the knee point
- Linearly decreasing between them

### Knee Point Detection Algorithm

1. Fit linear regression to the first `le` cycles (linear region)
2. Extrapolate that line across all cycles
3. Compute residuals: `|actual - fitted|`
4. Knee point = first cycle where `residuals >= (n_percent/100) × fitted`

### Optimizing n_percent

The threshold `n_percent` is optimized via grid search:

```python
from knee_point_detection import optimize_n_percent

optimal_n = optimize_n_percent(all_soh_trajectories)
```

**Scoring function** (lower = better):
```
score = (1 - R²_before) + R²_after + |1 - slope_ratio|
```

- `(1 - R²_before)`: penalizes poor linear fit before knee
- `R²_after`: penalizes good linear fit after knee
- `|1 - slope_ratio|`: penalizes weak slope change

---

## 🔧 Loss Functions

DOPLAX uses a composite loss function combining multiple objectives:

### DeepOPINN Loss

$$
\mathcal{L}_{DOP} = \mathcal{L}_{Data} + \alpha \mathcal{L}_{PDE} + \beta \mathcal{L}_{Mono}
$$

### LAX Loss

$$
\mathcal{L}_{LAX} = \mathcal{L}_{Data} + \beta \mathcal{L}_{Mono} + \gamma \mathcal{L}_{Dual}
$$

### Bagging MLP Loss

$$
\mathcal{L}_{Bag} = \mathcal{L}_{Data}^{bag} + \mu \mathcal{L}_{Mono}^{bag}
$$

Where:
- $\mathcal{L}_{Data}^{bag}$: MSE between Bagging MLP prediction and true capacity
- $\mathcal{L}_{Mono}^{bag}$: Monotonicity constraint on Bagging MLP predictions

### Individual Components

**Data Loss** — Supervised SOH prediction
$$
\mathcal{L}_{Data} = \sum_{i=1}^{N} (\hat{u}(t_i) - u^{tar}(t_i))^2
$$
**PDE Loss** — Physics residual enforcement

$$
\mathcal{L}_{PDE} = \sum_{i=1}^{N} \|H(x_i, t_i)\|^2
$$


**Monotonicity Loss** — Capacity fade constraint
$$
\mathcal{L}_{Mono} = \sum_{i=1}^{N} \text{ReLU}\left\{\left(\hat{u}(t_{i+1})-\hat{u}(t_i)\right)\left(u^{\text{tar}}(t_{i+1})-u^{\text{tar}}(t_i)\right)\right\}
$$
**Dual Loss** — Convexity preservation
$$
\mathcal{L}_{Dual} = \|\Phi^* - (\Phi^*)^{**}\|
$$

---

## 📊 Metrics Summary

### Knee-Aware DOPLAX Performance (All 4 Datasets)

| Dataset | Model | RMSE | MAE | MAPE | Pearson R |
|---------|-------|------|-----|------|-----------|
| XJTU | DeepOPINN | 0.0153 | 0.0110 | 1.20% | 0.9146 |
| XJTU | LAX | 0.0202 | 0.0164 | 1.76% | 0.8779 |
| XJTU | **Knee-Aware DOPLAX** | **0.0169** | **0.0129** | **1.40%** | **0.9195** |
| TJU | DeepOPINN | 0.0085 | 0.0065 | 0.87% | 0.9970 |
| TJU | LAX | 0.0559 | 0.0494 | 6.43% | 0.9555 |
| TJU | **Knee-Aware DOPLAX** | **0.0062** | **0.0049** | **0.63%** | **0.9981** |
| MIT | DeepOPINN | 0.0081 | 0.0058 | 0.61% | 0.9624 |
| MIT | LAX | 0.0111 | 0.0081 | 0.85% | 0.9338 |
| MIT | **Knee-Aware DOPLAX** | **0.0078** | **0.0056** | **0.59%** | **0.9638** |
| HUST | DeepOPINN | 0.0093 | 0.0073 | 0.75% | 0.9937 |
| HUST | LAX | 0.0546 | 0.0346 | 3.74% | 0.7085 |
| HUST | **Knee-Aware DOPLAX** | **0.0087** | **0.0069** | **0.70%** | **0.9937** |

---

## 📁 Project Structure

```
DOPLAX-SOH-main 4/
├── Model/
│   ├── PI_nets/
│   │   ├── DeepOPINN.py          # DeepOPINN model
│   │   └── LAX.py                # LAX OptimizationNetwork
│   ├── Auxiliary_nets/
│   │   ├── Solution_u.py         # Solution_u network
│   │   └── MLP.py                # Base MLP
│   └── Combination_nets/
│       └── Bagging_u.py          # MLP_Bagging_NN (Knee-Aware fusion)
├── dataloader/
│   ├── dataloader.py             # Data loading with knee_point_distance
│   └── data_helper.py            # Train/test splits
├── utils/
│   └── arguments.py              # Hyperparameter definitions
├── knee_point_detection.py       # KneePointDetector & optimize_n_percent
├── run_all_bagging.py            # Unified training orchestrator
├── run_all_datasets.py           # LAX training orchestrator
├── plot_all_datasets.py          # Knee-Aware DOPLAX plotting
├── main_XJTU.py                  # XJTU dataset main
├── main_TJU.py                   # TJU dataset main
├── main_MIT.py                   # MIT dataset main
├── main_HUST.py                  # HUST dataset main
├── data/
│   └── Processed/                # Preprocessed CSV files
└── outputs/
    ├── plots/                    # Generated plots
    ├── XJTU/
    │   ├── Bagging/best_model.pth
    │   ├── DeepOPINN/best_model.pth
    │   └── best_weights.pth      # LAX weights
    ├── TJU/
    ├── MIT/
    └── HUST/
```

---

## 📖 References

Key references used in this work:

1. **DeepONet**: Lu, L., Jin, P., & Karniadakis, G. E. (2019). DeepONet: Learning nonlinear operators.
2. **PINN**: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks.
3. **Hamilton-Jacobi**: Darbon, J., & Meng, T. (2020). Neural network architectures for viscosity solutions.
4. **Battery PINN**: Wang, F., et al. (2024). Physics-informed neural network for lithium-ion battery degradation.
