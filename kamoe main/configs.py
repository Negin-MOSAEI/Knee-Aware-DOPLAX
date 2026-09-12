from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class TCNConfig:
    """Hyperparameters for Temporal Convolutional Network (TCN - Knee Distance Predictor)."""
    dataset: str
    num_inputs: int = 17
    num_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 64, 32])
    kernel_size: int = 3
    dropout: float = 0.2
    output_dim: int = 1
    lr: float = 0.001
    warmup_lr: float = 0.0001
    final_lr: float = 1e-6
    warmup_epochs: int = 20
    weight_decay: float = 1e-4
    epochs: int = 300
    early_stop: int = 40
    batch_size: int = 1  # 1 battery trajectory per batch


@dataclass
class DeepOPINNConfig:
    """Hyperparameters for Deep Operator Physics-Informed Neural Network."""
    dataset: str
    input_dim: int = 17
    dim_x: int = 1
    hidden_dim: int = 80
    layers_num: int = 5
    dropout: float = 0.126
    
    # Learning rates
    lr: float = 0.00625
    warmup_lr: float = 0.00627
    final_lr: float = 1.60e-6
    lr_F: float = 0.00882
    warmup_epochs: int = 30
    
    # Loss weights
    alpha: float = 0.11809  # PDE residual weight
    beta: float = 0.01596   # Monotonicity loss weight
    
    # Training
    epochs: int = 2000
    early_stop: int = 80
    batch_size: int = 1


@dataclass
class LAXConfig:
    """Hyperparameters for Hopf-Lax Hamilton-Jacobi Solver Network."""
    dataset: str
    x_dim: int = 16
    y_dim: int = 16
    h_dim: int = 49
    
    # Architectural blocks
    distance_block: str = "SumProductNetwork"
    center_block: str = "PhI"
    time_block: str = "theta"
    s_block: str = "MLP"
    
    # Sub-layer configurations
    inside_h_star_layers: List[int] = field(default_factory=lambda: [64, 32])
    inside_phi_layers: List[int] = field(default_factory=lambda: [64, 32])
    inside_g_layers: List[int] = field(default_factory=lambda: [16])
    inside_S_MLP_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    inside_betan_layers: List[int] = field(default_factory=lambda: [16, 8])
    inside_distance_block_MLP_layers: List[int] = field(default_factory=lambda: [64])
    inside_theta_layers: List[int] = field(default_factory=lambda: [32])
    
    # Learning rates
    lr_net: float = 0.040
    lr_y: float = 0.080
    warmup_epochs_net: int = 100
    warmup_epochs_y: int = 100
    min_lr_net: float = 1e-5
    min_lr_y: float = 1e-4
    
    # Loss weights
    betha_LAX: float = 0.0256       # Monotonicity weight
    theta_LAX: float = 0.0          # MAPE weight
    dual_LAX: float = 0.0           # Dual consistency loss weight
    beta_LAX: float = 0.0           # PDE weight
    zeta_LAX: float = 1.0           # MSE weight
    kata_LAX: float = 0.0           # MAE weight
    
    # Epochs
    epoch_net: int = 350
    epoch_y: int = 600
    epoch_th: int = 3000
    patience: int = 50
    min_delta: float = 1e-6
    epochs: int = 2000
    batch_size: int = 1


@dataclass
class FusionMLPConfig:
    """Hyperparameters for Fusion MLP / Bagging Network."""
    dataset: str
    input_dim: int = 2
    hidden_dims: List[int] = field(default_factory=lambda: [10, 10])
    dropout: float = 0.126
    include_knee_feature: bool = True
    
    # Learning rates
    bagging_lr: float = 0.020
    warmup_lr: float = 0.00627
    final_lr: float = 1.60e-6
    warmup_epochs: int = 30
    
    # Loss weights
    mono_bag: float = 0.40  # Monotonicity weight for fusion
    
    # Training
    epochs: int = 2000
    early_stop: int = 80
    batch_size: int = 1


# ==============================================================================
# DATASET-SPECIFIC PRESETS (XJTU, TJU, MIT, HUST)
# ==============================================================================

DATASET_CONFIGS = {
    # --------------------------------------------------------------------------
    # 1. XJTU Dataset
    # --------------------------------------------------------------------------
    "XJTU": {
        "TCN": TCNConfig(
            dataset="XJTU",
            num_inputs=17,
            num_channels=[32, 32, 32, 32, 32, 32, 32, 32, 32],
            kernel_size=9,
            dropout=0.15,
            lr=0.001,
            epochs=300,
            early_stop=40,
        ),
        "DeepOPINN": DeepOPINNConfig(
            dataset="XJTU",
            input_dim=17,
            hidden_dim=80,
            layers_num=5,
            dropout=0.1263514868613022,
            lr=0.006252096490545448,
            warmup_lr=0.006268229580119017,
            final_lr=1.5973286128658127e-06,
            lr_F=0.00881767573907948,
            warmup_epochs=30,
            alpha=0.11809194837918663,
            beta=0.015956048434866418,
            epochs=2000,
            early_stop=80,
        ),
        "LAX": LAXConfig(
            dataset="XJTU",
            h_dim=49,
            distance_block="SumProductNetwork",
            center_block="PhI",
            time_block="theta",
            lr_net=0.040,
            lr_y=0.080,
            betha_LAX=0.0256,
            theta_LAX=0.0,
            dual_LAX=0.0,
            beta_LAX=0.0,
            epoch_net=350,
            epoch_y=600,
            epochs=2000,
        ),
        "FusionMLP": FusionMLPConfig(
            dataset="XJTU",
            hidden_dims=[10, 10],
            dropout=0.1263514868613022,
            bagging_lr=0.020,
            warmup_lr=0.006268229580119017,
            final_lr=1.5973286128658127e-06,
            warmup_epochs=30,
            mono_bag=0.40,
            epochs=2000,
            early_stop=80,
            include_knee_feature=True,
        ),
    },

    # --------------------------------------------------------------------------
    # 2. TJU Dataset
    # --------------------------------------------------------------------------
    "TJU": {
        "TCN": TCNConfig(
            dataset="TJU",
            num_inputs=17,
            num_channels=[32, 32, 32, 32, 32, 32, 32, 32, 32],
            kernel_size=9,
            dropout=0.15,
            lr=0.001,
            epochs=300,
            early_stop=40,
        ),
        "DeepOPINN": DeepOPINNConfig(
            dataset="TJU",
            input_dim=17,
            hidden_dim=128,
            layers_num=5,
            dropout=0.12374924038427233,
            lr=0.001221299541334721,
            warmup_lr=0.006456520169381107,
            final_lr=2.1099224334559236e-05,
            lr_F=0.0004916804735455082,
            warmup_epochs=50,
            alpha=0.5999950741786956,
            beta=0.01804601857570287,
            epochs=2000,
            early_stop=80,
        ),
        "LAX": LAXConfig(
            dataset="TJU",
            h_dim=58,
            distance_block="SumProductNetwork",
            center_block="PhI",
            time_block="theta",
            lr_net=0.024314095112883502,
            lr_y=0.080,
            betha_LAX=0.7447313058846572,
            theta_LAX=0.9290600824788182,
            dual_LAX=0.008827241117241124,
            beta_LAX=0.0,
            epoch_net=350,
            epoch_y=600,
            epochs=2000,
        ),
        "FusionMLP": FusionMLPConfig(
            dataset="TJU",
            hidden_dims=[10, 10],
            dropout=0.12374924038427233,
            bagging_lr=0.006456520169381107,
            warmup_lr=0.006456520169381107,
            final_lr=2.1099224334559236e-05,
            warmup_epochs=50,
            mono_bag=0.01804601857570287,
            epochs=2000,
            early_stop=80,
            include_knee_feature=True,
        ),
    },

    # --------------------------------------------------------------------------
    # 3. MIT Dataset
    # --------------------------------------------------------------------------
    "MIT": {
        "TCN": TCNConfig(
            dataset="MIT",
            num_inputs=17,
            num_channels=[32, 32, 32, 32, 32, 32, 32, 32, 32],
            kernel_size=9,
            dropout=0.15,
            lr=0.001,
            epochs=300,
            early_stop=40,
        ),
        "DeepOPINN": DeepOPINNConfig(
            dataset="MIT",
            input_dim=17,
            hidden_dim=128,
            layers_num=5,
            dropout=0.1235582146396002,
            lr=0.00011720460436102193,
            warmup_lr=0.006158883939912325,
            final_lr=1.447250132592353e-06,
            lr_F=2.181510135329394e-05,
            warmup_epochs=50,
            alpha=0.8843241457826115,
            beta=0.02840429623105723,
            epochs=2000,
            early_stop=80,
        ),
        "LAX": LAXConfig(
            dataset="MIT",
            h_dim=30,
            distance_block="SumProductNetwork",
            center_block="PhI",
            time_block="theta",
            lr_net=0.011935902068420634,
            lr_y=0.080,
            betha_LAX=0.5927778785956674,
            theta_LAX=0.4808822091755273,
            dual_LAX=0.0003537265723182956,
            beta_LAX=0.0,
            epoch_net=350,
            epoch_y=600,
            epochs=2000,
        ),
        "FusionMLP": FusionMLPConfig(
            dataset="MIT",
            hidden_dims=[10, 10],
            dropout=0.1235582146396002,
            bagging_lr=0.006663648646643286,
            warmup_lr=0.006158883939912325,
            final_lr=1.447250132592353e-06,
            warmup_epochs=50,
            mono_bag=0.014599210344919067,
            epochs=2000,
            early_stop=80,
            include_knee_feature=True,
        ),
    },

    # --------------------------------------------------------------------------
    # 4. HUST Dataset
    # --------------------------------------------------------------------------
    "HUST": {
        "TCN": TCNConfig(
            dataset="HUST",
            num_inputs=17,
            num_channels=[32, 32, 32, 32, 32, 32, 32, 32, 32],
            kernel_size=9,
            dropout=0.15,
            lr=0.001,
            epochs=300,
            early_stop=40,
        ),
        "DeepOPINN": DeepOPINNConfig(
            dataset="HUST",
            input_dim=17,
            hidden_dim=80,
            layers_num=4,
            dropout=0.10014057317121909,
            lr=0.0007215089581483758,
            warmup_lr=0.005237775841792184,
            final_lr=1.747286485446226e-05,
            lr_F=4.8990846543190585e-05,
            warmup_epochs=30,
            alpha=0.2795906470184894,
            beta=0.1308092300047794,
            epochs=2000,
            early_stop=80,
        ),
        "LAX": LAXConfig(
            dataset="HUST",
            h_dim=58,
            distance_block="SumProductNetwork",
            center_block="PhI",
            time_block="theta",
            lr_net=0.014526149878088328,
            lr_y=0.080,
            betha_LAX=0.846945564070922,
            theta_LAX=0.809772111877841,
            dual_LAX=0.570865721084567,
            beta_LAX=0.0,
            epoch_net=350,
            epoch_y=600,
            epochs=2000,
        ),
        "FusionMLP": FusionMLPConfig(
            dataset="HUST",
            hidden_dims=[10, 10],
            dropout=0.10014057317121909,
            bagging_lr=0.00022334184499247844,
            warmup_lr=0.005237775841792184,
            final_lr=1.747286485446226e-05,
            warmup_epochs=30,
            mono_bag=0.9933428738041119,
            epochs=2000,
            early_stop=80,
            include_knee_feature=True,
        ),
    },
}


def get_config(model_name: str, dataset_name: str) -> Any:
    """
    Retrieves the specific dataclass configuration for a given model and dataset.

    :param model_name: 'TCN', 'DeepOPINN', 'LAX', or 'FusionMLP'
    :param dataset_name: 'XJTU', 'TJU', 'MIT', or 'HUST'
    :return: Dataclass instance containing hyperparameter settings
    """
    d_name = dataset_name.upper()
    if d_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: {list(DATASET_CONFIGS.keys())}")

    dataset_models = DATASET_CONFIGS[d_name]
    
    # Normalize model name lookup
    m_norm = model_name.upper()
    for k, v in dataset_models.items():
        if k.upper() == m_norm or m_norm in k.upper():
            return v

    raise ValueError(f"Unknown model '{model_name}'. Supported for {d_name}: {list(dataset_models.keys())}")


def get_tcn_config(dataset_name: str) -> TCNConfig:
    return get_config("TCN", dataset_name)


def get_deepopinn_config(dataset_name: str) -> DeepOPINNConfig:
    return get_config("DeepOPINN", dataset_name)


def get_lax_config(dataset_name: str) -> LAXConfig:
    return get_config("LAX", dataset_name)


def get_fusion_config(dataset_name: str) -> FusionMLPConfig:
    return get_config("FusionMLP", dataset_name)
