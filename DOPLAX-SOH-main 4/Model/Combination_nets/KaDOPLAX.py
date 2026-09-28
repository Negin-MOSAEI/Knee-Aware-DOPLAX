import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union
from Model.Backbones.bagging_mlp import BaggingMLP
from Model.PI_nets.DeepOPINN import Model as DeepOPINN
from Model.PI_nets.LAX import OptimizationNetwork as LAXModel
from Model.utils.learnable_smoother import ConstrainedLearnableSmoother, tv_regularization_loss


class KaDOPLAX(nn.Module):
    """
    Knee-Aware Deep Operator Lax-Friedrichs (KaDOPLAX) Framework.

    Combines:
      - DeepOPINN (Physics-Informed Deep Operator Network)
      - LAX (Lax-Friedrichs Conservation Law Numerical Solver)
      - Residual Gated MoE Fusion with Entropy Annealing
      - Self-Adaptive Homoscedastic Uncertainty Weighting (Kendall & Gal)
      - Optional Constrained Learnable Smoother (TV regularized)
    """
    def __init__(self, deepopinn_model: DeepOPINN, lax_model: LAXModel, bagging_mlp=None, enable_smoother: bool = False):
        super(KaDOPLAX, self).__init__()
        self.deepopinn = deepopinn_model
        self.lax = lax_model

        # Legacy BaggingMLP parameter kept for backwards compatibility
        self.fusion_mlp = bagging_mlp

        # Residual gated MoE routing network (Predicts Mean weighting)
        self.moe_gate = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

        # Variance Head for Aleatoric Uncertainty (Predicts log(sigma^2))
        self.variance_head = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        # Optional constrained learnable smoother
        self.enable_smoother = enable_smoother
        self.smoother = ConstrainedLearnableSmoother(kernel_size=5)

        # Self-adaptive loss log-variance parameters: s = log(sigma^2)
        # Kendall & Gal (CVPR 2018) homoscedastic multi-task uncertainty weighting
        self.s_data = nn.Parameter(torch.tensor(0.0))
        self.s_pde = nn.Parameter(torch.tensor(0.0))
        self.s_mono = nn.Parameter(torch.tensor(0.0))
        self.s_fusion = nn.Parameter(torch.tensor(0.0))

        # Initially freeze pre-trained expert backbones
        self.experts_frozen = True
        self.freeze_experts()

        # Run LAX in inference mode
        self.lax.inference_mode = True

        # Track latest gating outputs
        self.last_gate_weights = None
        self.last_entropy = None

    def freeze_experts(self):
        """Freeze both DeepOPINN and LAX expert parameters (Stage 1 warmup)."""
        self.experts_frozen = True
        for param in self.deepopinn.parameters():
            param.requires_grad = False
        for param in self.lax.parameters():
            param.requires_grad = False

    def unfreeze_experts(self, unfreeze_lax: bool = False):
        """
        Unfreeze expert parameters for Stage 2 physics-preserving joint fine-tuning.
        By default unfreezes DeepOPINN; LAX can optionally be unfrozen.
        """
        self.experts_frozen = False
        for param in self.deepopinn.parameters():
            param.requires_grad = True
        if unfreeze_lax:
            for param in self.lax.parameters():
                param.requires_grad = True

    def compute_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Computes the Shannon entropy of the MoE gating distribution:
            H(w) = - sum_k w_k * log(w_k + eps)
        Maximum entropy for 2 experts is ln(2) ~= 0.6931.
        """
        eps = 1e-8
        entropy = -torch.sum(weights * torch.log(weights + eps), dim=-1)
        return entropy.mean()

    def get_effective_loss_weights(self) -> Dict[str, float]:
        """Returns the current effective multi-task loss weights exp(-s_i)."""
        with torch.no_grad():
            return {
                "data": float(torch.exp(-self.s_data).item()),
                "pde": float(torch.exp(-self.s_pde).item()),
                "mono": float(torch.exp(-self.s_mono).item()),
                "fusion": float(torch.exp(-self.s_fusion).item()),
            }

    def compute_multitask_loss(
        self,
        l_data: torch.Tensor,
        l_pde: torch.Tensor,
        l_mono: torch.Tensor,
        l_fusion: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Self-adaptive multi-task loss with learned log-variances s_i:
            L_total = sum_i [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]
        """
        term_data = 0.5 * torch.exp(-self.s_data) * l_data + 0.5 * self.s_data
        term_pde = 0.5 * torch.exp(-self.s_pde) * l_pde + 0.5 * self.s_pde
        term_mono = 0.5 * torch.exp(-self.s_mono) * l_mono + 0.5 * self.s_mono
        term_fusion = 0.5 * torch.exp(-self.s_fusion) * l_fusion + 0.5 * self.s_fusion

        total_loss = term_data + term_pde + term_mono + term_fusion
        weights_dict = self.get_effective_loss_weights()
        return total_loss, weights_dict

    def forward(
        self,
        features: torch.Tensor,
        kpd: torch.Tensor,
        cycle_t: torch.Tensor,
        epoch: Optional[int] = None,
        return_all: bool = False,
        return_gate_info: bool = False,
        apply_smoothing: bool = False
    ):
        """
        Forward pass through KaDOPLAX.

        Args:
            features: [B, window_size, num_features]
            kpd: [B, 1]
            cycle_t: [B, 1]
            epoch: Optional epoch integer for LAX
            return_all: If True, returns (final_out, u_1, u_2, std_pred) legacy tuple
            return_gate_info: If True, returns (final_out, weights, entropy)
            apply_smoothing: If True, passes output through ConstrainedLearnableSmoother
        """
        batch_size = features.shape[0]
        features_flat = features.reshape(batch_size, -1)

        # Context manager depending on expert freezing status
        if self.experts_frozen:
            with torch.no_grad():
                u_1, u_2 = self._forward_experts(features, features_flat, cycle_t, epoch)
        else:
            u_1, u_2 = self._forward_experts(features, features_flat, cycle_t, epoch)

        # Ensure correct shapes: experts must be [B, 1]
        if u_1.dim() > 2:
            u_1 = u_1.view(batch_size, -1)
        if u_2.dim() > 2:
            u_2 = u_2.view(batch_size, -1)
        if kpd.dim() > 2:
            kpd = kpd.view(batch_size, -1)
        if u_1.dim() == 1:
            u_1 = u_1.unsqueeze(1)
        if u_2.dim() == 1:
            u_2 = u_2.unsqueeze(1)

        # MoE Gating for Mean SOH
        gate_input = torch.cat([u_1, u_2], dim=-1)
        weights = self.moe_gate(gate_input)  # [B, 2]
        mean_out = weights[:, 0:1] * u_1 + weights[:, 1:2] * u_2
        
        # Predict Log-Variance for Aleatoric Uncertainty
        log_var_out = self.variance_head(gate_input)

        # Optional smoothing
        if (self.enable_smoother or apply_smoothing) and mean_out.shape[0] > 5:
            mean_out = self.smoother(mean_out)

        # Compute entropy
        entropy = self.compute_entropy(weights)
        self.last_gate_weights = weights.detach()
        self.last_entropy = entropy.detach()

        if return_all:
            return mean_out, u_1, u_2, log_var_out
        elif return_gate_info:
            return mean_out, log_var_out, weights, entropy
        else:
            return mean_out, log_var_out

    def _forward_experts(self, features, features_flat, cycle_t, epoch):
        # DeepOPINN forward
        try:
            u_1_tuple = self.deepopinn.predict(features_flat)
            u_1 = u_1_tuple[1] if isinstance(u_1_tuple, tuple) else u_1_tuple
        except Exception:
            u_1_tuple = self.deepopinn.forward_deepopinn(features_flat)
            u_1 = u_1_tuple[0] if isinstance(u_1_tuple, tuple) else u_1_tuple

        # LAX forward
        last_x = features[:, -1, :3]
        epoch_val = epoch if epoch is not None else 1000
        try:
            u_2 = self.lax(x=last_x, t=cycle_t, epoch=epoch_val, return_f=False)
        except Exception:
            u_2 = self.lax(x=last_x, t=cycle_t, epoch=epoch_val)

        if isinstance(u_2, tuple):
            u_2 = u_2[0]

        return u_1, u_2

    def predict_with_uq(
        self,
        features: torch.Tensor,
        kpd: torch.Tensor,
        cycle_t: torch.Tensor,
        n_samples: int = 30,
        epoch: Optional[int] = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout Uncertainty Quantification.

        Enables dropout in moe_gate during inference to sample epistemic uncertainty.

        Returns:
            mean_pred: [B, 1] Ensemble mean SOH
            std_pred:  [B, 1] Epistemic uncertainty (standard deviation)
        """
        self.eval()
        # Enable dropout specifically in moe_gate and variance_head
        for m in self.moe_gate.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        for m in self.variance_head.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        mean_preds = []
        aleatoric_vars = []
        with torch.no_grad():
            for _ in range(n_samples):
                mean_out, log_var_out = self.forward(features, kpd, cycle_t, epoch=epoch)
                mean_preds.append(mean_out.unsqueeze(0))
                aleatoric_vars.append(torch.exp(log_var_out).unsqueeze(0))

        # Restore eval mode for dropout
        self.moe_gate.eval()
        self.variance_head.eval()

        # Stack over sample dimension [S, B, 1]
        all_means = torch.cat(mean_preds, dim=0)
        all_aleatoric = torch.cat(aleatoric_vars, dim=0)
        
        final_mean = torch.mean(all_means, dim=0)
        epistemic_var = torch.var(all_means, dim=0)
        aleatoric_var = torch.mean(all_aleatoric, dim=0)
        
        total_std = torch.sqrt(epistemic_var + aleatoric_var)

        return final_mean, total_std
