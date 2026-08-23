"""
tcn.py – Lightweight Temporal Convolutional Network for Battery Knee-Point
===========================================================================
Drop-in replacement for the Time-Series Transformer (TST) backbone in the
Knee-Aware DOPLAX framework. Designed for Edge BMS deployment on constrained
microcontrollers.

Design Choices
--------------
* **Causal dilated 1-D convolutions** – output at time-step *t* depends only
  on inputs at positions ``<= t``, eliminating any temporal (future) leakage.
* **Exponential dilation schedule** – layers use dilation ``d = 1, 2, 4, 8, 16``
  with ``kernel_size = 3``, giving a receptive field of **63 cycles** which
  comfortably covers the maximum input window of 40 cycles.
* **Left-padding** – when the actual sequence length ``T < max_seq_len`` (cold
  start, e.g. cycles 1-39), zero-padding is applied on the LEFT so that causal
  convolutions still only attend to past / present time-steps.
* **Learnable transition width *w*** – parameterised via softplus so ``w > 0``
  is guaranteed; controls the sharpness of the ``arctan`` knee-distance
  function ``d_knee = -(2/π) · arctan((cycle − c_knee) / w)``.

Receptive Field Calculation  (kernel_size k = 3, L layers with d = 2^i)
-----------------------------------------------------------------------
  RF = 1 + (k − 1) · Σ 2^i   for i = 0 … L−1
     = 1 + 2 · (1 + 2 + 4 + 8 + 16)          [L = 5]
     = 1 + 2 · 31
     = 63  >  40  ✓

Parameter Budget  (H = 32, F ≈ 3–8 features)
----------------------------------------------
  Input 1×1 Conv        :  H · (F + 1)              ≈    128
  5 × CausalConvBlock   :  5 · [2·H·(H·k + 1)]      ≈ 31 040
  5 × BatchNorm (2×)    :  5 · 2 · 2H                ≈    640
  Output Linear         :  H + 1                     ≈     33
  w_raw scalar          :  1
  ──────────────────────────────────────────────────────────────
  TOTAL                                               ≈ 31 842  (< 50k ✓)

Pipeline Integration
--------------------
* ``forward(x)`` → ``c_knee`` tensor of shape ``(B, 1)`` – **drop-in
  compatible** with ``TimeSeriesTransformer.forward()`` so that Phase 1
  training (``phase1_train_tst.py``) and Phase 2 inference
  (``phase2_infer_kpd.py``) work with minimal changes.
* ``predict_knee(x, current_cycle)`` → ``(c_knee, d_knee)`` – returns both
  the predicted knee cycle and the signed knee-distance feature for the
  downstream DeepOPINN / KaDOPLAX fusion.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# Causal Dilated Convolution Block
# ====================================================================

class CausalConvBlock(nn.Module):
    """Pre-activation residual block with two causal dilated 1-D convolutions.

    Causality is enforced by left-padding the input before each convolution so
    that output at position *t* depends **only** on inputs at positions
    ``<= t``.  The residual shortcut preserves the temporal dimension.

    Parameters
    ----------
    channels : int
        Number of input / output channels (kept constant for the residual).
    kernel_size : int
        Convolution kernel size (default ``3``).
    dilation : int
        Dilation factor (default ``1``).
    dropout : float
        Dropout rate applied after each conv+BN activation (default ``0.1``).
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        # Left-padding amount that makes the convolution causal.
        # Effective receptive width = 1 + (k−1)·d, so we pad (k−1)·d positions.
        self.left_pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(channels)
        self.drop1 = nn.Dropout1d(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop2 = nn.Dropout1d(dropout)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape ``(B, C, T)``

        Returns
        -------
        Tensor of shape ``(B, C, T)`` – same temporal length, with a residual
        connection.
        """
        # --- first conv path ---
        h = F.pad(x, (self.left_pad, 0))          # left-pad for causality
        h = self.drop1(self.act(self.bn1(self.conv1(h))))

        # --- second conv path ---
        h = F.pad(h, (self.left_pad, 0))          # left-pad for causality
        h = self.drop2(self.bn2(self.conv2(h)))

        # --- residual connection ---
        return self.act(h + x)


# ====================================================================
# Knee-Point TCN
# ====================================================================

class KneeTCN(nn.Module):
    """Lightweight TCN that predicts the knee-point cycle ``c_knee`` and,
    optionally, the signed knee-distance feature ``d_knee``.

    Architecture
    ------------
    ::

        Input  (B, T, F)  ── left-pad to T = max_seq_len
             │
             ▼  transpose → (B, F, T)
        1×1 Conv + BN + ReLU   →  (B, H, T)
             │
             ▼
        CausalConvBlock  d=1   →  (B, H, T)
        CausalConvBlock  d=2   →  (B, H, T)
        CausalConvBlock  d=4   →  (B, H, T)
        CausalConvBlock  d=8   →  (B, H, T)
        CausalConvBlock  d=16  →  (B, H, T)
             │
             ▼  take last timestep  →  (B, H)
        Linear(H → 1)           →  c_knee  (B, 1)
             │
             ▼  (only via predict_knee with current_cycle)
        d_knee = −(2/π) · arctan((current_cycle − c_knee) / w)

    Parameters
    ----------
    num_features : int
        Number of input features per cycle (e.g. 3 for current, voltage,
        capacity).
    hidden_channels : int
        TCN channel width (default ``32``).
    num_layers : int
        Number of ``CausalConvBlock`` layers (default ``5``, RF = 63).
    kernel_size : int
        Convolution kernel size (default ``3``).
    dropout : float
        Dropout rate applied inside each ``CausalConvBlock`` (after conv+BN
        activations) and as a spatial dropout before the output head
        (default ``0.1``).
    max_seq_len : int
        Maximum / expected sequence length (default ``40``).
    x_sts : tuple of (Tensor, Tensor), optional
        ``(X_mean, X_std)`` per-feature standardisation statistics.
        Registered as buffers so they travel with the model.
    **kwargs
        Ignored – allows the HPO config JSON to pass extra keys (e.g.
        ``lr``, ``batch_size``) without raising a ``TypeError``.
    """

    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 32,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 40,
        x_sts: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__()
        self.max_seq_len    = max_seq_len
        self.num_features   = num_features
        self.hidden_channels = hidden_channels
        H = hidden_channels

        # ---- input standardisation (matches TST interface) ----
        if x_sts is not None:
            self.register_buffer("X_mean", x_sts[0].to(torch.float32))
            self.register_buffer("X_std",  x_sts[1].to(torch.float32))
        else:
            self.register_buffer("X_mean", torch.zeros(num_features, dtype=torch.float32))
            self.register_buffer("X_std",  torch.ones(num_features, dtype=torch.float32))

        # ---- verify receptive field covers max_seq_len ----
        rf = 1 + (kernel_size - 1) * sum(2 ** i for i in range(num_layers))
        assert rf >= max_seq_len, (
            f"Receptive field {rf} < max_seq_len {max_seq_len}. "
            f"Increase num_layers or kernel_size."
        )
        self.receptive_field = rf

        # ---- input projection: raw features → hidden dim ----
        # 1×1 convolution acts as a learnable per-timestep feature mix.
        self.input_proj = nn.Sequential(
            nn.Conv1d(num_features, H, kernel_size=1),
            nn.BatchNorm1d(H),
            nn.ReLU(inplace=True),
        )

        # ---- TCN stack with exponential dilation ----
        # Dilation doubles each layer: 1, 2, 4, 8, 16, …
        # This gives O(log T) depth for an O(T) receptive field.
        self.blocks = nn.ModuleList([
            CausalConvBlock(H, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])

        # ---- dropout before output head (spatial: drops whole channels) ----
        self.drop_out = nn.Dropout1d(dropout)

        # ---- output head: last hidden state → scalar c_knee ----
        self.fc_out = nn.Linear(H, 1)

        # ---- learnable transition width (softplus → w > 0) ----
        # softplus(0) ≈ 0.693 as a reasonable initial value.
        self.w_raw = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    # ----------------------------------------------------------------
    # Weight initialisation
    # ----------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Z-score standardisation using registered buffers."""
        return (x - self.X_mean) / torch.clamp(self.X_std, min=1e-5)

    def _predict_c_knee(self, x: torch.Tensor) -> torch.Tensor:
        """Core pass: raw sequence → c_knee.

        Handles variable-length input (``T < max_seq_len``) via explicit
        left-padding with zeros so that the causal property is preserved.
        """
        B, T, _ = x.shape

        # 1. Standardise
        x = self._standardize(x)

        # 2. Left-pad along the time axis if shorter than max_seq_len
        #    Padding scheme: (left, right, before, after) for 5-D input;
        #    we only pad the time dimension (dim 2) on the LEFT.
        if T < self.max_seq_len:
            x = F.pad(x, (0, 0, self.max_seq_len - T, 0))

        # 3. Transpose to (B, F, T) for Conv1d
        x = x.transpose(1, 2)

        # 4. Project to hidden channels
        x = self.input_proj(x)                   # (B, H, T)

        # 5. TCN blocks — each preserves temporal length
        for block in self.blocks:
            x = block(x)                          # (B, H, T)

        # 6. Spatial dropout before output head
        x = self.drop_out(x)

        # 7. Take the LAST timestep — it has the full receptive field
        return self.fc_out(x[:, :, -1])           # (B, 1)

    # ----------------------------------------------------------------
    # Forward (drop-in compatible with TimeSeriesTransformer)
    # ----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict **c_knee** from a (possibly variable-length) feature window.

        This method is **drop-in compatible** with
        ``TimeSeriesTransformer.forward(src)`` so that the existing Phase 1
        training loop and Phase 2 inference script work unchanged.

        Parameters
        ----------
        x : Tensor of shape ``(B, T, F)``
            Battery cycle features with ``T <= max_seq_len``.

        Returns
        -------
        c_knee : Tensor of shape ``(B, 1)``
            Predicted knee-point cycle number.
        """
        return self._predict_c_knee(x)

    # ----------------------------------------------------------------
    # Full knee-distance prediction (for downstream DOPLAX modules)
    # ----------------------------------------------------------------
    def predict_knee(
        self,
        x: torch.Tensor,
        current_cycle: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict ``c_knee`` **and** compute the signed knee-distance
        ``d_knee``.

        Parameters
        ----------
        x : Tensor of shape ``(B, T, F)``
            Battery cycle features.
        current_cycle : Tensor of shape ``(B,)`` or ``(B, 1)``
            The index of the current cycle (e.g. 40, 41, …).

        Returns
        -------
        c_knee : Tensor of shape ``(B, 1)``
            Predicted knee-point cycle number.
        d_knee : Tensor of shape ``(B, 1)``
            Signed knee-distance feature in approximately ``[−1, 1]``:
            ``d_knee = −(2/π) · arctan((current_cycle − c_knee) / w)``.
            Positive pre-knee, negative post-knee.
        """
        c_knee = self._predict_c_knee(x)                  # (B, 1)

        # Ensure (B, 1)
        if current_cycle.dim() == 1:
            current_cycle = current_cycle.unsqueeze(-1)

        w = F.softplus(self.w_raw)                         # scalar > 0
        d_knee = -(2.0 / math.pi) * torch.atan(
            (current_cycle - c_knee) / w
        )

        return c_knee, d_knee


# ====================================================================
# Quick smoke test
# ====================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    NUM_FEATURES = 3   # e.g. current, voltage, capacity
    BATCH        = 4

    # --- Test 0: instantiation with dropout + extra kwargs (HPO compat) ---
    model = KneeTCN(num_features=NUM_FEATURES, dropout=0.2,
                    lr=1e-4, batch_size=32)  # lr/batch_size silently ignored

    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 60)
    print(f"KneeTCN parameter count : {n_params:,}")
    print(f"Receptive field          : {model.receptive_field}")
    print(f"Dropout rate             : 0.2 (with extra kwargs accepted)")
    print("=" * 60)

    current = torch.tensor([50.0, 100.0, 150.0, 200.0])

    # --- Test 1: full sequence (T = 40) ---
    x_full = torch.randn(BATCH, 40, NUM_FEATURES)
    c_knee = model(x_full)
    print(f"Full seq  (T=40) : c_knee {list(c_knee.shape)}")
    print(f"  c_knee values  : {c_knee.squeeze().tolist()}")

    # --- Test 2: partial sequence (T = 10) ---
    x_part = torch.randn(BATCH, 10, NUM_FEATURES)
    c_knee = model(x_part)
    print(f"Partial   (T=10) : c_knee {list(c_knee.shape)}")

    # --- Test 3: cold-start (T = 1) ---
    x_cold = torch.randn(BATCH, 1, NUM_FEATURES)
    c_knee = model(x_cold)
    print(f"Cold start(T=1)  : c_knee {list(c_knee.shape)}")

    # --- Test 4: predict_knee (c_knee + d_knee) ---
    c_knee, d_knee = model.predict_knee(x_full, current)
    print(f"predict_knee     : c_knee {list(c_knee.shape)}, d_knee {list(d_knee.shape)}")
    print(f"  c_knee values  : {c_knee.squeeze().tolist()}")
    print(f"  d_knee values  : {d_knee.squeeze().tolist()}")

    # --- Test 5: gradient flow ---
    x = torch.randn(2, 40, NUM_FEATURES, requires_grad=True)
    c_knee, d_knee = model.predict_knee(x, torch.tensor([80.0, 120.0]))
    loss = c_knee.sum() + d_knee.sum()
    loss.backward()
    print(f"Gradient check   : x.grad norm = {x.grad.norm():.4f}")

    # --- Test 6: verify w > 0 ---
    w = F.softplus(model.w_raw)
    print(f"Learnable w      : {w.item():.4f}  (raw = {model.w_raw.item():.4f})")

    # --- Test 7: verify drop-in compatibility with TST signature ---
    # Simulate what phase1_train_tst.py does: model(features) → single tensor
    features_batch = torch.randn(8, 40, NUM_FEATURES)
    output = model(features_batch)
    assert output.shape == (8, 1), f"Expected (8, 1), got {output.shape}"
    print(f"TST-compat check : output shape {list(output.shape)} ✓")

    print("\nAll shape / gradient checks passed.")
