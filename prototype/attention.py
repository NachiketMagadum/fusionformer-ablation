"""
Fusion Attention Module for Multivariate Time-Series Anomaly Detection.

Core attention mechanism of the Fusionformer architecture.
Combines attention along two axes:

  1. Time axis  — captures dependencies across time steps.
  2. Channel axis — captures relationships between sensor channels.

The two attention outputs are fused via a learnable weighted sum, letting
the model adaptively balance temporal versus cross-channel information.

Why fusion attention?
    Standard transformer attention operates on one axis only (typically time).
    For industrial multivariate sensor data, both axes matter: a fault might
    manifest as an unusual temporal pattern OR as an unusual relationship
    between sensors. Fusion attention captures both.

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
import torch
import torch.nn as nn


class FusionAttention(nn.Module):
    """
    Dual-axis attention over multivariate time-series.

    Given input of shape (batch, seq_len, n_features), this module:
        1. Projects features into a d_model latent space.
        2. Applies self-attention across the time axis.
        3. Applies self-attention across the channel axis (via transpose).
        4. Fuses the two outputs with learnable weights.
        5. Projects back to original feature dimension.

    Args:
        seq_len:    Length of the input time window (number of time steps).
        n_features: Number of sensor channels in the input.
        d_model:    Internal latent dimension for projections. Larger = more
                    capacity but slower training. Default 32 works well for
                    industrial sensor tasks with ~10 channels.
    """

    def __init__(self, seq_len: int, n_features: int, d_model: int = 32):
        super().__init__()

        # Store shape parameters for downstream inspection / debugging.
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model

        # Input projection: lifts raw features into a richer latent space.
        # Without this, attention would be too narrow at just n_features dims.
        self.input_projection = nn.Linear(n_features, d_model)

        # Time-axis attention: each time step attends to every other time step.
        # embed_dim = d_model because tokens here are time steps in latent space.
        # num_heads = 1 for interpretability in the baseline; can be raised later.
        self.time_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,
            batch_first=True,
        )

        # Channel-axis attention: each channel attends to every other channel.
        # embed_dim = seq_len because tokens are channels, each represented
        # by their values across all time steps.
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=seq_len,
            num_heads=1,
            batch_first=True,
        )

        # Learnable weights to fuse time-attention and channel-attention outputs.
        # Initialised equal (0.5, 0.5), then softmaxed at forward time so both
        # weights stay positive and sum to 1 during training.
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        # Output projection: maps fused latent back to original feature space.
        # Lets us plug this module in place of a standard attention block
        # without changing downstream dimensions.
        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Tensor of shape (batch, seq_len, n_features).
        """
        # Lift features into latent space:  (B, T, F) -> (B, T, d_model).
        h = self.input_projection(x)

        # Time-axis self-attention. Queries, keys, values all = h.
        # Attention weights implicitly have shape (B, T, T).
        time_out, _ = self.time_attention(h, h, h)

        # Transpose so CHANNELS become the token axis:
        # (B, T, d_model) -> (B, d_model, T).
        # Each of the d_model "channels" now attends across all time steps.
        h_t = h.transpose(1, 2)

        # Channel-axis self-attention. Output has same shape as h_t.
        channel_out, _ = self.channel_attention(h_t, h_t, h_t)

        # Transpose back so shape matches time_out:
        # (B, d_model, T) -> (B, T, d_model).
        channel_out = channel_out.transpose(1, 2)

        # Softmax fusion weights so they behave like a probability distribution.
        # Keeps the fused output on the same scale as its inputs.
        weights = torch.softmax(self.fusion_weights, dim=0)

        # Weighted combination of the two attention outputs.
        fused = weights[0] * time_out + weights[1] * channel_out

        # Project back to feature space: (B, T, d_model) -> (B, T, n_features).
        return self.output_projection(fused)
