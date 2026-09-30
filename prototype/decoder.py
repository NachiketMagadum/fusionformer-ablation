"""
FusionformerDecoder for the Fusionformer autoencoder architecture.

Mirror structure to FusionformerEncoder — stacks N TransformerBlocks to
reconstruct the input from the encoded representation.

Design choices:
    - No positional encoding needed here. The encoder's output already carries
      the temporal information injected upstream.
    - Same TransformerBlock as the encoder, keeping the two halves symmetric
      and parameter-counts easy to reason about.
    - No explicit bottleneck. Reconstruction pressure through the MSE loss
      forces the model to learn meaningful representations, which is the
      standard "flat autoencoder" pattern used for anomaly detection.

Why a flat autoencoder rather than a dimensionality bottleneck?
    Industrial sensor windows are short (typically 30-60 time steps × ~10
    channels). A hard bottleneck at these small dimensions risks throwing
    away useful signal. The reconstruction objective alone provides enough
    pressure to learn a normal-behaviour representation; anomalies then
    fail to reconstruct cleanly and are flagged by their reconstruction
    error.

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
import torch
import torch.nn as nn

from src.models.blocks import TransformerBlock


class FusionformerDecoder(nn.Module):
    """
    Stack of N TransformerBlocks that reconstructs the input from encoded features.

    Args:
        seq_len:    Length of input time window.
        n_features: Number of sensor channels.
        d_model:    Latent dim inside each block's attention.
        ff_hidden:  Hidden dim of each block's feed-forward layer.
        n_layers:   How many TransformerBlocks to stack. Keep same as encoder
                    by default so the model is symmetric.
        dropout:    Dropout rate inside the blocks.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 32,
        ff_hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers

        # Stack of N TransformerBlocks that transform the encoded input back
        # into feature space. nn.ModuleList registers each block so the
        # optimiser tracks all their parameters.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                seq_len=seq_len,
                n_features=n_features,
                d_model=d_model,
                ff_hidden=ff_hidden,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded representation of shape (batch, seq_len, n_features)
               produced by FusionformerEncoder.

        Returns:
            Reconstructed input of shape (batch, seq_len, n_features).
        """
        # Pass through each transformer block sequentially.
        # Each block refines the representation toward the reconstruction target.
        for block in self.blocks:
            x = block(x)

        return x
