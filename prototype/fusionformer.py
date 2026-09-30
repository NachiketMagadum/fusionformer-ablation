"""
Fusionformer — full autoencoder model for multivariate time-series anomaly detection.

Combines FusionformerEncoder + FusionformerDecoder into a reconstruction-based
autoencoder. Trained on normal (non-anomalous) sensor windows, the model learns
to reconstruct the pattern of normal behaviour. At inference time, windows that
differ from normal produce large reconstruction errors and are flagged as
anomalies.

Reconstruction-based anomaly detection principle:
    reconstruction_error(x) = mean((x - decoder(encoder(x)))**2)
    high error → not-normal → anomaly candidate

The model is described end-to-end so the training loop and evaluation code
can simply call `model(x)` for reconstruction and `model.reconstruction_error(x)`
for per-window anomaly scores.

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
import torch
import torch.nn as nn

from src.models.encoder import FusionformerEncoder
from src.models.decoder import FusionformerDecoder


class Fusionformer(nn.Module):
    """
    Full Fusionformer autoencoder for multivariate time-series anomaly detection.

    Architecture:
        input (B, T, F)
            → FusionformerEncoder (positional encoding + N transformer blocks)
                → encoded representation (B, T, F)
                    → FusionformerDecoder (N transformer blocks)
                        → reconstruction (B, T, F)

    Args:
        seq_len:    Length of input time window (number of time steps).
        n_features: Number of sensor channels.
        d_model:    Latent dim used inside attention modules.
        ff_hidden:  Hidden dim of the feed-forward layer inside each block.
        n_layers:   Number of TransformerBlocks in encoder AND decoder.
        dropout:    Dropout rate throughout the model.
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

        # Store config for logging / reproducibility.
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        self.n_layers = n_layers

        # Encoder: input → positional encoding → N transformer blocks.
        self.encoder = FusionformerEncoder(
            seq_len=seq_len,
            n_features=n_features,
            d_model=d_model,
            ff_hidden=ff_hidden,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Decoder: encoded representation → N transformer blocks → reconstruction.
        self.decoder = FusionformerDecoder(
            seq_len=seq_len,
            n_features=n_features,
            d_model=d_model,
            ff_hidden=ff_hidden,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full autoencoder forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Reconstruction of shape (batch, seq_len, n_features).
        """
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-window reconstruction error for anomaly scoring.

        Higher error = window looks less like what the model learned as normal.
        Sort windows by this score and threshold to produce anomaly flags.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Tensor of shape (batch,) with the mean squared reconstruction
            error for each input window.
        """
        # Put model in eval mode so dropout is disabled during scoring.
        self.eval()

        # Reconstruct.
        reconstructed = self(x)

        # Mean squared error per window, averaged across time and features.
        # dim=(1, 2) reduces the (T, F) axes, leaving a (batch,) tensor.
        error = ((x - reconstructed) ** 2).mean(dim=(1, 2))

        return error
