"""
FusionformerEncoder for the Fusionformer autoencoder architecture.

Stacks N TransformerBlocks with a learned positional encoding at the input.
The encoder processes an input window through repeated attention + feed-forward
blocks, producing a representation that the decoder will reconstruct back into
the original space.

Why positional encoding?
    Attention is permutation-invariant by default — it doesn't know that
    time step 5 comes after time step 4. For time-series data where temporal
    order is critical (rising trends, drift, cycles), we inject position
    information so the model can learn temporal patterns.

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
import torch
import torch.nn as nn

from src.models.blocks import TransformerBlock


class FusionformerEncoder(nn.Module):
    """
    Stack of N TransformerBlocks with learned positional encoding.

    Args:
        seq_len:    Length of input time window (number of time steps).
        n_features: Number of sensor channels.
        d_model:    Latent dim used inside each block's attention.
        ff_hidden:  Hidden dim of each block's feed-forward layer.
        n_layers:   How many TransformerBlocks to stack.
        dropout:    Dropout rate used inside blocks and on positional encoding.
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

        # Learned positional encoding — one vector per time step per feature.
        # Small initialisation (std 0.02) so it doesn't dominate the input signal
        # early in training. Same trick used by BERT and other modern transformers.
        # Shape: (1, seq_len, n_features) so it broadcasts across the batch dim.
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_len, n_features) * 0.02
        )

        # Dropout on the input embedding + positional encoding sum.
        # Helps regularise on limited industrial datasets like SKAB.
        self.dropout = nn.Dropout(dropout)

        # Stack of N TransformerBlocks. nn.ModuleList registers each block
        # as a submodule so all parameters are tracked by the optimiser.
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
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Tensor of shape (batch, seq_len, n_features).
        """
        # Add positional information to the input.
        # Broadcasting: pos_encoding is (1, T, F), x is (B, T, F).
        x = x + self.pos_encoding

        # Dropout on the input embedding for regularisation.
        x = self.dropout(x)

        # Pass through each transformer block sequentially.
        for block in self.blocks:
            x = block(x)

        return x
