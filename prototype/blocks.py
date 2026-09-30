"""
Transformer Block for the Fusionformer architecture.

Wraps FusionAttention into a standard transformer block with:
    - Residual connections around attention and feed-forward layers
    - Layer normalisation after each residual add
    - A position-wise feed-forward network (MLP)
    - Dropout on both sub-layer outputs

This follows the classic post-norm transformer pattern from Vaswani et al. 2017,
but with FusionAttention replacing standard multi-head self-attention. The block
is the fundamental unit that will be stacked N times inside the encoder and
decoder halves of Fusionformer.

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
import torch
import torch.nn as nn

from src.models.attention import FusionAttention


class TransformerBlock(nn.Module):
    """
    A single transformer block using FusionAttention.

    Given input of shape (batch, seq_len, n_features), applies:
        1. FusionAttention (dual-axis) with residual + LayerNorm.
        2. Position-wise feed-forward network with residual + LayerNorm.

    Args:
        seq_len:    Length of the input time window.
        n_features: Number of channels in the input.
        d_model:    Internal latent dim used inside FusionAttention.
        ff_hidden:  Hidden dim of the feed-forward sub-layer. Common practice
                    is 2-4x n_features. Default 64 balances capacity and cost.
        dropout:    Dropout probability applied to both sub-layer outputs.
                    Helps regularisation on limited industrial datasets.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 32,
        ff_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Attention sub-layer: our dual-axis FusionAttention module.
        self.attention = FusionAttention(seq_len, n_features, d_model)

        # LayerNorm after the attention residual. Normalises across the feature
        # dim only — the standard choice for post-norm transformers.
        self.norm1 = nn.LayerNorm(n_features)

        # Feed-forward sub-layer: a small MLP applied position-wise (i.e., the
        # same MLP is applied independently to every time step). Widens to
        # ff_hidden and squeezes back — classic transformer FFN pattern.
        self.feed_forward = nn.Sequential(
            nn.Linear(n_features, ff_hidden),
            nn.ReLU(),                       # non-linearity so FFN adds capacity
            nn.Dropout(dropout),             # regularise inside the FFN
            nn.Linear(ff_hidden, n_features),
        )

        # LayerNorm after the feed-forward residual.
        self.norm2 = nn.LayerNorm(n_features)

        # Dropout applied to sub-layer outputs before the residual add.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Tensor of shape (batch, seq_len, n_features).
        """
        # Sub-layer 1: attention with residual + norm.
        #   x + dropout(attn(x))  →  LayerNorm
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2: feed-forward with residual + norm.
        #   x + dropout(ffn(x))  →  LayerNorm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x
