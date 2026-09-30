"""
Fusionformer with SWSE (Segment-wise Sequence Embedding).

Extends the FAM-only Fusionformer with the second core component from
Wang et al. 2025: per-channel segment embedding, following PatchTST-style
patching to preserve channel structure for meaningful channel attention.

Shape flow:
    Input:            (B, T, F)         # batch, time, channels
    After SWSE:       (B, F, N_seg, D)  # per-channel patches embedded to D
    After N blocks:   (B, F, N_seg, D)  # same shape, refined
    After recon head: (B, T, F)         # reconstruction

Where N_seg = T / segment_len.

Design rationale:
    - Per-channel patching (not flattened patching) preserves the F channel
      axis so channel attention has semantic meaning.
    - Each patch summarises segment_len timesteps of one channel into a
      D-dim embedding. Reduces attention sequence length from T to N_seg,
      typically ~5-10x reduction.
    - Fusion attention then operates on (B, F, N_seg, D):
        * Time attention: over N_seg segments per channel
        * Channel attention: over F channels per segment

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
import torch
import torch.nn as nn


# ============================================================
#  SWSE input embedding
# ============================================================

class SWSE(nn.Module):
    """
    Segment-wise Sequence Embedding (per-channel patching).

    Splits each channel's time-series into non-overlapping segments and
    embeds each segment into a D-dim vector via a linear projection.

    Args:
        seq_len:     Length of input time window.
        segment_len: Length of each segment (must divide seq_len).
        embed_dim:   Output dimension of each segment embedding.
    """

    def __init__(self, seq_len: int, segment_len: int, embed_dim: int):
        super().__init__()
        assert seq_len % segment_len == 0, (
            f"seq_len ({seq_len}) must be divisible by segment_len ({segment_len})"
        )
        self.seq_len = seq_len
        self.segment_len = segment_len
        self.n_segments = seq_len // segment_len
        self.embed_dim = embed_dim
        # One shared linear projection applied to every (channel, segment).
        self.projection = nn.Linear(segment_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            (B, F, N_seg, D)
        """
        B, T, F = x.shape
        # (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        # (B, F, T) -> (B, F, N_seg, segment_len)
        x = x.reshape(B, F, self.n_segments, self.segment_len)
        # (B, F, N_seg, segment_len) -> (B, F, N_seg, D)
        return self.projection(x)


# ============================================================
#  Fusion attention adapted for SWSE tokens
# ============================================================

class FusionAttentionSWSE(nn.Module):
    """
    FAM operating on SWSE segment tokens.

    Input shape: (B, F, N_seg, D)
    - Time attention: attend over N_seg segments, per channel
    - Channel attention: attend over F channels, per segment
    - Fusion via learnable softmax-normalised weights
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, batch_first=True
        )
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, batch_first=True
        )
        # Fusion weights, softmaxed at forward time.
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F, N_seg, D)
        Returns:
            (B, F, N_seg, D)
        """
        B, F, N_seg, D = x.shape

        # Time attention: for each channel, attend across segments.
        # Reshape (B, F, N_seg, D) -> (B*F, N_seg, D) so attention treats
        # each channel as an independent sequence over segments.
        x_time = x.reshape(B * F, N_seg, D)
        time_out, _ = self.time_attention(x_time, x_time, x_time)
        time_out = time_out.reshape(B, F, N_seg, D)

        # Channel attention: for each segment, attend across channels.
        # Permute (B, F, N_seg, D) -> (B, N_seg, F, D), then reshape
        # (B, N_seg, F, D) -> (B*N_seg, F, D) so each segment sees a
        # sequence of F channels.
        x_chan = x.permute(0, 2, 1, 3).reshape(B * N_seg, F, D)
        chan_out, _ = self.channel_attention(x_chan, x_chan, x_chan)
        chan_out = chan_out.reshape(B, N_seg, F, D).permute(0, 2, 1, 3)

        # Fuse with softmax-normalised learnable weights.
        weights = torch.softmax(self.fusion_weights, dim=0)
        return weights[0] * time_out + weights[1] * chan_out


# ============================================================
#  Transformer block on SWSE tokens
# ============================================================

class TransformerBlockSWSE(nn.Module):
    """
    Transformer block using FusionAttentionSWSE + position-wise FFN.

    Applies attention with residual + LayerNorm, then FFN with residual +
    LayerNorm. Operates on the (B, F, N_seg, D) tensor shape.
    """

    def __init__(
        self,
        embed_dim: int,
        ff_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = FusionAttentionSWSE(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer.
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward sub-layer.
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# ============================================================
#  Reconstruction head
# ============================================================

class ReconstructionHead(nn.Module):
    """
    Projects SWSE tokens back to the original time-series shape.

    (B, F, N_seg, D) -> (B, F, N_seg, segment_len)
                     -> (B, F, T)
                     -> (B, T, F)
    """

    def __init__(self, embed_dim: int, segment_len: int):
        super().__init__()
        self.segment_len = segment_len
        self.projection = nn.Linear(embed_dim, segment_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, N_seg, D = x.shape
        # Project each token back to segment_len values.
        x = self.projection(x)  # (B, F, N_seg, segment_len)
        # Reshape segments back to time dim: (B, F, T)
        x = x.reshape(B, F, N_seg * self.segment_len)
        # Transpose to (B, T, F).
        return x.transpose(1, 2)


# ============================================================
#  Full FusionformerSWSE
# ============================================================

class FusionformerSWSE(nn.Module):
    """
    Fusionformer with SWSE input embedding.

    Adds the second core component from Wang et al. 2025 to our earlier
    FAM-only implementation. Retains the reconstruction-autoencoder
    pattern used for anomaly detection.

    Args:
        seq_len:     Input time window length.
        n_features:  Number of sensor channels.
        segment_len: Length of each SWSE segment (must divide seq_len).
        embed_dim:   Latent dimension per segment token.
        ff_hidden:   Hidden dim of position-wise FFN.
        n_layers:    Number of encoder blocks (decoder mirrors).
        dropout:     Dropout rate throughout.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        segment_len: int = 5,
        embed_dim: int = 32,
        ff_hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert seq_len % segment_len == 0, (
            f"seq_len ({seq_len}) must be divisible by segment_len ({segment_len})"
        )
        self.seq_len = seq_len
        self.n_features = n_features
        self.segment_len = segment_len
        self.n_segments = seq_len // segment_len
        self.embed_dim = embed_dim

        # SWSE input embedding.
        self.swse = SWSE(seq_len, segment_len, embed_dim)

        # Learned positional encoding — one vector per (channel, segment).
        # Shape (1, F, N_seg, D) broadcasts over the batch dim.
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_features, self.n_segments, embed_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

        # Encoder blocks.
        self.encoder_blocks = nn.ModuleList([
            TransformerBlockSWSE(embed_dim, ff_hidden, dropout)
            for _ in range(n_layers)
        ])

        # Decoder blocks (symmetric to encoder).
        self.decoder_blocks = nn.ModuleList([
            TransformerBlockSWSE(embed_dim, ff_hidden, dropout)
            for _ in range(n_layers)
        ])

        # Reconstruction back to (B, T, F).
        self.reconstruction_head = ReconstructionHead(embed_dim, segment_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            (B, T, F) reconstruction
        """
        # Embed segments per channel.
        h = self.swse(x)  # (B, F, N_seg, D)
        # Add positional encoding + dropout.
        h = h + self.pos_encoding
        h = self.dropout(h)
        # Encoder.
        for block in self.encoder_blocks:
            h = block(h)
        # Decoder.
        for block in self.decoder_blocks:
            h = block(h)
        # Reconstruct.
        return self.reconstruction_head(h)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-window mean squared reconstruction error for anomaly scoring.

        Args:
            x: (B, T, F)
        Returns:
            (B,) tensor of per-window MSE.
        """
        self.eval()
        reconstruction = self(x)
        return ((x - reconstruction) ** 2).mean(dim=(1, 2))
