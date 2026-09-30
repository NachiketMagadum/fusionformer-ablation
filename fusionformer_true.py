"""
Fusionformer model definition, following the architecture in Wang et al. 2025
(IEEE TNNLS 36(8)), Section III.

Architecture:
  - SWSE: segment-wise sequence embedding with learnable projection and positional encoding
  - Encoder: N0 modules of (MSWAA -> FFN -> MSWEA -> FFN), each with residual + LayerNorm
  - Decoder: N1 modules of same structure
  - Discriminator: 3 FC layers + Sigmoid, operating on concat(X_hist, Y_pred_or_true)

Training objective (not in this file): the paper trains the forecaster on the
adversarial loss alone (eq 14, Algorithm 1). My training script instead uses a
Huber prediction loss, plus lambda_adv times the adversarial loss when
lambda_adv > 0 (lambda_adv = 0 for the "full" model, 0.01 for adv_on). This is
listed as a deviation in Section 3.6 of the dissertation.

Paper hyperparameters (Section IV-D):
  - d_model = 256, num_heads = 6, batch_size = 32
  - learning_rate from {5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5} by grid search
  - L_seg, N0, N1: swept in the sensitivity analysis

The use_swse and use_mswea flags build the SWSE-off and MSWEA-off ablation
variants. Training and scoring are in train_ff_forecast_and_score_anomaly.py.

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""

import torch
import torch.nn as nn


# ============================================================
# SWSE: Segment-Wise Sequence Embedding (Paper Section III-C)
# ============================================================

class SWSE(nn.Module):
    """
    Segment-Wise Sequence Embedding per paper eqs (2-5).

    Partitions X in R^{T x D} into non-overlapping segments of length L_seg
    per variable, projects each segment to a d_model vector via a learnable
    linear transformation, and adds a learnable positional encoding.

    Input:  x in R^{B x T x D}
    Output: U in R^{B x D x L_sn x d_model}  where L_sn = T / L_seg
    """

    def __init__(self, seq_len: int, n_vars: int, segment_len: int, d_model: int):
        super().__init__()
        assert seq_len % segment_len == 0, (
            f"seq_len ({seq_len}) must be divisible by segment_len ({segment_len})"
        )
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.segment_len = segment_len
        self.n_segments = seq_len // segment_len
        self.d_model = d_model

        # Learnable projection W in R^{d_model x L_seg}, shared across (i, d).
        # Paper eq 4: u_{i,d} = W x_{i,d} + W^{pos}_{i,d}
        self.projection = nn.Linear(segment_len, d_model, bias=False)

        # Learnable positional encoding W^{pos} in R^{D x L_sn x d_model}.
        # One vector per (variable, segment_index) position.
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_vars, self.n_segments, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        assert T == self.seq_len and D == self.n_vars, (
            f"expected (B, {self.seq_len}, {self.n_vars}), got {x.shape}"
        )
        # (B, T, D) -> (B, D, T) -> (B, D, L_sn, L_seg)
        x = x.transpose(1, 2).reshape(B, D, self.n_segments, self.segment_len)
        # Project each segment: (B, D, L_sn, L_seg) -> (B, D, L_sn, d_model)
        u = self.projection(x)
        # Add positional encoding
        u = u + self.pos_encoding
        return u  # (B, D, L_sn, d_model)


# ============================================================
# MSWAA: Multi-head Segment-Wise Intravariable Attention (Paper eq 6-7)
# ============================================================

class MSWAA(nn.Module):
    """
    Multi-head Segment-Wise Intravariable Attention.

    For each variable d independently, attention over the L_sn segment tokens
    in that variable. Implements paper equations (6-7) and Fig 3.

    Input/Output: U in R^{B x D x L_sn x d_model}
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, D, L_sn, d_model)
        B, D, L_sn, d_model = u.shape
        # Treat each variable independently by folding D into batch dim:
        # (B, D, L_sn, d) -> (B*D, L_sn, d)
        u_flat = u.reshape(B * D, L_sn, d_model)
        attn_out, _ = self.attn(u_flat, u_flat, u_flat, need_weights=False)
        # Residual + LayerNorm (paper eq 6)
        out = self.norm(u_flat + attn_out)
        # Restore shape
        return out.reshape(B, D, L_sn, d_model)


# ============================================================
# MSWEA: Multi-head Segment-Wise Intervariable Attention (Paper eq 8-11)
# ============================================================

class MSWEA(nn.Module):
    """
    Multi-head Segment-Wise Intervariable Attention (per paper eqs 8-11).

    Attention across the D variable tokens at each segment position. Following
    common practice in patch-based transformers (iTransformer, Liu et al. 2024;
    PatchTST, Nie et al. 2023) and the natural reading of paper eqs 8-11, we
    implement MSWEA as multi-head attention with embed_dim = d_model applied
    per-segment across the D variables. The paper's phrase "all segments
    within U^{tim} are merged in SWEA" is realised by batching all L_sn
    segment positions into a single attention call, not by concatenating
    them into a single feature vector (which would produce an intractably
    large attention matrix).

    Input/Output: U in R^{B x D x L_sn x d_model}
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, D, L_sn, d_model)
        B, D, L_sn, d_model = u.shape
        # Move L_sn into batch: (B, D, L_sn, d) -> (B, L_sn, D, d) -> (B*L_sn, D, d)
        u_perm = u.permute(0, 2, 1, 3).contiguous().reshape(B * L_sn, D, d_model)
        attn_out, _ = self.attn(u_perm, u_perm, u_perm, need_weights=False)
        out = self.norm(u_perm + attn_out)  # (B*L_sn, D, d_model)
        # Restore: (B*L_sn, D, d) -> (B, L_sn, D, d) -> (B, D, L_sn, d)
        return out.reshape(B, L_sn, D, d_model).permute(0, 2, 1, 3).contiguous()


# ============================================================
# FFN + Residual + LayerNorm block
# ============================================================

class PositionWiseFFN(nn.Module):
    """Position-wise feed-forward with residual + LayerNorm, operating on
    the last dim (d_model). Applied after both MSWAA and MSWEA per paper
    equations (7) and (11)."""

    def __init__(self, d_model: int, ff_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.norm(u + self.ff(u))


# ============================================================
# Encoder / Decoder blocks
# ============================================================

class FusionformerBlock(nn.Module):
    """
    One Fusionformer encoder/decoder module: MSWAA -> FFN -> MSWEA -> FFN.

    Paper Section III-B: "the encoder and decoder ... consist of N identical
    modules, which include two key sub-modules: the multihead FAM layer and
    the fully connected feed-forward network." Where FAM = MSWAA + MSWEA.

    Ablation flag `use_mswea`: if False, drops the intervariable attention branch,
    keeping only MSWAA + FFN. This is the honest MSWEA ablation for the study.
    """

    def __init__(self, n_segments: int, d_model: int, n_heads: int,
                 ff_hidden: int, dropout: float = 0.1, use_mswea: bool = True):
        super().__init__()
        self.mswaa = MSWAA(d_model, n_heads, dropout)
        self.ffn1 = PositionWiseFFN(d_model, ff_hidden, dropout)
        self.use_mswea = use_mswea
        if use_mswea:
            self.mswea = MSWEA(d_model, n_heads, dropout)
            self.ffn2 = PositionWiseFFN(d_model, ff_hidden, dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u = self.mswaa(u)   # intravariable attention with residual+LN
        u = self.ffn1(u)    # FFN with residual+LN
        if self.use_mswea:
            u = self.mswea(u)   # intervariable attention with residual+LN
            u = self.ffn2(u)    # FFN with residual+LN
        return u


# ============================================================
# Full Fusionformer forecasting model
# ============================================================

class Fusionformer(nn.Module):
    """
    Full Fusionformer forecasting model per Wang et al. 2025.

    Input:  X_{1:T} in R^{B x T x D}
    Output: Y_hat_{T+1:T+tau} in R^{B x tau x D}

    Args:
        seq_len:     historical window length T
        pred_len:    forecast horizon tau
        n_vars:      number of variables D
        segment_len: L_seg (must divide seq_len)
        d_model:     hidden dim (paper: 256)
        n_heads:     attention heads (paper: 6)
        ff_hidden:   feed-forward hidden dim (paper does not specify; default 4*d_model)
        n_enc:       encoder blocks N_0
        n_dec:       decoder blocks N_1
        dropout:     dropout rate
    """

    def __init__(self, seq_len: int, pred_len: int, n_vars: int,
                 segment_len: int = 4, d_model: int = 252, n_heads: int = 6,
                 ff_hidden: int = None, n_enc: int = 2, n_dec: int = 1,
                 dropout: float = 0.1, use_swse: bool = True,
                 use_mswea: bool = True):
        # NOTE: paper reports d_model=256 with n_heads=6, but 256 is not
        # divisible by 6. We use d_model=252 (nearest multiple of 6 below 256)
        # to keep n_heads=6 as the paper specifies. This 1.6% dimension
        # reduction is documented as a deviation in the methodology chapter.
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}). "
                f"Try d_model={(d_model // n_heads) * n_heads}."
            )
        if ff_hidden is None:
            ff_hidden = 4 * d_model

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.segment_len = segment_len
        self.n_segments = seq_len // segment_len
        self.d_model = d_model

        self.use_swse = use_swse
        self.use_mswea = use_mswea

        if use_swse:
            self.swse = SWSE(seq_len, n_vars, segment_len, d_model)
        else:
            # Ablation: no segmentation. Project each timestep to d_model directly
            # and treat each of T timesteps as a token (so n_segments = T, seg_len = 1).
            self.n_segments = seq_len
            self.per_step_proj = nn.Linear(1, d_model)
            self.pos_encoding_alt = nn.Parameter(
                torch.randn(1, n_vars, seq_len, d_model) * 0.02
            )

        self.encoder = nn.ModuleList([
            FusionformerBlock(self.n_segments, d_model, n_heads, ff_hidden, dropout, use_mswea)
            for _ in range(n_enc)
        ])
        self.decoder = nn.ModuleList([
            FusionformerBlock(self.n_segments, d_model, n_heads, ff_hidden, dropout, use_mswea)
            for _ in range(n_dec)
        ])

        # Output head: (B, D, L_sn, d_model) -> (B, pred_len, D)
        # Flatten (L_sn * d_model) per variable and project to pred_len values.
        self.output_head = nn.Linear(self.n_segments * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if self.use_swse:
            u = self.swse(x)                    # (B, D, L_sn, d_model)
        else:
            # x -> (B, D, T, 1) -> project to (B, D, T, d_model) + pos encoding
            B, T, D = x.shape
            u = x.transpose(1, 2).unsqueeze(-1)  # (B, D, T, 1)
            u = self.per_step_proj(u) + self.pos_encoding_alt
        for block in self.encoder:
            u = block(u)
        for block in self.decoder:
            u = block(u)
        # Flatten per-variable, project to pred_len
        B, D, L_sn, d = u.shape
        u_flat = u.reshape(B, D, L_sn * d)
        y_hat = self.output_head(u_flat)    # (B, D, pred_len)
        return y_hat.transpose(1, 2)        # (B, pred_len, D)


# ============================================================
# Discriminator (paper Section III-E)
# ============================================================

class FusionformerDiscriminator(nn.Module):
    """
    3-layer fully connected discriminator with Sigmoid activation.

    Input: Y_seq in R^{B x L_w x D} where L_w = T + tau
    Output: scalar in [0, 1] (probability that the input is real).
    """

    def __init__(self, seq_len: int, pred_len: int, n_vars: int,
                 hidden: int = 128):
        super().__init__()
        input_dim = (seq_len + pred_len) * n_vars
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, y_seq: torch.Tensor) -> torch.Tensor:
        # y_seq: (B, L_w, D)
        B = y_seq.size(0)
        return self.mlp(y_seq.reshape(B, -1)).squeeze(-1)


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    B, T, tau, D = 4, 96, 24, 7
    x = torch.randn(B, T, D)
    y_true = torch.randn(B, tau, D)

    model = Fusionformer(seq_len=T, pred_len=tau, n_vars=D,
                         segment_len=4, d_model=252, n_heads=6, n_enc=2, n_dec=1)
    print(f"Fusionformer params: {sum(p.numel() for p in model.parameters()):,}")
    y_hat = model(x)
    print(f"y_hat shape: {y_hat.shape}  (expected [{B}, {tau}, {D}])")

    disc = FusionformerDiscriminator(seq_len=T, pred_len=tau, n_vars=D)
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")
    y_fake = torch.cat([x, y_hat.detach()], dim=1)  # (B, T+tau, D)
    d_score = disc(y_fake)
    print(f"Discriminator output: {d_score.shape} (expected [{B}])")
    print(f"MSE(y_hat, y_true) = {((y_hat - y_true) ** 2).mean().item():.4f}")
    print("OK")
