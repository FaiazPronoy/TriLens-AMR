# src/models/deep_meta_v11.py
"""DeepMetaV11 — neural meta-learner with cross-antibiotic attention.

Input  : (B, n_meta_features=170) — concat of
            v8 OOF (5 base learners × 7 antibiotics = 35 dims),
            GeneAttentionNet logits (7 dims),
            GeneAttentionNet pooled embedding (128 dims).
Output : (B, 7) logits — one per antibiotic.

Design rationale (consensus from external consultations):
    - Small + heavily regularised (LayerNorm + Dropout + early stopping).
    - Cross-AB attention treats the 7 antibiotics as tokens, letting the
      meta-learner discover co-resistance patterns (e.g. CTX↔CAZ in
      β-lactams). Each token is decoded by its own linear head — shared
      trunk, separate heads.
    - Single transformer block (1 layer, 4 heads). Anything deeper
      overfits the small effective sample (≈ 5K train rows × 7 AB tokens).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepMetaV11(nn.Module):
    def __init__(
        self,
        n_meta_features: int = 170,
        n_antibiotics: int = 7,
        token_dim: int = 32,
        n_heads: int = 4,
        dropout: float = 0.3,
        attn_dropout: float = 0.2,
    ):
        super().__init__()
        self.n_antibiotics = n_antibiotics
        self.token_dim = token_dim

        # 140-(or 170-)dim → 64 dim trunk → n_ab × token_dim
        self.trunk = nn.Sequential(
            nn.Linear(n_meta_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_antibiotics * token_dim),
        )

        # Cross-antibiotic attention (treat each AB as a token)
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(token_dim)

        # Per-AB output heads (shared trunk, separate decoders)
        self.heads = nn.ModuleList(
            [nn.Linear(token_dim, 1) for _ in range(n_antibiotics)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_meta_features) → logits (B, n_antibiotics)."""
        B = x.size(0)
        # 1) project to per-AB token grid
        h = self.trunk(x).reshape(B, self.n_antibiotics, self.token_dim)
        # 2) cross-AB attention with residual + LN
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm(h + a)
        # 3) per-AB heads
        logits = torch.cat(
            [head(h[:, i]) for i, head in enumerate(self.heads)], dim=1
        )
        return logits        # (B, n_antibiotics)
