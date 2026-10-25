"""
src/models/gene_attention_net_v2.py
====================================
GeneAttentionNetV2 — pluggable attention pooling for AMR gene-level prediction.

New vs v1
---------
- pool_type: "softmax" (v1 baseline) | "gated" (Ilse 2018) | "entmax" | "sparsemax"
- n_layers=0 → lean attention-MIL (encoder + pool only, no transformer)
- multi_task=False → single-task 1-head variant (per-AB training)
- return_attention=True → exposes (B, n_genes) weights for interpretability

Placement: src/models/gene_attention_net_v2.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

__all__ = ["GeneAttentionNetV2"]


# ─── Pooling sub-modules ────────────────────────────────────────────────────


class _SoftmaxPool(nn.Module):
    """
    v1 baseline: single learned query vector + softmax over n_genes.
    Reproduces v1 pooling exactly for a fair E0 control.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # h: (B, G, D)
        scores = self.query(h).squeeze(-1)               # (B, G)
        a      = torch.softmax(scores, dim=-1)           # (B, G)
        z      = torch.bmm(a.unsqueeze(1), h).squeeze(1) # (B, D)
        return (z, a) if return_attention else z


class _GatedPool(nn.Module):
    """
    Gated attention-MIL pooling (Ilse et al., 2018).
    https://arxiv.org/abs/1802.04712

    e_k = w^T ( tanh(V h_k) ⊙ σ(U h_k) )    # content-aware gate
    a   = softmax(e)                           # (B, G)
    z   = Σ_k a_k h_k                          # (B, D)

    The sigmoid gate forces the model to select only genes where
    both the value projection (V) and gate projection (U) agree,
    naturally producing sparser, more focused attention.
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 64) -> None:
        super().__init__()
        self.V = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.U = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.w = nn.Linear(attn_dim,   1,        bias=False)

    def forward(
        self,
        h: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        gating = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))  # (B, G, attn_dim)
        e      = self.w(gating).squeeze(-1)                          # (B, G)
        a      = torch.softmax(e, dim=-1)                            # (B, G)
        z      = torch.bmm(a.unsqueeze(1), h).squeeze(1)             # (B, D)
        return (z, a) if return_attention else z


class _SparsePool(nn.Module):
    """
    Sparse attention pool via entmax15 or sparsemax.
    Produces attention vectors with many EXACT zeros — genuine gene selection.
    Unlike softmax, nearly-irrelevant genes get weight 0, not 0.0002.

    Requires: pip install entmax
    """

    def __init__(self, hidden_dim: int, mode: str = "entmax15") -> None:
        super().__init__()
        if mode not in ("entmax15", "sparsemax"):
            raise ValueError(f"mode must be 'entmax15' or 'sparsemax', got {mode!r}")
        self.mode  = mode
        self.query = nn.Linear(hidden_dim, 1, bias=False)
        # Validate import at construction to fail fast with a clear message
        try:
            import entmax as _  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "entmax package required for sparse pooling.  "
                "Run: pip install entmax"
            ) from exc

    def forward(
        self,
        h: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        scores   = self.query(h).squeeze(-1)             # (B, G)
        # Cast to float32: entmax/sparsemax are numerically unstable in fp16
        scores_f = scores.float()
        if self.mode == "entmax15":
            from entmax import entmax15
            a = entmax15(scores_f, dim=-1).to(h.dtype)
        else:
            from entmax import sparsemax
            a = sparsemax(scores_f, dim=-1).to(h.dtype)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return (z, a) if return_attention else z


# ─── Per-gene shared encoder ───────────────────────────────────────────────


class _PerGeneEncoder(nn.Module):
    """Two-layer shared MLP applied identically to every gene token."""

    def __init__(self, n_stats: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_stats, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_genes, n_stats) → (B, n_genes, hidden_dim)
        return self.net(x)


# ─── Main model ─────────────────────────────────────────────────────────────


class GeneAttentionNetV2(nn.Module):
    """
    GeneAttentionNetV2 — per-gene encoder + optional transformer blocks
    + pluggable attention pool + single- or multi-task classification head.

    Parameters
    ----------
    n_genes       : number of gene tokens  (default 5000; use 1000 for E6)
    n_stats       : per-gene summary stats  (default 4: mean/var/n_nonref/max)
    hidden_dim    : token embedding dim     (default 128)
    n_heads       : transformer MHA heads   (default 4)
    n_layers      : transformer depth; 0 = lean attention-MIL (no transformer)
    ffn_mult      : FFN expansion ratio     (default 2)
    dropout       : dropout probability     (default 0.3)
    pool_type     : 'softmax' | 'gated' | 'entmax' | 'sparsemax'
    attn_dim      : gated-pool projection dim (default 64)
    multi_task    : True → 7-head multi-task; False → 1-head single-task
    n_antibiotics : AB output heads when multi_task=True (default 7)
    """

    _VALID_POOLS = frozenset({"softmax", "gated", "entmax", "sparsemax"})

    def __init__(
        self,
        n_genes:        int   = 5000,
        n_stats:        int   = 4,
        hidden_dim:     int   = 128,
        n_heads:        int   = 4,
        n_layers:       int   = 2,
        ffn_mult:       int   = 2,
        dropout:        float = 0.3,
        pool_type:      str   = "gated",
        attn_dim:       int   = 64,
        multi_task:     bool  = True,
        n_antibiotics:  int   = 7,
    ) -> None:
        super().__init__()

        if pool_type not in self._VALID_POOLS:
            raise ValueError(
                f"pool_type must be one of {sorted(self._VALID_POOLS)}, "
                f"got {pool_type!r}"
            )

        self.n_genes    = n_genes
        self.n_layers   = n_layers
        self.multi_task = multi_task
        self.pool_type  = pool_type

        # ── per-gene shared encoder ─────────────────────────────────────────
        self.encoder = _PerGeneEncoder(n_stats, hidden_dim, dropout)

        # ── optional transformer blocks (skip entirely when n_layers == 0) ──
        if n_layers > 0:
            self.pos_embed = nn.Embedding(n_genes, hidden_dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * ffn_mult,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,   # pre-LN: more stable for deeper transformers
            )
            # enable_nested_tensor=False suppresses a harmless UserWarning
            # with batch_first=True; fall back for older PyTorch versions
            try:
                self.transformer = nn.TransformerEncoder(
                    enc_layer,
                    num_layers=n_layers,
                    enable_nested_tensor=False,
                )
            except TypeError:
                self.transformer = nn.TransformerEncoder(
                    enc_layer, num_layers=n_layers
                )
        else:
            self.pos_embed   = None
            self.transformer = None

        # ── attention pool (the ablation axis) ──────────────────────────────
        if pool_type == "softmax":
            self.pool: nn.Module = _SoftmaxPool(hidden_dim)
        elif pool_type == "gated":
            self.pool = _GatedPool(hidden_dim, attn_dim)
        elif pool_type in ("entmax", "sparsemax"):
            mode = "sparsemax" if pool_type == "sparsemax" else "entmax15"
            self.pool = _SparsePool(hidden_dim, mode)

        # ── output head(s) ──────────────────────────────────────────────────
        if multi_task:
            self.heads: nn.Module = nn.ModuleList(
                [nn.Linear(hidden_dim, 1) for _ in range(n_antibiotics)]
            )
            self.head = None
        else:
            self.heads = None
            self.head  = nn.Linear(hidden_dim, 1)

    # ────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x                : (B, n_genes, n_stats)  float tensor on the model device
        return_attention : if True, also return (B, n_genes) attention weights

        Returns
        -------
        logits   : (B, n_antibiotics) if multi_task  else  (B, 1)
        attn     : (B, n_genes)  [only when return_attention=True]
        """
        # 1. Per-gene encoding: (B, G, n_stats) → (B, G, D)
        h = self.encoder(x)

        # 2. Optional transformer (position-aware gene context)
        if self.n_layers > 0:
            pos = torch.arange(self.n_genes, device=x.device)
            h   = h + self.pos_embed(pos)    # positional embedding broadcast over batch
            h   = self.transformer(h)

        # 3. Attention pooling: (B, G, D) → (B, D)
        pool_out = self.pool(h, return_attention=return_attention)
        if return_attention:
            z, attn = pool_out               # (B, D),  (B, G)
        else:
            z = pool_out                     # (B, D)

        # 4. Classification head(s): (B, D) → logits
        if self.multi_task:
            # Concatenate all AB logits into one tensor
            logits = torch.cat(
                [head(z) for head in self.heads], dim=-1
            )                                # (B, n_antibiotics)
        else:
            logits = self.head(z)            # (B, 1)

        return (logits, attn) if return_attention else logits

    # ────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
