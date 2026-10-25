# src/models/cnn_variants.py
"""1D CNN variants for Pillar 1 experiments.

Three architectures, all inheriting directly from L.LightningModule
(NOT from AMRBaseLightningModule) so they control their own training loop
and accept plain (x, y) batches from the single-antibiotic loaders.

* Naive1DCNN              — locality-assuming global CNN.
* ContigAwareCNN_VariantA — per-contig vectorised CNN + attention pool.
* ContigAwareCNN_VariantB — single CNN with boundary-masked activations.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from sklearn.metrics import average_precision_score


# ---------------------------------------------------------------------------
# Shared mixin — training/val loop, optimizer, metrics logging
# ---------------------------------------------------------------------------
class _AMRSingleTaskMixin(L.LightningModule):
    """Mixin that provides training/val/test steps for single-antibiotic CNN.

    Subclass must:
        • define self.pos_weight (float)
        • define self.lr, self.weight_decay, self.t_max (floats)
        • implement forward(x) -> (B,) logits
    """

    # ── forward: implemented by subclass ────────────────────────────────────

    # ── loss helper ─────────────────────────────────────────────────────────
    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor([self.pos_weight], device=self.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pw)

    # ── training ─────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ── validation ───────────────────────────────────────────────────────────
    def on_validation_epoch_start(self):
        self._val_probs: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=False)
        self._val_probs.append(torch.sigmoid(logits).detach().cpu())
        self._val_labels.append(y.detach().cpu().long())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_probs:
            self.log("val/pr_auc", 0.0, prog_bar=True)
            return
        probs  = torch.cat(self._val_probs).numpy().astype(np.float32)
        labels = torch.cat(self._val_labels).numpy().astype(np.int32)
        if len(np.unique(labels)) < 2:
            pr_auc = 0.0
        else:
            pr_auc = float(average_precision_score(labels, probs))
        self.log("val/pr_auc", pr_auc, prog_bar=True)

    # ── test ─────────────────────────────────────────────────────────────────
    def on_test_epoch_start(self):
        self._test_probs: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self._test_probs.append(torch.sigmoid(logits).detach().cpu())
        self._test_labels.append(y.detach().cpu().long())

    def on_test_epoch_end(self):
        if not self._test_probs:
            return
        probs  = torch.cat(self._test_probs).numpy().astype(np.float32)
        labels = torch.cat(self._test_labels).numpy().astype(np.int32)
        if len(np.unique(labels)) >= 2:
            pr_auc = float(average_precision_score(labels, probs))
        else:
            pr_auc = 0.0
        self.log("test/pr_auc", pr_auc)

    # ── optimizer + scheduler ────────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.t_max, eta_min=1e-6
        )
        return [opt], [{"scheduler": sched, "interval": "epoch"}]


# ===========================================================================
# Naive 1D CNN
# ===========================================================================
class Naive1DCNN(_AMRSingleTaskMixin):
    """Locality-assuming 1D CNN over the full SNP sequence.

    Expected to succeed on Giessen (single-reference, true positional
    structure) but fail on Arcadia (pan-genome, broken locality).
    """

    def __init__(
        self,
        n_features: int,
        vocab_size: int = 5,
        padding_idx: int = 4,
        embed_dim: int = 32,
        dropout: float = 0.4,
        pos_weight: float = 1.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_t_max: int = 30,
        # kept for API compat with experiment code
        antibiotic_names: Optional[list[str]] = None,
        target_antibiotic: Optional[str] = None,
        pos_weights: Optional[list[float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["antibiotic_names", "target_antibiotic", "pos_weights"]
        )

        # pos_weight: prefer pos_weights list first (single-AB list), else scalar
        if pos_weights is not None and len(pos_weights) > 0:
            self.pos_weight = float(pos_weights[0])
        else:
            self.pos_weight = float(pos_weight)
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.t_max = scheduler_t_max

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 64,  kernel_size=9, padding=4),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,  96,  kernel_size=7, padding=3),
            nn.BatchNorm1d(96),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(96,  128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) int → (B,) logits."""
        x = self.embed(x.long())        # (B, L, D)
        x = x.transpose(1, 2)           # (B, D, L)
        x = self.conv(x).squeeze(-1)    # (B, 128)
        return self.head(x).squeeze(-1) # (B,)


# ===========================================================================
# Contig-aware Variant A — vectorised per-contig CNN + attention aggregation
# ===========================================================================
class ContigAwareCNN_VariantA(_AMRSingleTaskMixin):
    """Per-contig CNN with attention pooling, fully vectorised.

    Procedure:
        1. Gather SNPs per contig into padded tensor (B, C, max_len).
        2. Mask padding positions → embed → (B*C, D, max_len).
        3. Shared CNN → (B*C, 128).
        4. Reshape (B, C, 128) → attend with learned query → (B, 128).
        5. Head → (B,) logit.

    All contig book-keeping is precomputed in __init__ as non-persistent buffers.
    """

    def __init__(
        self,
        n_features: int,
        gather_idx: np.ndarray,    # (n_contigs, max_contig_len) int64
        valid_mask: np.ndarray,    # (n_contigs, max_contig_len) float32
        vocab_size: int = 5,
        padding_idx: int = 4,
        embed_dim: int = 32,
        dropout: float = 0.4,
        attention_heads: int = 4,
        pos_weight: float = 1.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_t_max: int = 30,
        antibiotic_names: Optional[list[str]] = None,
        target_antibiotic: Optional[str] = None,
        pos_weights: Optional[list[float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["antibiotic_names", "target_antibiotic", "pos_weights",
                    "gather_idx", "valid_mask"]
        )

        if pos_weights is not None and len(pos_weights) > 0:
            self.pos_weight = float(pos_weights[0])
        else:
            self.pos_weight = float(pos_weight)
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.t_max = scheduler_t_max

        self.padding_idx = padding_idx
        self.embed_dim   = embed_dim
        self.n_contigs   = gather_idx.shape[0]
        self.max_contig_len = gather_idx.shape[1]

        self.register_buffer(
            "gather_idx",
            torch.from_numpy(gather_idx).long(),
            persistent=False,
        )
        # (1, n_contigs, max_len) for batched broadcasting
        self.register_buffer(
            "valid_mask",
            torch.from_numpy(valid_mask).float().unsqueeze(0),
            persistent=False,
        )

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.contig_cnn = nn.Sequential(
            nn.Conv1d(embed_dim, 64,  kernel_size=7, padding=3),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,  128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.attn_pool  = nn.MultiheadAttention(
            embed_dim=128, num_heads=attention_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_query = nn.Parameter(torch.randn(1, 1, 128) * 0.02)

        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Gather: (B, n_contigs, max_len)
        flat  = self.gather_idx.view(-1)          # (C*M,)
        gath  = x[:, flat].view(B, self.n_contigs, self.max_contig_len)
        pad   = torch.full_like(gath, self.padding_idx)
        gath  = torch.where(self.valid_mask > 0, gath, pad)

        # Embed + reshape: (B*C, D, M)
        emb = self.embed(gath.long())             # (B, C, M, D)
        emb = emb.permute(0, 1, 3, 2).contiguous()
        emb = emb.view(B * self.n_contigs, self.embed_dim, self.max_contig_len)

        # CNN: (B*C, 128)
        cv  = self.contig_cnn(emb).squeeze(-1)
        cv  = cv.view(B, self.n_contigs, 128)     # (B, C, 128)

        # Attention pool
        q = self.attn_query.expand(B, 1, -1)      # (B, 1, 128)
        # Key-padding mask: mark wholly-invalid contigs
        contig_valid = self.valid_mask.squeeze(0).sum(dim=-1) > 0   # (C,)
        kpm = (~contig_valid).unsqueeze(0).expand(B, -1)            # (B, C)
        pooled, _ = self.attn_pool(
            q, cv, cv,
            key_padding_mask=kpm if kpm.any() else None,
        )
        return self.head(pooled.squeeze(1)).squeeze(-1)


# ===========================================================================
# Contig-aware Variant B — single CNN with boundary-masked activations
# ===========================================================================
class ContigAwareCNN_VariantB(_AMRSingleTaskMixin):
    """Naive-style 1D CNN with contig-boundary masks applied after each conv.

    Positions whose receptive field crosses a contig boundary are zeroed so
    the model cannot exploit spurious cross-contig patterns. Architecture is
    identical to Naive1DCNN; only the activations are masked.
    """

    def __init__(
        self,
        n_features: int,
        contig_ids_per_snp: np.ndarray,   # (n_snps,) int32
        vocab_size: int = 5,
        padding_idx: int = 4,
        embed_dim: int = 32,
        dropout: float = 0.4,
        pos_weight: float = 1.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_t_max: int = 30,
        antibiotic_names: Optional[list[str]] = None,
        target_antibiotic: Optional[str] = None,
        pos_weights: Optional[list[float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["antibiotic_names", "target_antibiotic", "pos_weights",
                    "contig_ids_per_snp"]
        )

        if pos_weights is not None and len(pos_weights) > 0:
            self.pos_weight = float(pos_weights[0])
        else:
            self.pos_weight = float(pos_weight)
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.t_max = scheduler_t_max

        # Precompute boundary masks for each conv layer
        kernel_sizes = [9, 7, 5, 3]
        pool_after   = [True, True, True, False]

        cur_ids = np.asarray(contig_ids_per_snp, dtype=np.int32)
        for i, (k, p) in enumerate(zip(kernel_sizes, pool_after)):
            half     = k // 2
            cur_len  = len(cur_ids)
            mask     = np.ones(cur_len, dtype=np.float32)
            for offset in range(-half, half + 1):
                if offset == 0:
                    continue
                shifted = np.roll(cur_ids, -offset)
                if offset > 0:
                    shifted[-offset:] = -1
                else:
                    shifted[:-offset] = -1
                mask *= (shifted == cur_ids).astype(np.float32)
            self.register_buffer(
                f"mask_{i}",
                torch.from_numpy(mask),
                persistent=False,
            )
            if p:
                new_len = cur_len // 2
                cur_ids = cur_ids[:2 * new_len].reshape(new_len, 2)[:, 0]

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.conv1  = nn.Conv1d(embed_dim, 64,  kernel_size=9, padding=4)
        self.bn1    = nn.BatchNorm1d(64)
        self.pool1  = nn.MaxPool1d(2)
        self.conv2  = nn.Conv1d(64,  96,  kernel_size=7, padding=3)
        self.bn2    = nn.BatchNorm1d(96)
        self.pool2  = nn.MaxPool1d(2)
        self.conv3  = nn.Conv1d(96,  128, kernel_size=5, padding=2)
        self.bn3    = nn.BatchNorm1d(128)
        self.pool3  = nn.MaxPool1d(2)
        self.conv4  = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4    = nn.BatchNorm1d(128)
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 1))

    @staticmethod
    def _mask_act(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x * mask.view(1, 1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x.long()).transpose(1, 2)        # (B, D, L)
        x = self._mask_act(F.relu(self.bn1(self.conv1(x))), self.mask_0)
        x = self.pool1(x)
        x = self._mask_act(F.relu(self.bn2(self.conv2(x))), self.mask_1)
        x = self.pool2(x)
        x = self._mask_act(F.relu(self.bn3(self.conv3(x))), self.mask_2)
        x = self.pool3(x)
        x = self._mask_act(F.relu(self.bn4(self.conv4(x))), self.mask_3)
        x = self.gap(x).squeeze(-1)
        return self.head(x).squeeze(-1)
