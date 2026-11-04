from __future__ import annotations

from typing import Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


class _PerGeneEncoder(nn.Module):
    def __init__(self, n_stats: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_stats, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        x = x + self.ffn(self.norm2(x))
        return x


class _AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        kv = self.norm(x)
        pooled, attn_w = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        return pooled.squeeze(1), attn_w


class GeneAttentionNet(L.LightningModule):
    def __init__(
        self,
        n_genes: int,
        n_stats: int,
        n_antibiotics: int,
        antibiotic_names: list[str],
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_mult: int = 2,
        dropout: float = 0.3,
        pos_weights: Optional[list[float]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_t_max: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weights", "antibiotic_names"])

        self.n_genes = n_genes
        self.n_antibiotics = n_antibiotics
        self.antibiotic_names = list(antibiotic_names)
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.t_max = scheduler_t_max

        if pos_weights is None:
            pos_weights = [1.0] * n_antibiotics

        self.register_buffer(
            "pos_weights",
            torch.tensor(pos_weights, dtype=torch.float32),
        )

        self.gene_encoder = _PerGeneEncoder(n_stats, hidden_dim, dropout)
        self.gene_pos_emb = nn.Embedding(n_genes, hidden_dim)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(hidden_dim, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
        )
        self.pool = _AttentionPool(hidden_dim, n_heads, dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_antibiotics)])

        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = self.gene_encoder(x)

        gene_ids = torch.arange(self.n_genes, device=x.device)
        pos = self.gene_pos_emb(gene_ids)
        h = h + pos.unsqueeze(0)

        for blk in self.blocks:
            h = blk(h)

        pooled, attn_w = self.pool(h)
        self.last_attention = attn_w.detach()

        logits = torch.cat([head(pooled) for head in self.heads], dim=1)
        return logits

    def _loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pw = self.pos_weights.view(1, -1).expand_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pw, reduction="none")
        denom = mask.float().sum().clamp(min=1.0)
        return (bce * mask.float()).sum() / denom

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        logits = self(x)
        loss = self._loss(logits, y, mask)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_logits: list[torch.Tensor] = []
        self._val_y: list[torch.Tensor] = []
        self._val_mask: list[torch.Tensor] = []

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        logits = self(x)
        loss = self._loss(logits, y, mask)
        self.log("val/loss", loss, on_epoch=True, prog_bar=False)
        self._val_logits.append(logits.detach().cpu())
        self._val_y.append(y.detach().cpu().long())
        self._val_mask.append(mask.detach().cpu().bool())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_logits:
            self.log("val/pr_auc", 0.0, prog_bar=True)
            return
        logits = torch.cat(self._val_logits).numpy()
        y = torch.cat(self._val_y).numpy()
        mask = torch.cat(self._val_mask).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

        per_ab = []
        for i, name in enumerate(self.antibiotic_names):
            m = mask[:, i]
            if m.sum() == 0:
                continue
            yy = y[m, i]
            pp = probs[m, i]
            if len(np.unique(yy)) < 2:
                continue
            pr = float(average_precision_score(yy, pp))
            per_ab.append(pr)
            self.log(f"val/pr_auc_{name}", pr, prog_bar=False)
        avg = float(np.mean(per_ab)) if per_ab else 0.0
        self.log("val/pr_auc", avg, prog_bar=True)

    def on_test_epoch_start(self):
        self._test_logits: list[torch.Tensor] = []
        self._test_y: list[torch.Tensor] = []
        self._test_mask: list[torch.Tensor] = []

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        logits = self(x)
        self._test_logits.append(logits.detach().cpu())
        self._test_y.append(y.detach().cpu().long())
        self._test_mask.append(mask.detach().cpu().bool())

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.t_max, eta_min=1e-6)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]
