from __future__ import annotations

from typing import Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


class _AMRSingleTaskMixin(L.LightningModule):
    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor([self.pos_weight], device=self.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pw)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

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
        probs = torch.cat(self._val_probs).numpy().astype(np.float32)
        labels = torch.cat(self._val_labels).numpy().astype(np.int32)
        if len(np.unique(labels)) < 2:
            pr_auc = 0.0
        else:
            pr_auc = float(average_precision_score(labels, probs))
        self.log("val/pr_auc", pr_auc, prog_bar=True)

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
        probs = torch.cat(self._test_probs).numpy().astype(np.float32)
        labels = torch.cat(self._test_labels).numpy().astype(np.int32)
        if len(np.unique(labels)) >= 2:
            pr_auc = float(average_precision_score(labels, probs))
        else:
            pr_auc = 0.0
        self.log("test/pr_auc", pr_auc)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.t_max, eta_min=1e-6)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]


class Naive1DCNN(_AMRSingleTaskMixin):
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
        antibiotic_names: Optional[list[str]] = None,
        target_antibiotic: Optional[str] = None,
        pos_weights: Optional[list[float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["antibiotic_names", "target_antibiotic", "pos_weights"])

        if pos_weights is not None and len(pos_weights) > 0:
            self.pos_weight = float(pos_weights[0])
        else:
            self.pos_weight = float(pos_weight)
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.t_max = scheduler_t_max

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=7, padding=3),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x.long())
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.head(x).squeeze(-1)
