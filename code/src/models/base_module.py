"""Base PyTorch Lightning module for all AMR models."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from omegaconf import DictConfig

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.statistical import bootstrap_ci
from sklearn.metrics import average_precision_score


class AMRBaseLightningModule(L.LightningModule):
    """Base Lightning module that all AMR models inherit from.
    
    Handles:
    - Per-antibiotic multi-task loss with label masking
    - Per-antibiotic metric computation
    - Bootstrap CI on test set
    - WandB logging
    - Checkpoint with metadata
    
    Subclasses must implement:
    - build_model(): create self.model
    - forward(x): return logits (B, n_antibiotics) or (B,) for single-task
    """
    
    def __init__(
        self,
        n_antibiotics: int,
        antibiotic_names: list[str],
        pos_weights: Optional[list[float]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_t_max: int = 30,
        target_antibiotic: Optional[str] = None,
    ):
        """
        Args:
            n_antibiotics: Total number of antibiotics (output dim).
            antibiotic_names: Ordered list of antibiotic names.
            pos_weights: Per-antibiotic positive class weights for BCE.
            learning_rate: Initial learning rate.
            weight_decay: Weight decay for AdamW.
            scheduler_t_max: T_max for cosine annealing.
            target_antibiotic: If set, single-task mode for this antibiotic.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weights"])
        
        self.n_antibiotics = n_antibiotics
        self.antibiotic_names = antibiotic_names
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_t_max = scheduler_t_max
        self.target_antibiotic = target_antibiotic
        
        if target_antibiotic is not None:
            self.target_idx = antibiotic_names.index(target_antibiotic)
        else:
            self.target_idx = None
        
        if pos_weights is not None:
            pw = torch.tensor(pos_weights, dtype=torch.float32)
        else:
            pw = torch.ones(n_antibiotics, dtype=torch.float32)
        self.register_buffer("pos_weights", pw)
        
        # Storage for test predictions
        self.test_probs: list[torch.Tensor] = []
        self.test_labels: list[torch.Tensor] = []
        self.test_masks: list[torch.Tensor] = []
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked BCE loss with per-antibiotic positive weighting.
        
        Args:
            logits: Model outputs (B, n_antibiotics) or (B,) single-task.
            y: Binary labels (B, n_antibiotics) or (B,).
            mask: Valid label mask, same shape as y.
        
        Returns:
            Scalar loss.
        """
        if self.target_idx is not None:
            # Single-task: extract target antibiotic
            if y.dim() == 2:
                y_t = y[:, self.target_idx]
                mask_t = mask[:, self.target_idx]
            else:
                y_t = y
                mask_t = mask if mask.dim() == 1 else mask[:, self.target_idx]
            
            if logits.dim() == 2:
                logits_t = logits[:, self.target_idx] if logits.size(1) == self.n_antibiotics else logits.squeeze(-1)
            else:
                logits_t = logits
            
            pw = self.pos_weights[self.target_idx]
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
            losses = loss_fn(logits_t, y_t.float())
            losses = losses * mask_t.float()
            return losses.sum() / mask_t.float().sum().clamp(min=1)
        
        # Multi-task: BCE per antibiotic, masked
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        losses = loss_fn(logits, y.float())  # (B, n_ab)
        
        # Apply per-antibiotic pos weights
        pw_expanded = self.pos_weights.unsqueeze(0)  # (1, n_ab)
        y_float = y.float()
        weight = y_float * pw_expanded + (1 - y_float)
        losses = losses * weight
        
        # Mask invalid labels
        losses = losses * mask.float()
        return losses.sum() / mask.float().sum().clamp(min=1)
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        X, y, mask = batch["X"], batch["Y"], batch["mask"]
        logits = self(X)
        loss = self.compute_loss(logits, y, mask)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        X, y, mask = batch["X"], batch["Y"], batch["mask"]
        logits = self(X)
        loss = self.compute_loss(logits, y, mask)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store for epoch-end PR-AUC
        probs = torch.sigmoid(logits).detach()
        if not hasattr(self, "_val_probs"):
            self._val_probs, self._val_labels, self._val_masks = [], [], []
        self._val_probs.append(probs.cpu())
        self._val_labels.append(y.cpu())
        self._val_masks.append(mask.cpu())
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        if not hasattr(self, "_val_probs") or len(self._val_probs) == 0:
            return
        
        probs = torch.cat(self._val_probs).numpy()
        labels = torch.cat(self._val_labels).numpy()
        masks = torch.cat(self._val_masks).numpy()
        
        if self.target_idx is not None:
            if probs.ndim == 2:
                p = probs[:, self.target_idx] if probs.shape[1] == self.n_antibiotics else probs.squeeze(-1)
            else:
                p = probs
            y = labels[:, self.target_idx]
            m = masks[:, self.target_idx].astype(bool)
            if m.sum() > 0 and len(np.unique(y[m])) > 1:
                pr = average_precision_score(y[m], p[m])
                self.log("val/pr_auc", pr, prog_bar=True)
        else:
            prs = []
            for i in range(self.n_antibiotics):
                m = masks[:, i].astype(bool)
                if m.sum() < 10 or len(np.unique(labels[m, i])) < 2:
                    continue
                pr = average_precision_score(labels[m, i], probs[m, i])
                prs.append(pr)
                self.log(f"val/pr_auc_{self.antibiotic_names[i]}", pr)
            if prs:
                self.log("val/pr_auc", float(np.mean(prs)), prog_bar=True)
        
        # Clear
        self._val_probs, self._val_labels, self._val_masks = [], [], []
    
    def test_step(self, batch: dict, batch_idx: int) -> None:
        X, y, mask = batch["X"], batch["Y"], batch["mask"]
        logits = self(X)
        probs = torch.sigmoid(logits).detach()
        
        self.test_probs.append(probs.cpu())
        self.test_labels.append(y.cpu())
        self.test_masks.append(mask.cpu())
    
    def on_test_epoch_end(self) -> None:
        probs = torch.cat(self.test_probs).numpy()
        labels = torch.cat(self.test_labels).numpy()
        masks = torch.cat(self.test_masks).numpy()
        
        results = {"per_antibiotic": {}, "average": {}}
        
        if self.target_idx is not None:
            ab_indices = [self.target_idx]
            if probs.ndim == 1 or probs.shape[1] == 1:
                probs = probs.reshape(-1, 1) if probs.ndim == 1 else probs
                ab_indices_in_probs = [0]
            else:
                ab_indices_in_probs = [self.target_idx]
        else:
            ab_indices = list(range(self.n_antibiotics))
            ab_indices_in_probs = ab_indices
        
        pr_aucs, mccs = [], []
        for ab_i, prob_i in zip(ab_indices, ab_indices_in_probs):
            ab_name = self.antibiotic_names[ab_i]
            m = masks[:, ab_i].astype(bool)
            
            if m.sum() < 10 or len(np.unique(labels[m, ab_i])) < 2:
                continue
            
            y_true = labels[m, ab_i]
            y_prob = probs[m, prob_i]
            
            metrics = compute_all_metrics(y_true, y_prob)
            
            # Bootstrap CI on PR-AUC
            pr_point, pr_lo, pr_hi = bootstrap_ci(
                y_true, y_prob, average_precision_score, n_bootstrap=1000, seed=42
            )
            metrics["pr_auc_ci_lo"] = pr_lo
            metrics["pr_auc_ci_hi"] = pr_hi
            
            results["per_antibiotic"][ab_name] = metrics
            pr_aucs.append(metrics["pr_auc"])
            mccs.append(metrics["mcc"])
            
            for k, v in metrics.items():
                self.log(f"test/{ab_name}/{k}", v)
        
        if pr_aucs:
            results["average"]["pr_auc"] = float(np.mean(pr_aucs))
            results["average"]["mcc"] = float(np.mean(mccs))
            self.log("test/avg_pr_auc", float(np.mean(pr_aucs)))
            self.log("test/avg_mcc", float(np.mean(mccs)))
        
        self._test_results = results
        
        # Clear
        self.test_probs, self.test_labels, self.test_masks = [], [], []
    
    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.scheduler_t_max, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
    
    def save_results_json(self, path: str | Path) -> None:
        """Save test results dictionary as JSON.
        
        Args:
            path: Output file path.
        """
        if not hasattr(self, "_test_results"):
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._test_results, f, indent=2)


def compute_pos_weights(Y: np.ndarray, mask: np.ndarray) -> list[float]:
    """Compute per-antibiotic positive class weights for BCE.
    
    Args:
        Y: Label matrix (n_samples, n_antibiotics).
        mask: Valid label mask same shape.
    
    Returns:
        List of weights, length = n_antibiotics.
    """
    weights = []
    for i in range(Y.shape[1]):
        m = mask[:, i].astype(bool)
        if m.sum() == 0:
            weights.append(1.0)
            continue
        y = Y[m, i]
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        w = n_neg / max(n_pos, 1)
        weights.append(float(w))
    return weights