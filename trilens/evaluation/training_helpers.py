from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, np.nan_to_num(y_prob, nan=0.5)))


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def sensitivity_at_specificity(y_true: np.ndarray, y_prob: np.ndarray, target_spec: float = 0.9) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    valid = spec >= target_spec
    return float(tpr[valid].max()) if valid.any() else float("nan")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:

    y_pred = (y_prob >= threshold).astype(int)
    out: dict[str, float] = {
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "acc": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, np.clip(y_prob, 0, 1))),
        "sens_at_90_spec": sensitivity_at_specificity(y_true, y_prob, 0.9),
    }
    return out


def bootstrap_ci_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
    ci_lo: float = 2.5,
    ci_hi: float = 97.5,
) -> tuple[float, float]:

    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            s = metric_fn(y_true[idx], y_prob[idx])
            if not np.isnan(s):
                scores.append(s)
        except Exception:
            continue
    if len(scores) < max(50, n_boot // 20):
        return float("nan"), float("nan")
    arr = np.asarray(scores)
    return float(np.percentile(arr, ci_lo)), float(np.percentile(arr, ci_hi))


def bootstrap_pr_auc(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    return bootstrap_ci_metric(y_true, y_prob, safe_pr_auc, n_boot=n_boot, seed=seed)


def bootstrap_mcc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
) -> tuple[float, float]:
    def _mcc(y, p):
        pred = (p >= threshold).astype(int)
        if len(np.unique(pred)) < 2:
            return 0.0
        return float(matthews_corrcoef(y, pred))

    return bootstrap_ci_metric(y_true, y_prob, _mcc, n_boot=n_boot, seed=seed)


def pos_weight_from_labels(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return max(1.0, neg / max(1.0, pos))


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    import platform

    if platform.system() == "Windows" and num_workers > 0:
        num_workers = 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )


@dataclass
class TrainEvalResult:
    metrics: dict[str, float]
    pr_auc_ci: tuple[float, float]
    mcc_ci: tuple[float, float]
    y_true: np.ndarray
    y_prob: np.ndarray
    fit_time_s: float
    best_ckpt: str
    epochs_run: int


def train_and_evaluate(
    *,
    model: L.LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    training_cfg: DictConfig,
    evaluation_cfg: DictConfig,
    run_name: str,
    ckpt_dir: Path,
    wandb_logger: Optional[WandbLogger],
) -> TrainEvalResult:

    es = EarlyStopping(
        monitor=training_cfg.early_stop_metric,
        mode=training_cfg.early_stop_mode,
        patience=training_cfg.early_stop_patience,
        verbose=False,
    )
    mc = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{run_name}_best",
        monitor=training_cfg.early_stop_metric,
        mode=training_cfg.early_stop_mode,
        save_top_k=1,
        save_last=False,
        verbose=False,
    )

    trainer = L.Trainer(
        max_epochs=training_cfg.max_epochs,
        min_epochs=training_cfg.min_epochs,
        accelerator="gpu",
        devices=1,
        precision=training_cfg.precision,
        gradient_clip_val=training_cfg.gradient_clip,
        callbacks=[es, mc],
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=False,
    )

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    fit_time = time.time() - t0
    epochs_run = trainer.current_epoch + 1

    best_path = mc.best_model_path
    if best_path:
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])

    model.eval()
    device = next(model.parameters()).device
    all_probs: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            if logits.ndim == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_y.append(yb.numpy())
    y_true = np.concatenate(all_y).astype(np.int32)
    y_prob = np.concatenate(all_probs).astype(np.float32)

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    pr_ci = bootstrap_pr_auc(
        y_true,
        y_prob,
        n_boot=evaluation_cfg.bootstrap_n,
        seed=evaluation_cfg.bootstrap_seed,
    )
    mcc_ci = bootstrap_mcc(
        y_true,
        y_prob,
        n_boot=evaluation_cfg.bootstrap_n,
        seed=evaluation_cfg.bootstrap_seed,
    )

    return TrainEvalResult(
        metrics=metrics,
        pr_auc_ci=pr_ci,
        mcc_ci=mcc_ci,
        y_true=y_true,
        y_prob=y_prob,
        fit_time_s=fit_time,
        best_ckpt=str(best_path),
        epochs_run=epochs_run,
    )


def maybe_wandb_logger(
    cfg: DictConfig,
    run_name: str,
    extra_tags: Sequence[str],
    save_dir: Path,
) -> Optional[WandbLogger]:

    if not cfg.get("use_wandb", True):
        return None
    try:
        tags = list(cfg.experiment.wandb.tags) + list(extra_tags)
        return WandbLogger(
            project=cfg.experiment.wandb.project,
            name=run_name,
            tags=tags,
            save_dir=str(save_dir),
        )
    except Exception as ex:
        print(f"  [WARN] WandbLogger init failed ({ex!r}); continuing without it.")
        return None


def result_to_row(
    dataset: str,
    antibiotic: str,
    model_name: str,
    variant: str | None,
    result: TrainEvalResult,
    elapsed: float,
) -> dict[str, Any]:

    m = result.metrics
    return {
        "dataset": dataset,
        "antibiotic": antibiotic,
        "model": model_name,
        "variant": variant or "",
        "status": "ok",
        "test_pr_auc": m["pr_auc"],
        "test_pr_auc_lo": result.pr_auc_ci[0],
        "test_pr_auc_hi": result.pr_auc_ci[1],
        "test_mcc": m["mcc"],
        "test_mcc_lo": result.mcc_ci[0],
        "test_mcc_hi": result.mcc_ci[1],
        "test_roc_auc": m["roc_auc"],
        "test_f1": m["f1"],
        "test_f2": m["f2"],
        "test_precision": m["precision"],
        "test_recall": m["recall"],
        "test_acc": m["acc"],
        "test_brier": m["brier"],
        "test_sens_at_90_spec": m["sens_at_90_spec"],
        "n_test": int(len(result.y_true)),
        "n_pos_test": int(result.y_true.sum()),
        "epochs_run": result.epochs_run,
        "fit_time_s": result.fit_time_s,
        "elapsed_s": elapsed,
        "best_ckpt": result.best_ckpt,
    }


def error_row(dataset: str, antibiotic: str, model_name: str, variant: str | None, error: Exception) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "antibiotic": antibiotic,
        "model": model_name,
        "variant": variant or "",
        "status": "error",
        "error": repr(error),
    }
