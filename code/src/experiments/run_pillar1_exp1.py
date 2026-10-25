# src/experiments/run_pillar1_exp1.py
"""Pillar 1 Experiment 1.1 — Giessen vs Arcadia controlled 1D CNN comparison.

For each (dataset ∈ {arcadia, giessen}) × (antibiotic ∈ exp.antibiotics):
    - Load raw-SNP data + labels for the single antibiotic.
    - Build identical-architecture Naive1DCNN with locked hyperparameters.
    - Train, early-stop on val PR-AUC, evaluate on test.
    - Record metrics + 1000-bootstrap 95% CI on PR-AUC and MCC.

Outputs in ``cfg.experiment.output_dir``:
    {dataset}_{antibiotic}_predictions.npz
    exp1_results.csv
    exp1_results.json
    config_used.yaml
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.arcadia_snp_loader import (
    ArcadiaSNPSingleABDataset, load_arcadia_raw_for_antibiotic,
)
from src.data.giessen_loader import (
    GiessenSingleABDataset, arcadia_to_giessen_ab, load_giessen_for_antibiotic,
)
from src.experiments._shared import (
    error_row, make_loader, maybe_wandb_logger, pos_weight_from_labels,
    result_to_row, train_and_evaluate,
)
from src.models.cnn_variants import Naive1DCNN
from src.utils.seeds import set_global_seed


def _build_naive_cnn(
    n_features: int, antibiotic: str, train_y: np.ndarray, cfg: DictConfig,
) -> Naive1DCNN:
    return Naive1DCNN(
        n_features=n_features,
        vocab_size=cfg.model.vocab_size,
        padding_idx=cfg.model.padding_idx,
        embed_dim=cfg.model.embed_dim,
        dropout=cfg.model.dropout,
        antibiotic_names=[antibiotic],
        target_antibiotic=antibiotic,
        pos_weights=[pos_weight_from_labels(train_y)],
        learning_rate=cfg.experiment.training.learning_rate,
        weight_decay=cfg.experiment.training.weight_decay,
        scheduler_t_max=cfg.experiment.training.scheduler_t_max,
    )


def _run_arcadia(antibiotic: str, cfg: DictConfig, output_dir: Path) -> dict:
    bundle = load_arcadia_raw_for_antibiotic(
        snp_pkl_path=cfg.paths.arcadia_snp_matrix_pkl,
        master_pkl_path=cfg.paths.arcadia_master_pkl,
        antibiotic=antibiotic,
        split_type=cfg.data.arcadia.split_type,
    )
    train_ds = ArcadiaSNPSingleABDataset(bundle.train)
    val_ds = ArcadiaSNPSingleABDataset(bundle.val)
    test_ds = ArcadiaSNPSingleABDataset(bundle.test)

    bs = cfg.experiment.training.batch_size_arcadia
    nw = cfg.experiment.training.num_workers
    train_loader = make_loader(train_ds, bs, nw, shuffle=True)
    val_loader = make_loader(val_ds, bs, nw, shuffle=False)
    test_loader = make_loader(test_ds, bs, nw, shuffle=False)

    model = _build_naive_cnn(bundle.n_snps, antibiotic, bundle.train.y, cfg)
    run_name = f"arcadia_{antibiotic}_naive_cnn"
    wandb_logger = maybe_wandb_logger(
        cfg, run_name, extra_tags=["arcadia", antibiotic], save_dir=output_dir,
    )

    result = train_and_evaluate(
        model=model,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        training_cfg=cfg.experiment.training,
        evaluation_cfg=cfg.experiment.evaluation,
        run_name=run_name, ckpt_dir=output_dir / "ckpts",
        wandb_logger=wandb_logger,
    )
    return {"result": result, "n_snps": bundle.n_snps}


def _run_giessen(antibiotic: str, cfg: DictConfig, output_dir: Path) -> dict:
    giessen_ab = arcadia_to_giessen_ab(antibiotic)
    if giessen_ab is None:
        raise ValueError(f"No Giessen mapping for Arcadia antibiotic {antibiotic}")

    bundle = load_giessen_for_antibiotic(
        data_csv_path=cfg.paths.giessen_data_csv,
        pheno_csv_path=cfg.paths.giessen_pheno_csv,
        antibiotic=giessen_ab,
        seed=cfg.seed,
    )
    train_ds = GiessenSingleABDataset(bundle.train)
    val_ds = GiessenSingleABDataset(bundle.val)
    test_ds = GiessenSingleABDataset(bundle.test)

    bs = cfg.experiment.training.batch_size_giessen
    nw = cfg.experiment.training.num_workers
    train_loader = make_loader(train_ds, bs, nw, shuffle=True)
    val_loader = make_loader(val_ds, bs, nw, shuffle=False)
    test_loader = make_loader(test_ds, bs, nw, shuffle=False)

    model = _build_naive_cnn(bundle.n_snps, antibiotic, bundle.train.y, cfg)
    run_name = f"giessen_{antibiotic}_naive_cnn"
    wandb_logger = maybe_wandb_logger(
        cfg, run_name, extra_tags=["giessen", antibiotic], save_dir=output_dir,
    )

    result = train_and_evaluate(
        model=model,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        training_cfg=cfg.experiment.training,
        evaluation_cfg=cfg.experiment.evaluation,
        run_name=run_name, ckpt_dir=output_dir / "ckpts",
        wandb_logger=wandb_logger,
    )
    return {"result": result, "n_snps": bundle.n_snps}


def run(cfg: DictConfig) -> None:
    """Hydra entry-point."""
    set_global_seed(cfg.seed)
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ckpts").mkdir(exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config_used.yaml")

    rows: list[dict] = []
    nested: dict[str, dict] = {}

    for dataset in cfg.experiment.datasets:
        for ab in cfg.experiment.antibiotics:
            print("=" * 80)
            print(f"  EXP 1.1 — {dataset.upper()} × {ab}")
            print("=" * 80)
            t0 = time.time()
            try:
                if dataset == "arcadia":
                    bundle_result = _run_arcadia(ab, cfg, output_dir)
                elif dataset == "giessen":
                    bundle_result = _run_giessen(ab, cfg, output_dir)
                else:
                    raise ValueError(f"Unknown dataset {dataset}")
            except Exception as ex:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"  ERROR: {ex!r}")
                rows.append(error_row(dataset, ab, "cnn_1d_naive", None, ex))
                continue

            result = bundle_result["result"]
            elapsed = time.time() - t0
            row = result_to_row(dataset, ab, "cnn_1d_naive", None, result, elapsed)
            rows.append(row)

            pred_path = output_dir / f"{dataset}_{ab}_predictions.npz"
            np.savez_compressed(pred_path, y_true=result.y_true, y_prob=result.y_prob)
            nested[f"{dataset}_{ab}"] = {
                "predictions_npz": str(pred_path),
                "metrics": result.metrics,
                "pr_auc_ci": list(result.pr_auc_ci),
                "mcc_ci": list(result.mcc_ci),
                "best_ckpt": result.best_ckpt,
                "n_snps": bundle_result["n_snps"],
            }

            m = result.metrics
            print(f"  PR-AUC: {m['pr_auc']:.4f}  "
                  f"[{result.pr_auc_ci[0]:.4f}, {result.pr_auc_ci[1]:.4f}]")
            print(f"  MCC   : {m['mcc']:.4f}  "
                  f"[{result.mcc_ci[0]:.4f}, {result.mcc_ci[1]:.4f}]")
            print(f"  F2={m['f2']:.4f}  Sens@90={m['sens_at_90_spec']:.4f}")

    # Persist
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "exp1_results.csv", index=False)
    with open(output_dir / "exp1_results.json", "w") as f:
        json.dump(nested, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 80)
    print("  EXP 1.1 SUMMARY — Locality mismatch hypothesis test")
    print("=" * 80)
    ok = df[df["status"] == "ok"]
    if not ok.empty:
        for ds in cfg.experiment.datasets:
            sub = ok[ok["dataset"] == ds]
            if len(sub):
                print(f"  {ds:<10}  avg PR-AUC = {sub['test_pr_auc'].mean():.4f}  "
                      f"avg MCC = {sub['test_mcc'].mean():.4f}  "
                      f"(n_abs={len(sub)})")
        if {"arcadia", "giessen"}.issubset(set(ok["dataset"])):
            gap = (ok[ok["dataset"] == "giessen"]["test_pr_auc"].mean()
                   - ok[ok["dataset"] == "arcadia"]["test_pr_auc"].mean())
            verdict = "✅ CONFIRMED" if gap >= 0.10 else "⚠ INCONCLUSIVE"
            print(f"\n  Gap (Giessen − Arcadia) avg PR-AUC = {gap:+.4f}  [{verdict}]")
            print(f"  Success criterion: gap ≥ 0.10")

    print(f"\n  Outputs → {output_dir}")
