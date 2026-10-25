# src/experiments/run_pillar1_exp3.py
"""Pillar 1 Experiment 1.3 — Position shuffling asymmetric drop.

For each dataset ∈ {arcadia, giessen}, train Naive1DCNN on:
    - original column order  (baseline)
    - randomly shuffled column order (single fixed permutation)
The same permutation is applied to train/val/test of that dataset.

Hypothesis:
    Giessen drops ≥5 % PR-AUC under shuffling (real positional structure).
    Arcadia drops ≤2 % PR-AUC (pan-genome already broke locality).
The asymmetric drop is the cleanest mechanistic proof of the locality mismatch.

Outputs in ``cfg.experiment.output_dir``:
    {dataset}_{antibiotic}_{order}_predictions.npz   (order ∈ {orig, shuf})
    exp3_results.csv
    exp3_results.json
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
from src.data.transforms import make_column_permutation
from src.experiments._shared import (
    error_row, make_loader, maybe_wandb_logger, pos_weight_from_labels,
    result_to_row, train_and_evaluate,
)
from src.models.cnn_variants import Naive1DCNN
from src.utils.seeds import set_global_seed


def _build_naive(n_features: int, ab: str, train_y: np.ndarray,
                 cfg: DictConfig) -> Naive1DCNN:
    return Naive1DCNN(
        n_features=n_features,
        vocab_size=cfg.model.vocab_size,
        padding_idx=cfg.model.padding_idx,
        embed_dim=cfg.model.embed_dim,
        dropout=cfg.model.dropout,
        antibiotic_names=[ab],
        target_antibiotic=ab,
        pos_weights=[pos_weight_from_labels(train_y)],
        learning_rate=cfg.experiment.training.learning_rate,
        weight_decay=cfg.experiment.training.weight_decay,
        scheduler_t_max=cfg.experiment.training.scheduler_t_max,
    )


def _run_arcadia(ab: str, perm: np.ndarray | None, order_tag: str,
                 cfg: DictConfig, output_dir: Path) -> dict:
    bundle = load_arcadia_raw_for_antibiotic(
        snp_pkl_path=cfg.paths.arcadia_snp_matrix_pkl,
        master_pkl_path=cfg.paths.arcadia_master_pkl,
        antibiotic=ab,
        split_type=cfg.data.arcadia.split_type,
    )
    train_ds = ArcadiaSNPSingleABDataset(bundle.train, shuffle_perm=perm)
    val_ds = ArcadiaSNPSingleABDataset(bundle.val, shuffle_perm=perm)
    test_ds = ArcadiaSNPSingleABDataset(bundle.test, shuffle_perm=perm)

    bs = cfg.experiment.training.batch_size_arcadia
    nw = cfg.experiment.training.num_workers
    train_loader = make_loader(train_ds, bs, nw, shuffle=True)
    val_loader = make_loader(val_ds, bs, nw, shuffle=False)
    test_loader = make_loader(test_ds, bs, nw, shuffle=False)

    model = _build_naive(bundle.n_snps, ab, bundle.train.y, cfg)
    run_name = f"arcadia_{ab}_naive_{order_tag}"
    wandb_logger = maybe_wandb_logger(
        cfg, run_name, extra_tags=["arcadia", ab, order_tag],
        save_dir=output_dir,
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


def _run_giessen(ab: str, perm: np.ndarray | None, order_tag: str,
                 cfg: DictConfig, output_dir: Path) -> dict:
    giessen_ab = arcadia_to_giessen_ab(ab)
    if giessen_ab is None:
        raise ValueError(f"No Giessen mapping for {ab}")

    bundle = load_giessen_for_antibiotic(
        data_csv_path=cfg.paths.giessen_data_csv,
        pheno_csv_path=cfg.paths.giessen_pheno_csv,
        antibiotic=giessen_ab,
        seed=cfg.seed,
    )
    train_ds = GiessenSingleABDataset(bundle.train, shuffle_perm=perm)
    val_ds = GiessenSingleABDataset(bundle.val, shuffle_perm=perm)
    test_ds = GiessenSingleABDataset(bundle.test, shuffle_perm=perm)

    bs = cfg.experiment.training.batch_size_giessen
    nw = cfg.experiment.training.num_workers
    train_loader = make_loader(train_ds, bs, nw, shuffle=True)
    val_loader = make_loader(val_ds, bs, nw, shuffle=False)
    test_loader = make_loader(test_ds, bs, nw, shuffle=False)

    model = _build_naive(bundle.n_snps, ab, bundle.train.y, cfg)
    run_name = f"giessen_{ab}_naive_{order_tag}"
    wandb_logger = maybe_wandb_logger(
        cfg, run_name, extra_tags=["giessen", ab, order_tag],
        save_dir=output_dir,
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

    for ab in cfg.experiment.antibiotics:
        for dataset in cfg.experiment.datasets:
            for order_tag in ["orig", "shuf"]:
                print("=" * 80)
                print(f"  EXP 1.3 — {dataset.upper()} × {ab} × {order_tag.upper()}")
                print("=" * 80)
                t0 = time.time()

                # Permutation per dataset (n_features differ between datasets)
                if order_tag == "shuf":
                    if dataset == "arcadia":
                        # Generate AFTER we know n_features
                        # We can compute n_features by peeking — but it's cheaper
                        # to defer: build perm inside the _run_xxx call. For
                        # simplicity, we generate here using cached counts:
                        n = 535439
                    else:
                        n = 60936
                    perm = make_column_permutation(n, seed=cfg.experiment.shuffle_seed)
                else:
                    perm = None

                try:
                    if dataset == "arcadia":
                        bundle_result = _run_arcadia(
                            ab, perm, order_tag, cfg, output_dir,
                        )
                    elif dataset == "giessen":
                        bundle_result = _run_giessen(
                            ab, perm, order_tag, cfg, output_dir,
                        )
                    else:
                        raise ValueError(f"Unknown dataset {dataset}")
                except Exception as ex:  # noqa: BLE001
                    print(f"  ERROR: {ex!r}")
                    rows.append(error_row(
                        dataset, ab, f"cnn_1d_naive_{order_tag}", None, ex,
                    ))
                    continue

                result = bundle_result["result"]
                elapsed = time.time() - t0
                row = result_to_row(
                    dataset, ab, "cnn_1d_naive", order_tag, result, elapsed,
                )
                rows.append(row)

                key = f"{dataset}_{ab}_{order_tag}"
                pred_path = output_dir / f"{key}_predictions.npz"
                np.savez_compressed(pred_path, y_true=result.y_true,
                                    y_prob=result.y_prob)
                nested[key] = {
                    "predictions_npz": str(pred_path),
                    "metrics": result.metrics,
                    "pr_auc_ci": list(result.pr_auc_ci),
                    "mcc_ci": list(result.mcc_ci),
                    "best_ckpt": result.best_ckpt,
                }

                m = result.metrics
                print(f"  PR-AUC: {m['pr_auc']:.4f} "
                      f"[{result.pr_auc_ci[0]:.4f}, {result.pr_auc_ci[1]:.4f}]")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "exp3_results.csv", index=False)
    with open(output_dir / "exp3_results.json", "w") as f:
        json.dump(nested, f, indent=2, default=str)

    # Summary — asymmetric drop check
    print("\n" + "=" * 80)
    print("  EXP 1.3 SUMMARY — Asymmetric shuffling drop test")
    print("=" * 80)
    ok = df[df["status"] == "ok"]
    drops: dict[str, float] = {}
    for ds in cfg.experiment.datasets:
        for ab in cfg.experiment.antibiotics:
            orig = ok[(ok["dataset"] == ds) & (ok["antibiotic"] == ab)
                      & (ok["variant"] == "orig")]
            shuf = ok[(ok["dataset"] == ds) & (ok["antibiotic"] == ab)
                      & (ok["variant"] == "shuf")]
            if len(orig) and len(shuf):
                orig_pr = orig["test_pr_auc"].iloc[0]
                shuf_pr = shuf["test_pr_auc"].iloc[0]
                drop = orig_pr - shuf_pr
                drops[f"{ds}_{ab}"] = drop
                print(f"  {ds:<10} {ab:<6}  orig={orig_pr:.4f}  "
                      f"shuf={shuf_pr:.4f}  drop={drop:+.4f}")

    if drops:
        arc_drops = [v for k, v in drops.items() if k.startswith("arcadia")]
        gie_drops = [v for k, v in drops.items() if k.startswith("giessen")]
        if arc_drops and gie_drops:
            arc_avg = float(np.mean(arc_drops))
            gie_avg = float(np.mean(gie_drops))
            asym_ok = (gie_avg >= 0.05) and (arc_avg <= 0.02)
            verdict = "✅ ASYMMETRIC" if asym_ok else "⚠ INCONCLUSIVE"
            print(f"\n  avg shuffle drop Arcadia = {arc_avg:+.4f}")
            print(f"  avg shuffle drop Giessen = {gie_avg:+.4f}")
            print(f"  Asymmetry verdict: {verdict}")
            print(f"  Success criteria: Giessen ≥ 0.05, Arcadia ≤ 0.02")

    print(f"\n  Outputs → {output_dir}")
