# src/experiments/run_pillar1_exp2.py
"""Pillar 1 Experiment 1.2 — Contig-aware CNN recovers Arcadia performance.

For each antibiotic, train:
    (a) Naive1DCNN on Arcadia raw SNPs.
    (b) ContigAwareCNN_VariantA on Arcadia raw SNPs.
Compare PR-AUC + MCC. Hypothesis: variant A recovers ≥0.05 PR-AUC over naive.

If the experiment config requests Variant B, that is also run.

Outputs in ``cfg.experiment.output_dir``:
    arcadia_{antibiotic}_{model}_predictions.npz
    exp2_results.csv
    exp2_results.json
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
from src.data.contig_parser import (
    build_contig_gather_index, get_contig_size_stats, parse_contig_boundaries,
)
from src.experiments._shared import (
    error_row, make_loader, maybe_wandb_logger, pos_weight_from_labels,
    result_to_row, train_and_evaluate,
)
from src.models.cnn_variants import (
    ContigAwareCNN_VariantA, ContigAwareCNN_VariantB, Naive1DCNN,
)
from src.utils.seeds import set_global_seed


# ---------------------------------------------------------------------------
# Caching of contig structure
# ---------------------------------------------------------------------------
class ContigStructureCache:
    """Compute ``contig_ranges``, ``contig_ids_per_snp``, gather indices once
    across the whole experiment to amortise the parser cost (~535K strings)."""

    def __init__(self) -> None:
        self.parsed: bool = False
        self.contig_ranges: dict[str, tuple[int, int]] = {}
        self.contig_ids_per_snp: np.ndarray | None = None
        self.contig_id_list: np.ndarray | None = None
        self.gather_idx: np.ndarray | None = None
        self.valid_mask: np.ndarray | None = None
        self.max_contig_len: int = 0
        self.size_stats: dict = {}

    def ensure_parsed(self, snp_positions: np.ndarray,
                      min_contig_size: int, max_contig_len: int | None) -> None:
        if self.parsed:
            return
        print(f"  [Contig parser] parsing {len(snp_positions):,} positions...")
        t0 = time.time()
        ranges, ids_per_snp, id_list = parse_contig_boundaries(snp_positions)
        gi, vm, _, mlen = build_contig_gather_index(
            ids_per_snp,
            min_contig_size=min_contig_size,
            max_contig_len=max_contig_len,
        )
        stats = get_contig_size_stats(ranges)
        print(f"  [Contig parser] {stats['n_contigs']} contigs total, "
              f"{gi.shape[0]} kept (>= {min_contig_size} SNPs), "
              f"max_contig_len={mlen}, parsed in {time.time()-t0:.1f}s")
        print(f"  [Contig parser] sizes: min={stats['min_size']}, "
              f"median={stats['median_size']:.0f}, "
              f"p95={stats['p95_size']:.0f}, max={stats['max_size']}")
        self.contig_ranges = ranges
        self.contig_ids_per_snp = ids_per_snp
        self.contig_id_list = id_list
        self.gather_idx = gi
        self.valid_mask = vm
        self.max_contig_len = mlen
        self.size_stats = stats
        self.parsed = True


_CONTIG_CACHE = ContigStructureCache()


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def _build_naive(n_features: int, ab: str, train_y: np.ndarray,
                 cfg: DictConfig) -> Naive1DCNN:
    return Naive1DCNN(
        n_features=n_features,
        vocab_size=cfg.model.vocab_size if "model" in cfg else 5,
        padding_idx=cfg.model.padding_idx if "model" in cfg else 4,
        embed_dim=cfg.model.embed_dim if "model" in cfg else 32,
        dropout=cfg.model.dropout if "model" in cfg else 0.4,
        antibiotic_names=[ab],
        target_antibiotic=ab,
        pos_weights=[pos_weight_from_labels(train_y)],
        learning_rate=cfg.experiment.training.learning_rate,
        weight_decay=cfg.experiment.training.weight_decay,
        scheduler_t_max=cfg.experiment.training.scheduler_t_max,
    )


def _build_contig_aware_A(n_features: int, ab: str, train_y: np.ndarray,
                          cfg: DictConfig) -> ContigAwareCNN_VariantA:
    assert _CONTIG_CACHE.parsed, "Contig cache not initialised"
    return ContigAwareCNN_VariantA(
        n_features=n_features,
        gather_idx=_CONTIG_CACHE.gather_idx,
        valid_mask=_CONTIG_CACHE.valid_mask,
        vocab_size=5,
        padding_idx=4,
        embed_dim=32,
        dropout=0.4,
        attention_heads=4,
        antibiotic_names=[ab],
        target_antibiotic=ab,
        pos_weights=[pos_weight_from_labels(train_y)],
        learning_rate=cfg.experiment.training.learning_rate,
        weight_decay=cfg.experiment.training.weight_decay,
        scheduler_t_max=cfg.experiment.training.scheduler_t_max,
    )


def _build_contig_aware_B(n_features: int, ab: str, train_y: np.ndarray,
                          cfg: DictConfig) -> ContigAwareCNN_VariantB:
    assert _CONTIG_CACHE.parsed, "Contig cache not initialised"
    return ContigAwareCNN_VariantB(
        n_features=n_features,
        contig_ids_per_snp=_CONTIG_CACHE.contig_ids_per_snp,
        vocab_size=5,
        padding_idx=4,
        embed_dim=32,
        dropout=0.4,
        antibiotic_names=[ab],
        target_antibiotic=ab,
        pos_weights=[pos_weight_from_labels(train_y)],
        learning_rate=cfg.experiment.training.learning_rate,
        weight_decay=cfg.experiment.training.weight_decay,
        scheduler_t_max=cfg.experiment.training.scheduler_t_max,
    )


_MODEL_FACTORY = {
    "cnn_1d_naive": _build_naive,
    "cnn_1d_contig_aware_A": _build_contig_aware_A,
    "cnn_1d_contig_aware_B": _build_contig_aware_B,
}


# ---------------------------------------------------------------------------
# Single (model × antibiotic) cycle
# ---------------------------------------------------------------------------
def _run_one(model_name: str, ab: str, cfg: DictConfig, output_dir: Path) -> dict:
    bundle = load_arcadia_raw_for_antibiotic(
        snp_pkl_path=cfg.paths.arcadia_snp_matrix_pkl,
        master_pkl_path=cfg.paths.arcadia_master_pkl,
        antibiotic=ab,
        split_type=cfg.data.arcadia.split_type,
    )

    if model_name.startswith("cnn_1d_contig_aware"):
        # Reuse cached snp_positions parsed at first contig-aware call
        _CONTIG_CACHE.ensure_parsed(
            snp_positions=bundle.snp_positions,
            min_contig_size=16,
            max_contig_len=None,
        )

    train_ds = ArcadiaSNPSingleABDataset(bundle.train)
    val_ds = ArcadiaSNPSingleABDataset(bundle.val)
    test_ds = ArcadiaSNPSingleABDataset(bundle.test)

    bs = cfg.experiment.training.batch_size
    nw = cfg.experiment.training.num_workers
    train_loader = make_loader(train_ds, bs, nw, shuffle=True)
    val_loader = make_loader(val_ds, bs, nw, shuffle=False)
    test_loader = make_loader(test_ds, bs, nw, shuffle=False)

    model = _MODEL_FACTORY[model_name](bundle.n_snps, ab, bundle.train.y, cfg)

    run_name = f"arcadia_{ab}_{model_name}"
    wandb_logger = maybe_wandb_logger(
        cfg, run_name, extra_tags=["arcadia", ab, model_name], save_dir=output_dir,
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

    # Resolve which variants to run
    requested_models = list(cfg.experiment.models)
    variant = str(cfg.experiment.contig_aware_variant).upper()
    resolved: list[str] = []
    for m in requested_models:
        if m == "cnn_1d_contig_aware":
            resolved.append(f"cnn_1d_contig_aware_{variant}")
        else:
            resolved.append(m)

    rows: list[dict] = []
    nested: dict[str, dict] = {}

    for ab in cfg.experiment.antibiotics:
        for model_name in resolved:
            print("=" * 80)
            print(f"  EXP 1.2 — ARCADIA × {ab} × {model_name}")
            print("=" * 80)
            t0 = time.time()
            try:
                bundle_result = _run_one(model_name, ab, cfg, output_dir)
            except Exception as ex:  # noqa: BLE001
                print(f"  ERROR: {ex!r}")
                rows.append(error_row("arcadia", ab, model_name, variant, ex))
                continue

            result = bundle_result["result"]
            elapsed = time.time() - t0
            row = result_to_row(
                "arcadia", ab, model_name,
                variant if "contig_aware" in model_name else None,
                result, elapsed,
            )
            rows.append(row)

            key = f"arcadia_{ab}_{model_name}"
            pred_path = output_dir / f"{key}_predictions.npz"
            np.savez_compressed(pred_path, y_true=result.y_true, y_prob=result.y_prob)
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
            print(f"  MCC:    {m['mcc']:.4f} "
                  f"[{result.mcc_ci[0]:.4f}, {result.mcc_ci[1]:.4f}]")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "exp2_results.csv", index=False)
    with open(output_dir / "exp2_results.json", "w") as f:
        json.dump(nested, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("  EXP 1.2 SUMMARY — Boundary-awareness recovery test")
    print("=" * 80)
    ok = df[df["status"] == "ok"]
    if not ok.empty:
        for m_name in resolved:
            sub = ok[ok["model"] == m_name]
            if len(sub):
                print(f"  {m_name:<28}  avg PR-AUC = {sub['test_pr_auc'].mean():.4f}  "
                      f"avg MCC = {sub['test_mcc'].mean():.4f}")
        if "cnn_1d_naive" in resolved and any(m.startswith("cnn_1d_contig_aware")
                                              for m in resolved):
            ca_name = next(m for m in resolved if m.startswith("cnn_1d_contig_aware"))
            naive_avg = ok[ok["model"] == "cnn_1d_naive"]["test_pr_auc"].mean()
            ca_avg = ok[ok["model"] == ca_name]["test_pr_auc"].mean()
            delta = ca_avg - naive_avg
            verdict = "✅ RECOVERED" if delta >= 0.05 else "⚠ INCONCLUSIVE"
            print(f"\n  Δ avg PR-AUC ({ca_name} − cnn_1d_naive) = {delta:+.4f}  "
                  f"[{verdict}]")
            print(f"  Success criterion: Δ ≥ 0.05")

    print(f"\n  Outputs → {output_dir}")
