"""
src/experiments/run_trilens_t7_with_predictions.py
====================================================
Re-runs ONLY the T7 ablation (TriLens-MoE-Stack v10 on `phylogeny_split`) and
additionally saves per-sample test predictions to `predictions_saved.npz`,
in the same v9.1-compatible format that `run_trilens_v10.py` already uses
for T6 (see `save_all_outputs()` in that file, lines ~1113-1127). That
function only ever builds this file from `all_results["T6"]` — T7's
per-sample predictions are silently dropped, which is why
`results/pillar_dedup_eval/SUMMARY.md` states the phylogeny_split dedup
check could not be run ("T7's raw per-sample predictions were never saved
to disk").

This script does NOT modify run_trilens_v10.py. It imports and reuses its
data-loading / feature-building / stacking functions directly
(`load_all_data`, `build_expert_a_blocks`, `build_expert_c_features`,
`run_kfold_and_stack`) and only adds the missing npz-saving step for T7 —
mirroring the pattern already used by `run_pillar_cascadestack_phylogeny.py`
and `run_tabnet_baseline_phylogeny.py` elsewhere in this project.

REVISED (this version) — Expert A/C preprocessing is now fit on
phylogeny_split's OWN train_idx, not random_split's.

An earlier version of this script (and, it turns out, `run_trilens_v10.py`'s
own `run()` for its T7 ablation — see below) built Expert A's per-gene SNP
column selection and Expert C's gene-aggregated feature standardization
using `train_idx` from `cfg.split_key` (= "random_split", the config's
default), reasoning at the time that this was "split-agnostic
infrastructure." That reasoning was wrong: Expert A's column selection (for
genes with >L_max=64 SNPs, top-64-by-variance columns are chosen from
`X_snp[train_idx, ...]`) and Expert C's z-score standardization
(`feat_mean`/`feat_std` from `X_raw[train_idx]`) are both **unsupervised
but still real train-set statistics** -- computing them on random_split's
train and then evaluating T7 on phylogeny_split means a portion of
phylogeny_split's TEST genomes (confirmed elsewhere in this project: 73.1%
overlap between random_split-train and phylogeny_split-test) influenced
which SNP columns exist and how features are scaled for the "held-out"
evaluation.

**Confirmed this is not hypothetical**: the actual production run that
produced the currently-locked `results/trilens_v10/ablation_results.csv`
(T6 0.8462, T7 0.7493) logged, verbatim
(`experiments/2026-06-05_16-11-31_trilens_v10/run_experiment.log`):

    Loading cached Expert A blocks from .../results/trilens_v10/cache
    Loaded Expert C features from cache: .../gene_features_5000x4.npy

and the earlier run that originally built that cache
(`experiments/2026-06-05_11-30-29_trilens_v10/run_experiment.log`) built it
BEFORE any ablation (T3/T6/T7) ran, using the single outer `train_idx` from
`cfg.split_key` = "random_split". So the existing T7 = 0.7493 number carries
this caveat on top of B1 (phylogeny_split is a stratified split, not a
holdout). This does NOT mean T7 fell into the CascadeStack-style bug of
using the *wrong row-level split entirely* -- `run_trilens_v10.py`'s
`load_all_data()` reads `phylogeny_split` directly from the raw pkl (not
through the buggy `build_gene_feature_matrix_from_pkl()`), and both log
files confirm "K-fold OOF (phylogeny_split)" for T7's actual train/val/test
partitioning and stacking. Only the Expert A/C *preprocessing* statistics
were wrong-split.

This script fixes that: Expert A and Expert C are built using
phylogeny_split's own train_idx, into this script's own fresh, isolated
`cache/` directory (never `results/trilens_v10/cache/`), so there is no risk
of silently reusing the random_split-derived cache either.

Output: results/pillar_trilens_t7_predictions/
    predictions_saved.npz   (test_ens_probs, test_base_probs, oof_base_probs,
                              val_ens_probs, test_indices, val_indices,
                              train_indices, base_names)
    t7_results.csv           (per-antibiotic PR-AUC, MCC -- expect this to
                              differ somewhat from the existing T7 row in
                              results/trilens_v10/ablation_results.csv,
                              precisely because the Expert A/C leak above is
                              now fixed; that difference is expected and is
                              the point of this rerun, not a discrepancy to
                              be alarmed by)
    config_used.yaml

NOT run by this session (GPU job, same cost as the original T7 ablation
inside run_trilens_v10.py — full K-fold OOF + 3-lens MoE + meta-stack on
phylogeny_split, ~40-60 min) -- write-only, per instructions.

Usage:
    conda activate amr
    python -m src.experiments.run_trilens_t7_with_predictions

After this completes, `run_pillar_dedup_eval.py`'s `compute_dedup_eval`
function can be pointed at this script's `predictions_saved.npz` (instead of
`results/trilens_v10/predictions_saved.npz`) to get the phylogeny_split +
deduplicated PR-AUC number.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.experiments.run_trilens_v10 import (
    AB_NAMES,
    build_expert_a_blocks,
    build_expert_c_features,
    load_all_data,
    run_kfold_and_stack,
)
from src.utils.seeds import set_global_seed

ROOT = Path(__file__).resolve().parents[2]
CONFIG_YAML_PATH = ROOT / "configs" / "config.yaml"
TRILENS_CFG_PATH = ROOT / "configs" / "experiment" / "trilens_v10.yaml"
OUTPUT_DIR = ROOT / "results" / "pillar_trilens_t7_predictions"
SEED = 42  # identical to the original T6/T7 headline seed


def _build_cfg():
    base_cfg = OmegaConf.load(CONFIG_YAML_PATH)
    if "hydra" in base_cfg:
        del base_cfg["hydra"]
    if "wandb" in base_cfg:
        del base_cfg["wandb"]
    exp_cfg = OmegaConf.load(TRILENS_CFG_PATH)
    exp_cfg.output_dir = str(OUTPUT_DIR)

    cfg = OmegaConf.merge(base_cfg, {"experiment": exp_cfg})
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
    # run_trilens_v10 functions expect cfg == cfg.experiment (see run()'s own
    # unwrapping logic at the top of that file).
    return cfg.experiment


def main() -> None:
    cfg = _build_cfg()
    set_global_seed(SEED)

    output_dir = Path(str(cfg.output_dir))
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    OmegaConf.save(cfg, output_dir / "config_used.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    data = load_all_data(cfg)

    # FIX (see module docstring): Expert A / C preprocessing must be fit on
    # phylogeny_split's OWN train_idx, not random_split's -- using the wrong
    # split's train partition here is exactly the leak confirmed in the
    # existing T7 = 0.7493 result. cache_dir is this script's own isolated
    # directory (results/pillar_trilens_t7_predictions/cache/), never
    # results/trilens_v10/cache/, so there is no risk of silently loading
    # the random_split-derived cache either.
    from src.experiments.run_trilens_v10 import _get_split_indices
    train_idx_phylo, _, _ = _get_split_indices(data, "phylogeny_split")

    print("Building Expert A blocks (fit on phylogeny_split train_idx)...")
    X_blocks, M_blocks, col_indices = build_expert_a_blocks(
        data, cfg, cache_dir, train_idx_phylo
    )

    print("Building Expert C features (fit on phylogeny_split train_idx)...")
    X_gene_agg, feat_stats = build_expert_c_features(data, train_idx_phylo, cache_dir)

    print("\nRunning T7 (K-fold OOF + MoE stack) on phylogeny_split...")
    t0 = time.time()
    result = run_kfold_and_stack(
        "T7", cfg, data, X_blocks, M_blocks, X_gene_agg,
        "phylogeny_split", SEED, device,
    )
    elapsed_min = (time.time() - t0) / 60
    print(f"T7 done in {elapsed_min:.1f} min")

    metrics = result["metrics"]
    rows = []
    for ab in AB_NAMES:
        m = metrics.get(ab, {})
        rows.append({
            "antibiotic": ab,
            "pr_auc": m.get("prauc"),
            "mcc": m.get("mcc"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "t7_results.csv", index=False)
    print(df)
    print(f"Average PR-AUC: {df['pr_auc'].mean():.4f}")
    print("(Compare against the T7 row in results/trilens_v10/ablation_results.csv "
          "as a sanity check -- should match closely, since this reruns the "
          "identical pipeline/seed, just with predictions additionally saved.)")

    np.savez(
        str(output_dir / "predictions_saved.npz"),
        test_ens_probs=result["test_probs"],
        test_base_probs=result["test_base_probs"],
        oof_base_probs=result["oof_base_probs"],
        val_ens_probs=result.get("val_probs", np.zeros((0, len(AB_NAMES)))),
        test_indices=result["test_idx"],
        val_indices=result["val_idx"],
        train_indices=result["train_idx"],
        base_names=np.array(result.get("base_names", ["TriLens"])),
    )
    print(f"\nSaved predictions_saved.npz -> {output_dir}")


if __name__ == "__main__":
    main()
