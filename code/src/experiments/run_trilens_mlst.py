"""
src/experiments/run_trilens_mlst.py
=====================================
STEP 4: TriLens (the v10 neural stack) on the MLST lineage holdout split,
using the MLST-ranked clean gene set.

Two things this script must get right, both of which would otherwise be
silent or fatal bugs:

1. `run_trilens_v10.load_all_data()` only passes `random_split` and
   `phylogeny_split` through into its data dict -- an arbitrary split key in
   the pkl is simply dropped. So after loading, this script INJECTS the MLST
   split into the returned dict under `mlst_lineage_holdout_split`;
   `_get_split_indices(data, split_key)` then resolves it normally. No
   modification to run_trilens_v10.py is needed.

2. `run_kfold_and_stack()` unconditionally tries to load 5 classical base
   predictions from `cfg.v9_predictions`
   (results/stage3_cascadestack_v9/predictions_saved.npz). That file was
   produced on `random_split`, whose train partition has 4,872 rows -- the
   MLST holdout train has 4,836. Boolean-masking a 4,872-row array with a
   4,836-long mask raises IndexError, so the run would die partway through.
   Even if the lengths had matched it would be the wrong genomes (this is
   exactly the row-position misalignment already found earlier in this
   session). `cfg.v9_predictions` is therefore pointed at a deliberately
   non-existent path; run_kfold_and_stack handles that by logging a warning
   and stacking TriLens alone. That is what we want here -- this stage
   produces TriLens SOLO, and the properly index-aligned 5-classical +
   TriLens meta-stack is built afterwards by
   diagnose_mlst_final_comparison.py.

Expert A / Expert C preprocessing is fit on this split's OWN train_idx, into
this script's own isolated cache directory (never a shared one).

Does not modify any existing file.
Output: results/pillar_trilens_mlst/

Usage:
    conda activate amr
    python -m src.experiments.run_trilens_mlst
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.experiments.run_trilens_v10 import (
    AB_NAMES,
    _get_split_indices,
    build_expert_a_blocks,
    build_expert_c_features,
    load_all_data,
    run_kfold_and_stack,
)
from src.utils.seeds import set_global_seed

ROOT = Path(__file__).resolve().parents[2]
CONFIG_YAML_PATH = ROOT / "configs" / "config.yaml"
TRILENS_CFG_PATH = ROOT / "configs" / "experiment" / "trilens_v10.yaml"
GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered__mlst_holdout.pkl"
OUTPUT_DIR = ROOT / "results" / "pillar_trilens_mlst"
SPLIT_KEY = "mlst_lineage_holdout_split"
SEED = 42


def _build_cfg():
    base_cfg = OmegaConf.load(CONFIG_YAML_PATH)
    if "hydra" in base_cfg:
        del base_cfg["hydra"]
    if "wandb" in base_cfg:
        del base_cfg["wandb"]
    exp_cfg = OmegaConf.load(TRILENS_CFG_PATH)
    exp_cfg.output_dir = str(OUTPUT_DIR)
    exp_cfg.gene_filtered_pkl = str(GENE_FILTERED_PKL)
    # See docstring point 2: must NOT load the random_split classical file.
    exp_cfg.v9_predictions = str(
        OUTPUT_DIR / "_intentionally_absent_v9_predictions.npz"
    )
    cfg = OmegaConf.merge(base_cfg, {"experiment": exp_cfg})
    cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    )
    return cfg.experiment


def main() -> None:
    if not GENE_FILTERED_PKL.exists():
        raise FileNotFoundError(
            f"{GENE_FILTERED_PKL} not found -- run "
            f"src.experiments.build_gene_filtered_dataset_mlst first."
        )

    cfg = _build_cfg()
    set_global_seed(SEED)

    output_dir = Path(str(cfg.output_dir))
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config_used.yaml")

    assert not Path(str(cfg.v9_predictions)).exists(), (
        "v9_predictions path must NOT exist -- it would load random_split "
        "classical predictions with a mismatched train length."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data (MLST-holdout clean gene set)...")
    data = load_all_data(cfg)

    # Inject the MLST split -- load_all_data drops unknown split keys.
    with open(GENE_FILTERED_PKL, "rb") as f:
        gf = pickle.load(f)
    data[SPLIT_KEY] = gf[SPLIT_KEY]
    st = np.asarray(gf["st_per_genome"], dtype=object)

    train_idx, val_idx, test_idx = _get_split_indices(data, SPLIT_KEY)
    print(f"  split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Re-verify the holdout property right where the model will consume it.
    st_tr = {st[i] for i in train_idx if st[i] is not None}
    st_va = {st[i] for i in val_idx if st[i] is not None}
    st_te = {st[i] for i in test_idx if st[i] is not None}
    assert not (st_tr & st_te), "TRAIN/TEST ST LEAK"
    assert not (st_tr & st_va), "TRAIN/VAL ST LEAK"
    assert not (st_va & st_te), "VAL/TEST ST LEAK"
    print(f"  ST disjointness re-verified "
          f"(train {len(st_tr)} STs, val {len(st_va)}, test {len(st_te)})")

    print("Building Expert A blocks (fit on MLST-holdout train_idx)...")
    X_blocks, M_blocks, col_indices = build_expert_a_blocks(
        data, cfg, cache_dir, train_idx, split_name=SPLIT_KEY
    )

    print("Building Expert C features (fit on MLST-holdout train_idx)...")
    X_gene_agg, feat_stats = build_expert_c_features(
        data, train_idx, cache_dir, split_name=SPLIT_KEY
    )

    print("\nRunning TriLens (K-fold OOF + MoE stack) on the MLST lineage holdout...")
    t0 = time.time()
    result = run_kfold_and_stack(
        "T7", cfg, data, X_blocks, M_blocks, X_gene_agg,
        SPLIT_KEY, SEED, device,
        fold_dir=output_dir / "folds",
    )
    print(f"TriLens (MLST holdout) done in {(time.time() - t0)/60:.1f} min")

    metrics = result["metrics"]
    rows = [{"antibiotic": ab,
             "pr_auc": metrics.get(ab, {}).get("prauc"),
             "mcc": metrics.get(ab, {}).get("mcc")} for ab in AB_NAMES]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "trilens_mlst_results.csv", index=False)
    print(df)
    print(f"TriLens SOLO average PR-AUC: {df['pr_auc'].mean():.4f}")

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
    print("STEP 4 COMPLETE")


if __name__ == "__main__":
    main()
