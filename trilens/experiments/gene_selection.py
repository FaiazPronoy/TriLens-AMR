from __future__ import annotations

import time

from omegaconf import DictConfig

from trilens.experiments.gene_filtered_dataset import run as run_build_dataset
from trilens.experiments.gene_importance_ranking import run as run_xgb_importance


def run(cfg: DictConfig) -> None:
    print("\n" + "=" * 80)
    print("  Gene selection")
    print("=" * 80)
    print("  Step 1: XGBoost gene importance computation")
    print("  Step 2: Gene filtering + filtered.pkl build")
    print("=" * 80)

    total_start = time.time()

    print("\n" + "▼" * 80)
    print("  STEP 1/2 - XGBoost gene importance per antibiotic")
    print("▼" * 80)
    step1_start = time.time()

    try:
        run_xgb_importance(cfg)
    except Exception as e:
        print(f"\n  Step 1 failed: {type(e).__name__}: {e}")
        raise

    step1_time = time.time() - step1_start
    print(f"\n  Step 1 complete in {step1_time / 60:.1f} min")

    print("\n" + "▼" * 80)
    print("  STEP 2/2 - Gene filtering + filtered.pkl build")
    print("▼" * 80)
    step2_start = time.time()

    try:
        run_build_dataset(cfg)
    except Exception as e:
        print(f"\n  Step 2 failed: {type(e).__name__}: {e}")
        raise

    step2_time = time.time() - step2_start
    print(f"\n  Step 2 complete in {step2_time / 60:.1f} min")

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print("  Gene selection complete")
    print("=" * 80)
    print(f"  Total runtime: {total_time / 60:.1f} min ({total_time / 3600:.2f} hours)")
    print(f"  Step 1 (XGB): {step1_time / 60:.1f} min")
    print(f"  Step 2 (Build): {step2_time / 60:.1f} min")
    print(f"  Output → {cfg.experiment.output_pkl}")
    print(f"  Summary → {cfg.experiment.output_dir}/SUMMARY.md")
    print("=" * 80)
