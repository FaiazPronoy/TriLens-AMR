"""Stage 1.5 — Combined runner.

Executes:
1. XGBoost gene importance computation per antibiotic
2. Gene filtering + final dataset assembly

Single command:
    python -m src.run_experiment experiment=stage1_5_gene_filtering

Runtime estimate: ~5-8 hours total on RTX 4080 Super.
"""
from __future__ import annotations

import time
from omegaconf import DictConfig

from src.experiments.run_xgb_gene_importance import run as run_xgb_importance
from src.experiments.build_gene_filtered_dataset import run as run_build_dataset


def run(cfg: DictConfig) -> None:
    """Sequentially execute XGB importance computation + dataset build."""
    print("\n" + "=" * 80)
    print("  STAGE 1.5 — COMBINED RUNNER")
    print("=" * 80)
    print(f"  Step 1: XGBoost gene importance computation")
    print(f"  Step 2: Gene filtering + filtered.pkl build")
    print("=" * 80)
    
    total_start = time.time()
    
    # ============================================================
    # Step 1: XGBoost gene importance per antibiotic
    # ============================================================
    print("\n" + "▼" * 80)
    print("  STEP 1/2 — XGBoost gene importance per antibiotic")
    print("▼" * 80)
    step1_start = time.time()
    
    try:
        run_xgb_importance(cfg)
    except Exception as e:
        print(f"\n  ❌ STEP 1 FAILED: {type(e).__name__}: {e}")
        raise
    
    step1_time = time.time() - step1_start
    print(f"\n  ✅ STEP 1 complete in {step1_time / 60:.1f} min")
    
    # ============================================================
    # Step 2: Gene filtering + dataset build
    # ============================================================
    print("\n" + "▼" * 80)
    print("  STEP 2/2 — Gene filtering + filtered.pkl build")
    print("▼" * 80)
    step2_start = time.time()
    
    try:
        run_build_dataset(cfg)
    except Exception as e:
        print(f"\n  ❌ STEP 2 FAILED: {type(e).__name__}: {e}")
        raise
    
    step2_time = time.time() - step2_start
    print(f"\n  ✅ STEP 2 complete in {step2_time / 60:.1f} min")
    
    # ============================================================
    # Final summary
    # ============================================================
    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"  STAGE 1.5 FULLY COMPLETE")
    print("=" * 80)
    print(f"  Total runtime: {total_time / 60:.1f} min ({total_time / 3600:.2f} hours)")
    print(f"  Step 1 (XGB): {step1_time / 60:.1f} min")
    print(f"  Step 2 (Build): {step2_time / 60:.1f} min")
    print(f"  Output → {cfg.experiment.output_pkl}")
    print(f"  Summary → {cfg.experiment.output_dir}/SUMMARY.md")
    print(f"  Ready for Pillar 2 architecture build ✅")
    print("=" * 80)