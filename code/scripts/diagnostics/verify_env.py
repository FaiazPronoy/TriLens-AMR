"""Check that the Python environment, GPU, data files and loaders all work.

Run from inside code/:
    python scripts/diagnostics/verify_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the code/ folder
sys.path.insert(0, str(ROOT))

print("=" * 70)
print("AMR PROJECT — ENVIRONMENT VERIFICATION")
print("=" * 70)

# Check imports
print("\n[1/6] Checking imports...")
try:
    import torch
    import lightning as L
    import wandb
    import hydra
    from omegaconf import OmegaConf
    import xgboost
    import lightgbm
    import sklearn
    import pandas as pd
    import numpy as np
    import shap
    #import cyvcf2
    print("  All imports OK")
except ImportError as e:
    print(f"  IMPORT FAILED: {e}")
    sys.exit(1)

# Check GPU
print("\n[2/6] Checking GPU...")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    x = torch.randn(5000, 5000).cuda()
    y = x @ x.T
    print(f"  GPU compute OK: {y.shape}")
    del x, y
    torch.cuda.empty_cache()
else:
    print("  GPU NOT AVAILABLE — investigate driver")
    sys.exit(1)

# Check data files
print("\n[3/6] Checking data files...")
data_files = [
    ROOT / "data" / "processed" / "arcadia_v2_master_features.pkl",
    ROOT / "data" / "processed" / "arcadia_v2_snp_matrix.pkl",
    ROOT / "data" / "Giessen_dataset" / "cip_ctx_ctz_gen_multi_data.csv",
    ROOT / "data" / "Giessen_dataset" / "cip_ctx_ctz_gen_pheno.csv",
]
for f in data_files:
    if f.exists():
        size_mb = f.stat().st_size / 1e6
        print(f"  OK: {f.name} ({size_mb:.1f} MB)")
    else:
        print(f"  MISSING: {f}")
        sys.exit(1)

# Test data loading
print("\n[4/6] Testing Arcadia DataModule...")
try:
    from src.data.datamodule import ArcadiaDataModule
    cfg = OmegaConf.load(ROOT / "configs" / "data" / "arcadia_v2.yaml")
    # Resolve project_root manually for standalone test
    cfg.paths.master_pkl = str(ROOT / "data" / "processed" / "arcadia_v2_master_features.pkl")
    cfg.paths.snp_matrix_pkl = str(ROOT / "data" / "processed" / "arcadia_v2_snp_matrix.pkl")
    
    dm = ArcadiaDataModule(cfg, feature_type="pca", batch_size=32, num_workers=0)
    dm.prepare_data()
    dm.setup()
    print(f"  X shape: {dm.X.shape}")
    print(f"  Y shape: {dm.Y.shape}")
    print(f"  Train/Val/Test: {len(dm.train_idx)}/{len(dm.val_idx)}/{len(dm.test_idx)}")
    print(f"  Antibiotics: {dm.label_names}")
    
    batch = next(iter(dm.train_dataloader()))
    print(f"  Batch X: {batch['X'].shape}, Y: {batch['Y'].shape}")
    print("  Arcadia DataModule OK")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n[5/6] Testing Giessen DataModule...")
try:
    from src.data.datamodule import GiessenDataModule
    cfg_g = OmegaConf.load(ROOT / "configs" / "data" / "giessen.yaml")
    cfg_g.paths.snp_csv = str(ROOT / "data" / "Giessen_dataset" / "cip_ctx_ctz_gen_multi_data.csv")
    cfg_g.paths.pheno_csv = str(ROOT / "data" / "Giessen_dataset" / "cip_ctx_ctz_gen_pheno.csv")
    
    dm_g = GiessenDataModule(cfg_g, batch_size=32, num_workers=0)
    dm_g.prepare_data()
    dm_g.setup()
    print(f"  X shape: {dm_g.X.shape}")
    print(f"  Y shape: {dm_g.Y.shape}")
    print(f"  Train/Val/Test: {len(dm_g.train_idx)}/{len(dm_g.val_idx)}/{len(dm_g.test_idx)}")
    print(f"  Antibiotics: {dm_g.label_names}")
    print("  Giessen DataModule OK")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Test metrics
print("\n[6/6] Testing metrics...")
try:
    from src.evaluation.metrics import compute_all_metrics
    from src.evaluation.statistical import bootstrap_ci, paired_wilcoxon
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 200)
    y_prob = rng.random(200)
    m = compute_all_metrics(y_true, y_prob)
    print(f"  Metrics computed: {list(m.keys())}")
    
    from sklearn.metrics import average_precision_score
    pt, lo, hi = bootstrap_ci(y_true, y_prob, average_precision_score, n_bootstrap=100)
    print(f"  Bootstrap CI: {pt:.3f} [{lo:.3f}, {hi:.3f}]")
    print("  Metrics + Statistics OK")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("STAGE 0 VERIFICATION COMPLETE — ALL CHECKS PASSED")
print("=" * 70)
print("\nNext: Stage 1 — Pillar 1 pre-tests (paste prompt to other agent)")