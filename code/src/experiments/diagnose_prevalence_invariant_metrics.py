"""
src/experiments/diagnose_prevalence_invariant_metrics.py
===========================================================
CPU-only, no retraining. Extends the split comparison with two
prevalence-invariant metrics (ROC-AUC, MCC) and the raw test-set prevalence
per antibiotic, for the two arms whose full-stack test predictions already
exist on disk:

  * random_split (T6)              -- results/trilens_v10/predictions_saved.npz
  * phylogroup-stratified (clean)  -- recomputed by re-running the same,
                                       unmodified run_meta_stack() call that
                                       diagnose_t7_clean_final_comparison.py
                                       used (deterministic: identical seed=42
                                       LogisticRegression on identical saved
                                       base predictions reproduces the exact
                                       same test_meta array -- this is a
                                       reproduction, not a new experiment)

The MLST lineage holdout arm is intentionally NOT included here -- its
predictions do not exist yet (Step 3/4/5 of that experiment are still
queued). The identical metric computation has been added directly into
diagnose_mlst_final_comparison.py so it is produced automatically as part
of that step's own output once it runs.

Threshold convention: MCC uses a FIXED 0.5 threshold on the probability,
matching run_trilens_v10.py's own compute_metrics() convention (used for
every T6/T7 ablation's MCC in this project) -- not a per-split val-tuned
threshold, so the three arms are compared on identical footing. Stated
explicitly here rather than left implicit.

Does not modify run_trilens_v10.py or any other existing file (other than
the forward-looking addition to diagnose_mlst_final_comparison.py, which had
not yet run).

Usage:
    conda activate amr
    python -m src.experiments.diagnose_prevalence_invariant_metrics
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from src.experiments.run_trilens_v10 import run_meta_stack
from src.models.trilens import AB_NAMES

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
PROCESSED_DIR = ROOT / "data" / "processed"

# The main TriLens run only; no fallback to the pre-fix trilens_v10.
T6_NPZ = RESULTS_DIR / os.environ.get("TRILENS_MAIN_RUN", "trilens_attnfix") / "predictions_saved.npz"
MASTER_PKL = PROCESSED_DIR / "arcadia_v2_master_features.pkl"

PHYLO_CLEAN_PKL = PROCESSED_DIR / "arcadia_v2_gene_filtered__phylogeny_clean.pkl"
PHYLO_T7_NPZ = RESULTS_DIR / "pillar_trilens_t7_clean_gene_set" / "predictions_saved.npz"
PHYLO_CLASSICAL_NPZ = RESULTS_DIR / "pillar_cascadestack_classical_phylogeny_clean" / "predictions_saved.npz"

CONFIG_YAML_PATH = ROOT / "configs" / "config.yaml"
TRILENS_CFG_PATH = ROOT / "configs" / "experiment" / "trilens_v10.yaml"

OUTPUT_DIR = RESULTS_DIR / "pillar_prevalence_invariant_metrics"
THRESHOLD = 0.5


def _load_cfg():
    base_cfg = OmegaConf.load(CONFIG_YAML_PATH)
    if "hydra" in base_cfg:
        del base_cfg["hydra"]
    if "wandb" in base_cfg:
        del base_cfg["wandb"]
    exp_cfg = OmegaConf.load(TRILENS_CFG_PATH)
    cfg = OmegaConf.merge(base_cfg, {"experiment": exp_cfg})
    return OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    ).experiment


def compute_arm_metrics(test_probs: np.ndarray, test_idx: np.ndarray,
                        Y: np.ndarray, mask: np.ndarray) -> list[dict]:
    """One row per antibiotic: PR-AUC, ROC-AUC, MCC (threshold=0.5), and the
    actual test-set resistance prevalence."""
    Yt, Mt = Y[test_idx], mask[test_idx]
    rows = []
    for j, ab in enumerate(AB_NAMES):
        valid = Mt[:, j].astype(bool)
        y = Yt[valid, j].astype(int)
        p = test_probs[valid, j]
        pred = (p >= THRESHOLD).astype(int)
        rows.append({
            "antibiotic": ab,
            "n_test": int(valid.sum()),
            "prevalence_pct": round(100 * float(y.mean()), 2),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "roc_auc": (round(float(roc_auc_score(y, p)), 4)
                       if len(np.unique(y)) > 1 else float("nan")),
            "mcc": (round(float(matthews_corrcoef(y, pred)), 4)
                   if len(np.unique(pred)) > 1 else 0.0),
        })
    return rows


def random_split_arm() -> list[dict]:
    if not T6_NPZ.exists():
        raise FileNotFoundError(f"{T6_NPZ} missing")
    d = np.load(T6_NPZ, allow_pickle=True)
    with open(MASTER_PKL, "rb") as f:
        m = pickle.load(f)
    Y = m["Y"].astype(int)
    mask = m["label_mask"].astype(bool)
    return compute_arm_metrics(d["test_ens_probs"], d["test_indices"], Y, mask)


def phylogroup_clean_arm(cfg) -> list[dict]:
    for p in (PHYLO_T7_NPZ, PHYLO_CLASSICAL_NPZ, PHYLO_CLEAN_PKL):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing")
    with open(PHYLO_CLEAN_PKL, "rb") as f:
        gf = pickle.load(f)
    Y = gf["Y"].astype(np.float64)
    mask = gf["label_mask"].astype(bool)

    t7 = np.load(PHYLO_T7_NPZ, allow_pickle=True)
    cs = np.load(PHYLO_CLASSICAL_NPZ, allow_pickle=True)
    assert np.array_equal(t7["train_indices"], cs["train_indices"])
    assert np.array_equal(t7["test_indices"], cs["test_indices"])

    train_idx, test_idx = t7["train_indices"], t7["test_indices"]
    test_meta, _ = run_meta_stack(
        oof_trilens=t7["oof_base_probs"][:, -1, :],
        test_trilens=t7["test_base_probs"][:, -1, :],
        oof_classical=cs["oof_base_probs"], test_classical=cs["test_base_probs"],
        Y_tr=Y[train_idx], mask_tr=mask[train_idx], cfg=cfg,
    )
    return compute_arm_metrics(test_meta, test_idx, Y.astype(int), mask)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg()

    print("Computing random_split (T6) metrics from saved predictions...")
    rs_rows = random_split_arm()

    print("Recomputing phylogroup-stratified (clean) test_meta "
          "(deterministic reproduction of the already-reported 0.8519 run)...")
    pc_rows = phylogroup_clean_arm(cfg)

    rs_avg_pr = np.mean([r["pr_auc"] for r in rs_rows])
    print(f"  sanity check -- random_split avg PR-AUC recomputed here: "
          f"{rs_avg_pr:.4f} (locked reference: 0.8462)")
    pc_avg_pr = np.mean([r["pr_auc"] for r in pc_rows])
    print(f"  sanity check -- phylogroup-clean avg PR-AUC recomputed here: "
          f"{pc_avg_pr:.4f} (previously reported: 0.8519)")

    combined = []
    for r in rs_rows:
        combined.append({"split": "random_split_T6", **r})
    for r in pc_rows:
        combined.append({"split": "phylogroup_stratified_clean", **r})
    df = pd.DataFrame(combined)
    df.to_csv(OUTPUT_DIR / "prevalence_invariant_metrics.csv", index=False)

    print("\n" + "=" * 100)
    print(f"PER-ANTIBIOTIC: PR-AUC / ROC-AUC / MCC / test prevalence  (threshold={THRESHOLD})")
    print("=" * 100)
    for split_name, rows in [("random_split (T6)", rs_rows),
                             ("phylogroup-stratified (clean)", pc_rows)]:
        print(f"\n-- {split_name} --")
        print(f"  {'AB':<5}{'prev%':>8}{'PR-AUC':>9}{'ROC-AUC':>9}{'MCC':>8}")
        for r in rows:
            print(f"  {r['antibiotic']:<5}{r['prevalence_pct']:>8.1f}"
                  f"{r['pr_auc']:>9.4f}{r['roc_auc']:>9.4f}{r['mcc']:>8.4f}")
        print(f"  {'AVG':<5}{'':>8}{np.mean([r['pr_auc'] for r in rows]):>9.4f}"
              f"{np.mean([r['roc_auc'] for r in rows]):>9.4f}"
              f"{np.mean([r['mcc'] for r in rows]):>8.4f}")

    print(f"\nSaved -> {OUTPUT_DIR / 'prevalence_invariant_metrics.csv'}")
    print("\nMLST lineage holdout arm NOT included -- its predictions do not exist "
          "yet. The identical computation has been added into "
          "diagnose_mlst_final_comparison.py, so it will appear automatically "
          "once that step (currently queued) runs.")


if __name__ == "__main__":
    main()
