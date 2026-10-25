"""
src/experiments/diagnose_t7_clean_final_comparison.py
========================================================
CPU-only, no retraining. Final step of the Stage-1.5-clean-rerun batch:
combines TriLens T7 (retrained on the clean, phylogeny_split-ranked gene
set -- run_trilens_t7_clean_gene_set.py) with the 5 classical bases
(retrained on the SAME clean gene set -- run_cascadestack_classical_
phylogeny_clean.py), properly aligned by genome index (verified via
assertion, not assumed), fit on OOF/train only, predicted on test only
(same run_meta_stack() function, imported unchanged, as used in
diagnose_t7_classical_alignment.py for the leaked-gene-set version).

Produces the final 3-way comparison table:
    Random split (T6)              | Phylogeny (leaked gene set)     | Phylogeny (clean gene set)
    0.8462                          | 0.8512                           | ?

Does not modify any existing file.

Usage:
    conda activate amr
    python -m src.experiments.diagnose_t7_clean_final_comparison
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score

from src.experiments.run_trilens_v10 import run_meta_stack
from src.models.trilens import AB_NAMES

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
CLEAN_GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered__phylogeny_clean.pkl"
CONFIG_YAML_PATH = ROOT / "configs" / "config.yaml"
TRILENS_CFG_PATH = ROOT / "configs" / "experiment" / "trilens_v10.yaml"

T7_CLEAN_NPZ = RESULTS_DIR / "pillar_trilens_t7_clean_gene_set" / "predictions_saved.npz"
CLASSICAL_CLEAN_NPZ = RESULTS_DIR / "pillar_cascadestack_classical_phylogeny_clean" / "predictions_saved.npz"
OUTPUT_DIR = RESULTS_DIR / "pillar_t7_clean_final_comparison"

T6_RANDOM_SPLIT = {"AMP": None, "CIP": None, "AMC": None, "SXT": None,
                   "GEN": None, "CTX": None, "CAZ": None}  # filled from ablation_results.csv below
T6_AVG_LOCKED = 0.8462
ALIGNED_T7_LEAKED_GENE_SET = {"AMP": 0.9338, "CIP": 0.9600, "AMC": 0.7957, "SXT": 0.8295,
                              "GEN": 0.7108, "CTX": 0.8818, "CAZ": 0.8466}
ALIGNED_T7_LEAKED_AVG = 0.8512


def _load_cfg():
    base_cfg = OmegaConf.load(CONFIG_YAML_PATH)
    if "hydra" in base_cfg:
        del base_cfg["hydra"]
    if "wandb" in base_cfg:
        del base_cfg["wandb"]
    exp_cfg = OmegaConf.load(TRILENS_CFG_PATH)
    cfg = OmegaConf.merge(base_cfg, {"experiment": exp_cfg})
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
    return cfg.experiment


def _load_labels_clean():
    with open(CLEAN_GENE_FILTERED_PKL, "rb") as f:
        d = pickle.load(f)
    return d["Y"].astype(np.float64), d["label_mask"].astype(bool)


def _load_t6_from_ablation_results():
    # Prefer the delivered strain-conditioned run (trilens_attnfix, AMP 0.9440).
    # trilens_v10 is the same model BEFORE the attention fix (AMP 0.9408) and is
    # only a fallback. Reading it first put pre-fix numbers in the random column.
    p = next((q for q in (RESULTS_DIR / "trilens_attnfix" / "ablation_results.csv",
                          RESULTS_DIR / "trilens_v10" / "ablation_results.csv") if q.exists()),
             None)
    if p is None:
        return None
    print(f"  random-split T6 column read from: {p.parent.name}")
    df = pd.read_csv(p)
    t6 = df[(df["ablation"] == "T6") & (df["split_key"] == "random_split")]
    return dict(zip(t6["ab"], t6["prauc"]))


def main() -> None:
    for p, label in [(T7_CLEAN_NPZ, "T7 clean gene set"), (CLASSICAL_CLEAN_NPZ, "classical clean gene set")]:
        if not p.exists():
            raise FileNotFoundError(f"{p} ({label}) not found -- run the preceding stage first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg()
    Y, mask = _load_labels_clean()

    t7 = np.load(T7_CLEAN_NPZ, allow_pickle=True)
    cs = np.load(CLASSICAL_CLEAN_NPZ, allow_pickle=True)

    assert np.array_equal(t7["train_indices"], cs["train_indices"]), (
        "train_indices mismatch between T7-clean and classical-clean -- "
        "do NOT combine positionally, alignment assumption is false."
    )
    assert np.array_equal(t7["test_indices"], cs["test_indices"]), (
        "test_indices mismatch between T7-clean and classical-clean -- "
        "do NOT combine positionally, alignment assumption is false."
    )
    print("Alignment verified: train_indices and test_indices are identical "
          "(content and order) between T7-clean and classical-clean.")

    train_idx, test_idx = t7["train_indices"], t7["test_indices"]
    oof_trilens = t7["oof_base_probs"][:, -1, :]   # TriLens is always the last saved base
    test_trilens = t7["test_base_probs"][:, -1, :]
    oof_classical = cs["oof_base_probs"]    # already 5-base only (no GAN column)
    test_classical = cs["test_base_probs"]

    test_meta, meta_weights = run_meta_stack(
        oof_trilens=oof_trilens, test_trilens=test_trilens,
        oof_classical=oof_classical, test_classical=test_classical,
        Y_tr=Y[train_idx], mask_tr=mask[train_idx], cfg=cfg,
    )

    Yt, Mt = Y[test_idx], mask[test_idx]
    clean_t7_per_ab = {}
    for ab_idx, ab in enumerate(AB_NAMES):
        valid = Mt[:, ab_idx].astype(bool)
        clean_t7_per_ab[ab] = float(average_precision_score(Yt[valid, ab_idx], test_meta[valid, ab_idx]))
    clean_t7_per_ab["_avg"] = float(np.mean([clean_t7_per_ab[ab] for ab in AB_NAMES]))
    print(f"\nProperly-aligned T7, clean gene set: avg PR-AUC = {clean_t7_per_ab['_avg']:.4f}")

    coef_rows = []
    for ab in AB_NAMES:
        w = meta_weights.get(ab)
        row = {"antibiotic": ab, "pr_auc": clean_t7_per_ab[ab]}
        if w is not None:
            row.update(dict(zip(w["bases"], [round(c, 3) for c in w["coef"]])))
        coef_rows.append(row)
    pd.DataFrame(coef_rows).to_csv(OUTPUT_DIR / "clean_t7_coefficients_and_prauc.csv", index=False)

    t6_per_ab = _load_t6_from_ablation_results()

    print("\n" + "=" * 70)
    print("THREE-WAY COMPARISON (no interpretation -- numbers only)")
    print("=" * 70)
    rows = []
    for ab in AB_NAMES:
        rows.append({
            "antibiotic": ab,
            "random_split_T6": t6_per_ab.get(ab) if t6_per_ab else None,
            "phylogeny_leaked_gene_set": ALIGNED_T7_LEAKED_GENE_SET.get(ab),
            "phylogeny_clean_gene_set": round(clean_t7_per_ab[ab], 4),
        })
    df_final = pd.DataFrame(rows)
    avg_row = {
        "antibiotic": "AVERAGE",
        "random_split_T6": float(np.mean(list(t6_per_ab.values()))) if t6_per_ab else T6_AVG_LOCKED,
        "phylogeny_leaked_gene_set": ALIGNED_T7_LEAKED_AVG,
        "phylogeny_clean_gene_set": round(clean_t7_per_ab["_avg"], 4),
    }
    df_final = pd.concat([df_final, pd.DataFrame([avg_row])], ignore_index=True)
    df_final.to_csv(OUTPUT_DIR / "three_way_comparison_FINAL.csv", index=False)
    print(df_final.to_string(index=False))
    print(f"\nSaved all outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
