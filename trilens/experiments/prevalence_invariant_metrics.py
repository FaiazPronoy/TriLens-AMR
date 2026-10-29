from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from trilens.experiments.trilens_stack import run_meta_stack
from trilens.models.trilens_network import AB_NAMES

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
PROCESSED_DIR = ROOT / "data" / "processed"


FULL_STACK_NPZ = RESULTS_DIR / "trilens_stack_random" / "predictions_saved.npz"
MASTER_PKL = PROCESSED_DIR / "arcadia_v2_master_features.pkl"

PHYLO_CLEAN_PKL = PROCESSED_DIR / "arcadia_v2_gene_filtered__phylogeny_clean.pkl"
PHYLOGROUP_STACK_NPZ = RESULTS_DIR / "trilens_neural_phylogroup" / "predictions_saved.npz"
PHYLO_CLASSICAL_NPZ = RESULTS_DIR / "classical_stack_phylogroup" / "predictions_saved.npz"

CONFIG_YAML_PATH = ROOT / "configs" / "config.yaml"
TRILENS_CFG_PATH = ROOT / "configs" / "experiment" / "trilens_stack.yaml"

OUTPUT_DIR = RESULTS_DIR / "prevalence_invariant_metrics"
THRESHOLD = 0.5


def _load_cfg():
    base_cfg = OmegaConf.load(CONFIG_YAML_PATH)
    if "hydra" in base_cfg:
        del base_cfg["hydra"]
    if "wandb" in base_cfg:
        del base_cfg["wandb"]
    exp_cfg = OmegaConf.load(TRILENS_CFG_PATH)
    cfg = OmegaConf.merge(base_cfg, {"experiment": exp_cfg})
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)).experiment


def compute_arm_metrics(test_probs: np.ndarray, test_idx: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> list[dict]:

    Yt, Mt = Y[test_idx], mask[test_idx]
    rows = []
    for j, ab in enumerate(AB_NAMES):
        valid = Mt[:, j].astype(bool)
        y = Yt[valid, j].astype(int)
        p = test_probs[valid, j]
        pred = (p >= THRESHOLD).astype(int)
        rows.append(
            {
                "antibiotic": ab,
                "n_test": int(valid.sum()),
                "prevalence_pct": round(100 * float(y.mean()), 2),
                "pr_auc": round(float(average_precision_score(y, p)), 4),
                "roc_auc": (round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else float("nan")),
                "mcc": (round(float(matthews_corrcoef(y, pred)), 4) if len(np.unique(pred)) > 1 else 0.0),
            }
        )
    return rows


def random_split_arm() -> list[dict]:
    if not FULL_STACK_NPZ.exists():
        raise FileNotFoundError(f"{FULL_STACK_NPZ} missing")
    d = np.load(FULL_STACK_NPZ, allow_pickle=True)
    with open(MASTER_PKL, "rb") as f:
        m = pickle.load(f)
    Y = m["Y"].astype(int)
    mask = m["label_mask"].astype(bool)
    return compute_arm_metrics(d["test_ens_probs"], d["test_indices"], Y, mask)


def phylogroup_clean_arm(cfg) -> list[dict]:
    for p in (PHYLOGROUP_STACK_NPZ, PHYLO_CLASSICAL_NPZ, PHYLO_CLEAN_PKL):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing")
    with open(PHYLO_CLEAN_PKL, "rb") as f:
        gf = pickle.load(f)
    Y = gf["Y"].astype(np.float64)
    mask = gf["label_mask"].astype(bool)

    phylogroup_stack = np.load(PHYLOGROUP_STACK_NPZ, allow_pickle=True)
    cs = np.load(PHYLO_CLASSICAL_NPZ, allow_pickle=True)
    assert np.array_equal(phylogroup_stack["train_indices"], cs["train_indices"])
    assert np.array_equal(phylogroup_stack["test_indices"], cs["test_indices"])

    train_idx, test_idx = phylogroup_stack["train_indices"], phylogroup_stack["test_indices"]
    test_meta, _ = run_meta_stack(
        oof_trilens=phylogroup_stack["oof_base_probs"][:, -1, :],
        test_trilens=phylogroup_stack["test_base_probs"][:, -1, :],
        oof_classical=cs["oof_base_probs"],
        test_classical=cs["test_base_probs"],
        Y_tr=Y[train_idx],
        mask_tr=mask[train_idx],
        cfg=cfg,
    )
    return compute_arm_metrics(test_meta, test_idx, Y.astype(int), mask)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg()

    print("Computing random_split (full_stack) metrics from saved predictions...")
    rs_rows = random_split_arm()

    print(
        "Recomputing phylogroup-stratified (clean) test_meta "
        "(deterministic reproduction of the already-reported 0.8519 run)..."
    )
    pc_rows = phylogroup_clean_arm(cfg)

    rs_avg_pr = np.mean([r["pr_auc"] for r in rs_rows])
    print(f"  sanity check -- random_split avg PR-AUC recomputed here: {rs_avg_pr:.4f} (locked reference: 0.8462)")
    pc_avg_pr = np.mean([r["pr_auc"] for r in pc_rows])
    print(
        f"  sanity check -- phylogroup-clean avg PR-AUC recomputed here: {pc_avg_pr:.4f} (previously reported: 0.8519)"
    )

    combined = []
    for r in rs_rows:
        combined.append({"split": "random_split_full_stack", **r})
    for r in pc_rows:
        combined.append({"split": "phylogroup_stratified_clean", **r})
    df = pd.DataFrame(combined)
    df.to_csv(OUTPUT_DIR / "prevalence_invariant_metrics.csv", index=False)

    print("\n" + "=" * 100)
    print(f"PER-ANTIBIOTIC: PR-AUC / ROC-AUC / MCC / test prevalence  (threshold={THRESHOLD})")
    print("=" * 100)
    for split_name, rows in [("random_split (full_stack)", rs_rows), ("phylogroup-stratified (clean)", pc_rows)]:
        print(f"\n-- {split_name} --")
        print(f"  {'AB':<5}{'prev%':>8}{'PR-AUC':>9}{'ROC-AUC':>9}{'MCC':>8}")
        for r in rows:
            print(
                f"  {r['antibiotic']:<5}{r['prevalence_pct']:>8.1f}"
                f"{r['pr_auc']:>9.4f}{r['roc_auc']:>9.4f}{r['mcc']:>8.4f}"
            )
        print(
            f"  {'AVG':<5}{'':>8}{np.mean([r['pr_auc'] for r in rows]):>9.4f}"
            f"{np.mean([r['roc_auc'] for r in rows]):>9.4f}"
            f"{np.mean([r['mcc'] for r in rows]):>8.4f}"
        )

    print(f"\nSaved -> {OUTPUT_DIR / 'prevalence_invariant_metrics.csv'}")
    print(
        "\nMLST lineage holdout arm NOT included -- its predictions do not exist "
        "yet. The identical computation has been added into "
        "stack_lineage.py, so it will appear automatically "
        "once that step (currently queued) runs."
    )


if __name__ == "__main__":
    main()
