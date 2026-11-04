from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score

from trilens.experiments.trilens_stack import run_meta_stack
from trilens.models.trilens_network import AB_NAMES

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
CLEAN_GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered__phylogeny_clean.pkl"
CONFIG_YAML_PATH = ROOT / "configs" / "config.yaml"
TRILENS_CFG_PATH = ROOT / "configs" / "experiment" / "trilens_stack.yaml"

full_stack_phylogroup_CLEAN_NPZ = RESULTS_DIR / "trilens_neural_phylogroup" / "predictions_saved.npz"
CLASSICAL_CLEAN_NPZ = RESULTS_DIR / "classical_stack_phylogroup" / "predictions_saved.npz"
OUTPUT_DIR = RESULTS_DIR / "trilens_stack_phylogroup"

full_stack_RANDOM_SPLIT = {"AMP": None, "CIP": None, "AMC": None, "SXT": None, "GEN": None, "CTX": None, "CAZ": None}
full_stack_AVG_LOCKED = 0.8462
ALIGNED_T7_LEAKED_GENE_SET = {
    "AMP": 0.9338,
    "CIP": 0.9600,
    "AMC": 0.7957,
    "SXT": 0.8295,
    "GEN": 0.7108,
    "CTX": 0.8818,
    "CAZ": 0.8466,
}
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
    p = next(
        (
            q
            for q in (
                RESULTS_DIR / "trilens_stack_random" / "ablation_results.csv",
                RESULTS_DIR / "trilens_stack_random" / "ablation_results.csv",
            )
            if q.exists()
        ),
        None,
    )
    if p is None:
        return None
    print(f"  random-split full_stack column read from: {p.parent.name}")
    df = pd.read_csv(p)
    full_stack_result = df[(df["ablation"] == "full_stack") & (df["split_key"] == "random_split")]
    return dict(zip(full_stack_result["ab"], full_stack_result["prauc"]))


def main() -> None:
    for p, label in [
        (full_stack_phylogroup_CLEAN_NPZ, "full_stack_phylogroup clean gene set"),
        (CLASSICAL_CLEAN_NPZ, "classical clean gene set"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{p} ({label}) not found -- run the preceding stage first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg()
    Y, mask = _load_labels_clean()

    phylogroup_stack = np.load(full_stack_phylogroup_CLEAN_NPZ, allow_pickle=True)
    cs = np.load(CLASSICAL_CLEAN_NPZ, allow_pickle=True)

    assert np.array_equal(phylogroup_stack["train_indices"], cs["train_indices"]), (
        "train_indices mismatch between full_stack_phylogroup-clean and classical-clean -- "
        "do NOT combine positionally, alignment assumption is false."
    )
    assert np.array_equal(phylogroup_stack["test_indices"], cs["test_indices"]), (
        "test_indices mismatch between full_stack_phylogroup-clean and classical-clean -- "
        "do NOT combine positionally, alignment assumption is false."
    )
    print(
        "Alignment verified: train_indices and test_indices are identical "
        "(content and order) between full_stack_phylogroup-clean and classical-clean."
    )

    train_idx, test_idx = phylogroup_stack["train_indices"], phylogroup_stack["test_indices"]
    oof_trilens = phylogroup_stack["oof_base_probs"][:, -1, :]
    test_trilens = phylogroup_stack["test_base_probs"][:, -1, :]
    oof_classical = cs["oof_base_probs"]
    test_classical = cs["test_base_probs"]

    test_meta, meta_weights = run_meta_stack(
        oof_trilens=oof_trilens,
        test_trilens=test_trilens,
        oof_classical=oof_classical,
        test_classical=test_classical,
        Y_tr=Y[train_idx],
        mask_tr=mask[train_idx],
        cfg=cfg,
    )

    Yt, Mt = Y[test_idx], mask[test_idx]
    clean_t7_per_ab = {}
    for ab_idx, ab in enumerate(AB_NAMES):
        valid = Mt[:, ab_idx].astype(bool)
        clean_t7_per_ab[ab] = float(average_precision_score(Yt[valid, ab_idx], test_meta[valid, ab_idx]))
    clean_t7_per_ab["_avg"] = float(np.mean([clean_t7_per_ab[ab] for ab in AB_NAMES]))
    print(f"\nProperly-aligned full_stack_phylogroup, clean gene set: avg PR-AUC = {clean_t7_per_ab['_avg']:.4f}")

    coef_rows = []
    for ab in AB_NAMES:
        w = meta_weights.get(ab)
        row = {"antibiotic": ab, "pr_auc": clean_t7_per_ab[ab]}
        if w is not None:
            row.update(dict(zip(w["bases"], [round(c, 3) for c in w["coef"]])))
        coef_rows.append(row)
    pd.DataFrame(coef_rows).to_csv(OUTPUT_DIR / "meta_coefficients_and_prauc.csv", index=False)

    full_stack_per_ab = _load_t6_from_ablation_results()

    print("\n" + "=" * 70)
    print("THREE-WAY COMPARISON (no interpretation -- numbers only)")
    print("=" * 70)
    rows = []
    for ab in AB_NAMES:
        rows.append(
            {
                "antibiotic": ab,
                "random_split_full_stack": full_stack_per_ab.get(ab) if full_stack_per_ab else None,
                "phylogeny_leaked_gene_set": ALIGNED_T7_LEAKED_GENE_SET.get(ab),
                "phylogeny_clean_gene_set": round(clean_t7_per_ab[ab], 4),
            }
        )
    df_final = pd.DataFrame(rows)
    avg_row = {
        "antibiotic": "AVERAGE",
        "random_split_full_stack": float(np.mean(list(full_stack_per_ab.values())))
        if full_stack_per_ab
        else full_stack_AVG_LOCKED,
        "phylogeny_leaked_gene_set": ALIGNED_T7_LEAKED_AVG,
        "phylogeny_clean_gene_set": round(clean_t7_per_ab["_avg"], 4),
    }
    df_final = pd.concat([df_final, pd.DataFrame([avg_row])], ignore_index=True)
    df_final.to_csv(OUTPUT_DIR / "split_comparison.csv", index=False)
    print(df_final.to_string(index=False))
    print(f"\nSaved all outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
