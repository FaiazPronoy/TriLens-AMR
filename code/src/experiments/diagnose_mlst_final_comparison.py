"""
src/experiments/diagnose_mlst_final_comparison.py
===================================================
STEP 5: index-aligned meta-stack on the MLST lineage holdout, plus the final
three-way comparison table.

Combines:
  * TriLens neural OOF/test predictions   (results/pillar_trilens_mlst/)
  * the 5 classical base predictions      (results/pillar_cascadestack_classical_mlst/)
both trained on the SAME MLST holdout split and the SAME MLST-ranked gene set.

Alignment is asserted, not assumed: the two npz files must carry identical
`train_indices` and `test_indices` (content AND order). The meta-learner is
fit on OOF/train only and applied to test only, via run_trilens_v10's own
unmodified `run_meta_stack()`.

Also computes prevalence-invariant metrics (ROC-AUC, MCC at a fixed 0.5
threshold) and the raw test-set prevalence per antibiotic, matching
diagnose_prevalence_invariant_metrics.py's convention exactly, so this split
is directly comparable to the random_split and phylogroup-stratified-clean
arms computed there (this split's own predictions did not exist yet when
that script ran, so it is folded in here instead of a fourth script).

Does not modify any existing file.
Output: results/pillar_mlst_final_comparison/

Usage:
    conda activate amr
    python -m src.experiments.diagnose_mlst_final_comparison
"""
from __future__ import annotations

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
GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered__mlst_holdout.pkl"
CONFIG_YAML_PATH = ROOT / "configs" / "config.yaml"
TRILENS_CFG_PATH = ROOT / "configs" / "experiment" / "trilens_v10.yaml"

TRILENS_NPZ = RESULTS_DIR / "pillar_trilens_mlst" / "predictions_saved.npz"
CLASSICAL_NPZ = RESULTS_DIR / "pillar_cascadestack_classical_mlst" / "predictions_saved.npz"
CLASSICAL_CSV = RESULTS_DIR / "pillar_cascadestack_classical_mlst" / "classical5_results.csv"
TRILENS_CSV = RESULTS_DIR / "pillar_trilens_mlst" / "trilens_mlst_results.csv"
OUTPUT_DIR = RESULTS_DIR / "pillar_mlst_final_comparison"
MCC_THRESHOLD = 0.5  # matches diagnose_prevalence_invariant_metrics.py's convention

# Previously computed arms of the comparison (read from their own files).
# trilens_attnfix is the delivered strain-conditioned run and supplies the
# random-split column of the output table; trilens_v10 is the same architecture
# BEFORE the fix and is only the fallback. Getting this order wrong produces a
# comparison table whose random column (AMP 0.9408) disagrees with every other
# table in Chapter 5 (AMP 0.9440).
T6_ABLATION_CSV = next(
    (p for p in (RESULTS_DIR / "trilens_attnfix" / "ablation_results.csv",
                 RESULTS_DIR / "trilens_v10" / "ablation_results.csv") if p.exists()),
    RESULTS_DIR / "trilens_v10" / "ablation_results.csv",
)
PHYLO_CLEAN_CSV = RESULTS_DIR / "pillar_t7_clean_final_comparison" / "three_way_comparison_FINAL.csv"
PHYLO_CLEAN_CLASSICAL_CSV = (RESULTS_DIR / "pillar_cascadestack_classical_phylogeny_clean" /
                             "classical5_results.csv")


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


def main() -> None:
    for p, lbl in [(TRILENS_NPZ, "TriLens MLST"), (CLASSICAL_NPZ, "classical MLST")]:
        if not p.exists():
            raise FileNotFoundError(f"{p} ({lbl}) missing -- run the earlier step first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg()

    with open(GENE_FILTERED_PKL, "rb") as f:
        gf = pickle.load(f)
    Y = gf["Y"].astype(np.float64)
    mask = gf["label_mask"].astype(bool)
    st = np.asarray(gf["st_per_genome"], dtype=object)

    tl = np.load(TRILENS_NPZ, allow_pickle=True)
    cs = np.load(CLASSICAL_NPZ, allow_pickle=True)

    assert np.array_equal(tl["train_indices"], cs["train_indices"]), (
        "train_indices mismatch between TriLens-MLST and classical-MLST -- "
        "do NOT combine positionally."
    )
    assert np.array_equal(tl["test_indices"], cs["test_indices"]), (
        "test_indices mismatch between TriLens-MLST and classical-MLST -- "
        "do NOT combine positionally."
    )
    print("Alignment verified: train_indices and test_indices identical "
          "(content and order) across the neural and classical arms.")

    train_idx, test_idx = tl["train_indices"], tl["test_indices"]
    st_tr = {st[i] for i in train_idx if st[i] is not None}
    st_te = {st[i] for i in test_idx if st[i] is not None}
    assert not (st_tr & st_te), "TRAIN/TEST ST LEAK at final meta stage"
    print(f"ST disjointness re-verified at meta stage "
          f"(train {len(st_tr)} STs vs test {len(st_te)} STs, 0 shared)")

    oof_trilens = tl["oof_base_probs"][:, -1, :]
    test_trilens = tl["test_base_probs"][:, -1, :]
    oof_classical = cs["oof_base_probs"]
    test_classical = cs["test_base_probs"]

    test_meta, meta_weights = run_meta_stack(
        oof_trilens=oof_trilens, test_trilens=test_trilens,
        oof_classical=oof_classical, test_classical=test_classical,
        Y_tr=Y[train_idx], mask_tr=mask[train_idx], cfg=cfg,
    )

    Yt, Mt = Y[test_idx], mask[test_idx]
    mlst_per_ab = {}
    for j, ab in enumerate(AB_NAMES):
        v = Mt[:, j].astype(bool)
        mlst_per_ab[ab] = float(average_precision_score(Yt[v, j], test_meta[v, j]))
    mlst_avg = float(np.mean([mlst_per_ab[ab] for ab in AB_NAMES]))
    print(f"\nMLST lineage holdout, full stack (5 classical + TriLens): "
          f"avg PR-AUC = {mlst_avg:.4f}")

    # Save the properly-aligned final stack's raw predictions -- this is
    # the artifact run_pillar_dedup_eval_mlst.py needs. Without this, a
    # dedup eval would have to fall back to an earlier, non-final
    # intermediate (e.g. TriLens-solo), exactly the imprecision already
    # noted for the phylogeny_split dedup eval (see SESSION_HANDOFF.md).
    np.savez(
        str(OUTPUT_DIR / "predictions_saved.npz"),
        test_ens_probs=test_meta,
        test_indices=test_idx,
        train_indices=train_idx,
        antibiotic_names=np.array(list(AB_NAMES)),
    )
    print(f"Saved predictions_saved.npz -> {OUTPUT_DIR} (final aligned stack, for dedup eval)")

    # Prevalence-invariant metrics (ROC-AUC, MCC) + raw test-set prevalence,
    # same convention as diagnose_prevalence_invariant_metrics.py (fixed 0.5
    # threshold for MCC) so this arm is directly comparable to the
    # random_split and phylogroup-stratified-clean arms computed there.
    extended_rows = []
    for j, ab in enumerate(AB_NAMES):
        v = Mt[:, j].astype(bool)
        y = Yt[v, j].astype(int)
        p = test_meta[v, j]
        pred = (p >= MCC_THRESHOLD).astype(int)
        extended_rows.append({
            "antibiotic": ab,
            "n_test": int(v.sum()),
            "prevalence_pct": round(100 * float(y.mean()), 2),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "roc_auc": (round(float(roc_auc_score(y, p)), 4)
                       if len(np.unique(y)) > 1 else float("nan")),
            "mcc": (round(float(matthews_corrcoef(y, pred)), 4)
                   if len(np.unique(pred)) > 1 else 0.0),
        })
    ext_df = pd.DataFrame(extended_rows)
    ext_df.to_csv(OUTPUT_DIR / "prevalence_invariant_metrics_mlst.csv", index=False)
    print(f"\nPrevalence-invariant metrics (threshold={MCC_THRESHOLD}):")
    print(f"  {'AB':<5}{'prev%':>8}{'PR-AUC':>9}{'ROC-AUC':>9}{'MCC':>8}")
    for r in extended_rows:
        print(f"  {r['antibiotic']:<5}{r['prevalence_pct']:>8.1f}"
              f"{r['pr_auc']:>9.4f}{r['roc_auc']:>9.4f}{r['mcc']:>8.4f}")
    print(f"  {'AVG':<5}{'':>8}{np.mean([r['pr_auc'] for r in extended_rows]):>9.4f}"
          f"{np.mean([r['roc_auc'] for r in extended_rows]):>9.4f}"
          f"{np.mean([r['mcc'] for r in extended_rows]):>8.4f}")

    coef_rows = []
    for ab in AB_NAMES:
        w = meta_weights.get(ab)
        row = {"antibiotic": ab, "pr_auc": mlst_per_ab[ab]}
        if w is not None:
            row.update(dict(zip(w["bases"], [round(c, 3) for c in w["coef"]])))
        coef_rows.append(row)
    pd.DataFrame(coef_rows).to_csv(
        OUTPUT_DIR / "mlst_meta_coefficients_and_prauc.csv", index=False)

    # ---- assemble the comparison table from each arm's own saved file ----
    t6 = {}
    if T6_ABLATION_CSV.exists():
        d = pd.read_csv(T6_ABLATION_CSV)
        d = d[(d["ablation"] == "T6") & (d["split_key"] == "random_split")]
        t6 = dict(zip(d["ab"], d["prauc"]))

    phylo_clean = {}
    if PHYLO_CLEAN_CSV.exists():
        d = pd.read_csv(PHYLO_CLEAN_CSV)
        d = d[d["antibiotic"] != "AVERAGE"]
        phylo_clean = dict(zip(d["antibiotic"], d["phylogeny_clean_gene_set"]))

    rows = []
    for ab in AB_NAMES:
        rows.append({
            "antibiotic": ab,
            "random_split_T6": round(t6[ab], 4) if ab in t6 else None,
            "phylogroup_stratified_clean": (round(float(phylo_clean[ab]), 4)
                                            if ab in phylo_clean else None),
            "mlst_lineage_holdout": round(mlst_per_ab[ab], 4),
        })
    df_cmp = pd.DataFrame(rows)
    avg_row = {
        "antibiotic": "AVERAGE",
        "random_split_T6": round(float(np.mean(list(t6.values()))), 4) if t6 else None,
        "phylogroup_stratified_clean": (
            round(float(np.mean([float(v) for v in phylo_clean.values()])), 4)
            if phylo_clean else None),
        "mlst_lineage_holdout": round(mlst_avg, 4),
    }
    df_cmp = pd.concat([df_cmp, pd.DataFrame([avg_row])], ignore_index=True)
    df_cmp.to_csv(OUTPUT_DIR / "three_way_comparison_FINAL.csv", index=False)

    classical_avg = float(pd.read_csv(CLASSICAL_CSV)["ensemble_pr_auc"].mean())
    trilens_solo_avg = float(pd.read_csv(TRILENS_CSV)["pr_auc"].mean())
    phylo_classical_avg = (float(pd.read_csv(PHYLO_CLEAN_CLASSICAL_CSV)["ensemble_pr_auc"].mean())
                           if PHYLO_CLEAN_CLASSICAL_CSV.exists() else None)

    print("\n" + "=" * 78)
    print("FINAL THREE-WAY COMPARISON (numbers only)")
    print("=" * 78)
    print(df_cmp.to_string(index=False))
    print()
    print(f"  classical-only avg (MLST holdout)      : {classical_avg:.4f}")
    if phylo_classical_avg is not None:
        print(f"  classical-only avg (phylogroup clean)  : {phylo_classical_avg:.4f}")
    print(f"  TriLens SOLO avg (MLST holdout)        : {trilens_solo_avg:.4f}")

    lines = [
        "# FINAL: MLST lineage holdout vs the two earlier splits\n",
        "| Antibiotic | Random split (T6) | Phylogroup-stratified (clean) | "
        "**MLST lineage holdout** |",
        "|---|---|---|---|",
    ]
    for _, r in df_cmp.iterrows():
        def f(v):
            return "n/a" if pd.isna(v) else f"{v:.4f}"
        bold = "**" if r["antibiotic"] == "AVERAGE" else ""
        lines.append(
            f"| {bold}{r['antibiotic']}{bold} | {f(r['random_split_T6'])} | "
            f"{f(r['phylogroup_stratified_clean'])} | "
            f"{bold}{f(r['mlst_lineage_holdout'])}{bold} |"
        )
    lines += [
        "\n## Component averages\n",
        f"- classical-only (5 bases), MLST holdout: **{classical_avg:.4f}**",
    ]
    if phylo_classical_avg is not None:
        lines.append(f"- classical-only (5 bases), phylogroup-stratified clean: "
                     f"{phylo_classical_avg:.4f}")
    lines.append(f"- TriLens solo (no classical), MLST holdout: **{trilens_solo_avg:.4f}**")
    lines.append(
        "\nAll arms use the same gene-filtering method and the same meta-stacking "
        "protocol; only the split (and therefore the split-specific gene ranking) "
        "differs. ST disjointness between train and test was asserted at split "
        "construction, at classical training, at neural training, and again here."
    )
    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved -> {OUTPUT_DIR}")
    print("STEP 5 COMPLETE")


if __name__ == "__main__":
    main()
