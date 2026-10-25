"""
src/experiments/run_pillar_shap_t6.py
======================================
SHAP interpretability for T6 (headline model), computed on its 5 classical
base learners (XGB-gene, LGBM-gene, RF-gene, LogReg-gene, XGB-raw-chi2) —
NOT on the neural TriLens branch (computationally heavy and hard to justify
for a tree/linear-focused SHAP analysis), and NOT on CascadeStack's own
saved model (no trained model checkpoints exist on disk for any base
learner in this project — verified: no joblib/pickle.dump of a fitted
model anywhere in run_cascadestack_v9.py). This script therefore refits
each of the 5 classical bases once per antibiotic, using the exact same
hyperparameters, feature sets, and random_split as CascadeStack v9 / T6
(imported directly from run_cascadestack_v9.py and trilens_v10.yaml's
v9_predictions dependency — nothing in either file is modified).

Unlike the single 90/10 fit/early-stop split used here (not the full 5-fold
OOF loop used for the actual T6 predictions), this is a single representative
refit per (antibiotic, base learner) — appropriate for interpretability
(which genes drive predictions), not for reproducing the exact OOF
prediction values themselves.

IMPORTANT (repeating a past mistake must be avoided): a previous SHAP
analysis (results/pillar3_shap/SUMMARY.md) stated "high concordance (rho>=0.7)"
in prose while its own concordance_metrics.csv showed all-negative rho values
— a direct contradiction that was caught only by rereading the CSV. This
script computes concordance_metrics.csv FIRST, then _write_summary() reads
the actual computed DataFrame (not a remembered/assumed number) to write the
verdict sentence, so the prose cannot drift from the data.

Usage:
    conda activate amr
    python -m src.experiments.run_pillar_shap_t6
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.gene_aggregated_features import build_gene_feature_matrix_from_pkl
from src.experiments.run_cascadestack_v9 import (
    _train_lgbm_gene,
    _train_logreg_gene,
    _train_rf_gene,
    _train_xgb_gene,
    _train_xgb_raw_chi2,
)
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "pillar_shap_t6"
GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered.pkl"
CFG_PATH = ROOT / "configs" / "experiment" / "stage3_cascadestack_v9.yaml"

N_SHAP_SAMPLES = 200   # test-set subsample for SHAP value computation (speed)
TOP_K = 20
SEED = 42


def _get_pos_class_shap(sv) -> np.ndarray:
    """Normalize SHAP output across model types/shap versions to (n, n_features)."""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[..., 1]
    return sv


def _build_snp_to_gene(gene_to_range: dict, n_snps: int) -> np.ndarray:
    snp_to_gene = np.full(n_snps, -1, dtype=np.int64)
    gene_ids = list(gene_to_range.keys())
    for gi, gid in enumerate(gene_ids):
        s, e = gene_to_range[gid]
        if e > s:
            snp_to_gene[s:e] = gi
    return snp_to_gene, gene_ids


def _aggregate_gene_based(shap_abs: np.ndarray, feature_names: list[str]) -> pd.Series:
    """feature_names like 'GENEID__mean' -> sum |shap| across the 4 stat columns per gene."""
    gene_of_feat = [fn.rsplit("__", 1)[0] for fn in feature_names]
    s = pd.Series(shap_abs, index=gene_of_feat)
    return s.groupby(level=0).sum().sort_values(ascending=False)


def _aggregate_raw_chi2(shap_abs: np.ndarray, top_idx: np.ndarray, snp_to_gene: np.ndarray, gene_ids: list[str]) -> pd.Series:
    gene_idx_for_cols = snp_to_gene[top_idx]
    s = pd.Series(shap_abs, index=gene_idx_for_cols)
    s = s[s.index >= 0]
    agg = s.groupby(level=0).sum().sort_values(ascending=False)
    agg.index = [gene_ids[i] for i in agg.index]
    return agg


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(CFG_PATH)

    print("Loading gene-aggregated features (same as CascadeStack v9)...")
    bundle = build_gene_feature_matrix_from_pkl(
        gene_filtered_pkl_path=str(GENE_FILTERED_PKL),
        stats=tuple(cfg.gene_stats),
    )
    X_flat = bundle["X_gene_features"].astype(np.float32)
    feature_names = bundle["feature_names"]
    gene_ids_gene_based = bundle["gene_ids"]
    Y = bundle["Y"].astype(np.int32)
    label_mask = bundle["label_mask"].astype(bool)
    label_names = [str(x) for x in bundle["label_names"]]
    splits = bundle["random_split"]
    tr_idx = np.asarray(splits["train_idx"])
    te_idx = np.asarray(splits["test_idx"])

    with open(GENE_FILTERED_PKL, "rb") as f:
        d = pickle.load(f)
    X_snp_int8 = d["X_snp_filtered"]
    gene_to_range = d["gene_to_range"]
    snp_to_gene, gene_ids_snp = _build_snp_to_gene(gene_to_range, X_snp_int8.shape[1])

    gene_based_rows = []
    raw_chi2_rows = []
    rng = np.random.default_rng(SEED)

    for ab_idx, ab in enumerate(label_names):
        t_ab = time.time()
        m = label_mask[:, ab_idx]
        tr = tr_idx[m[tr_idx]]
        te = te_idx[m[te_idx]]
        y_tr = Y[tr, ab_idx]
        y_te = Y[te, ab_idx]

        X_tr_flat, X_te_flat = X_flat[tr], X_flat[te]
        X_tr_snp, X_te_snp = X_snp_int8[tr], X_snp_int8[te]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
        fit_l, es_l = next(sss.split(X_tr_flat, y_tr))
        X_fit, y_fit = X_tr_flat[fit_l], y_tr[fit_l]
        X_es, y_es = X_tr_flat[es_l], y_tr[es_l]
        X_fit_snp, X_es_snp = X_tr_snp[fit_l], X_tr_snp[es_l]

        te_sample = rng.choice(len(te), size=min(N_SHAP_SAMPLES, len(te)), replace=False)
        X_te_flat_s = X_te_flat[te_sample]
        X_te_snp_s = X_te_snp[te_sample]

        print(f"\n-- {ab} --  train={len(y_tr)} test={len(te)} (shap sample={len(te_sample)})")

        # 1. XGB-gene
        booster = _train_xgb_gene(X_fit, y_fit, X_es, y_es, dict(cfg.xgb_params))
        sv = shap.TreeExplainer(booster).shap_values(xgb.DMatrix(X_te_flat_s))
        gene_imp = _aggregate_gene_based(np.abs(_get_pos_class_shap(sv)).mean(axis=0), feature_names)
        for rank, (gid, val) in enumerate(gene_imp.head(TOP_K).items(), 1):
            gene_based_rows.append({"antibiotic": ab, "base_learner": "XGB-gene", "rank": rank, "gene_id": gid, "mean_abs_shap": val})
        del booster

        # 2. LGBM-gene
        clf = _train_lgbm_gene(X_fit, y_fit, X_es, y_es, dict(cfg.lgbm_params))
        sv = shap.TreeExplainer(clf).shap_values(X_te_flat_s)
        gene_imp = _aggregate_gene_based(np.abs(_get_pos_class_shap(sv)).mean(axis=0), feature_names)
        for rank, (gid, val) in enumerate(gene_imp.head(TOP_K).items(), 1):
            gene_based_rows.append({"antibiotic": ab, "base_learner": "LGBM-gene", "rank": rank, "gene_id": gid, "mean_abs_shap": val})
        del clf

        # 3. RF-gene
        clf = _train_rf_gene(X_fit, y_fit, dict(cfg.rf_params))
        sv = shap.TreeExplainer(clf).shap_values(X_te_flat_s)
        gene_imp = _aggregate_gene_based(np.abs(_get_pos_class_shap(sv)).mean(axis=0), feature_names)
        for rank, (gid, val) in enumerate(gene_imp.head(TOP_K).items(), 1):
            gene_based_rows.append({"antibiotic": ab, "base_learner": "RF-gene", "rank": rank, "gene_id": gid, "mean_abs_shap": val})
        del clf

        # 4. LogReg-gene
        clf, sc = _train_logreg_gene(X_fit, y_fit)
        X_fit_scaled = sc.transform(X_fit)
        X_te_scaled = sc.transform(X_te_flat_s)
        bg = shap.sample(X_fit_scaled, min(100, len(X_fit_scaled)), random_state=SEED)
        explainer = shap.LinearExplainer(clf, bg)
        sv = explainer.shap_values(X_te_scaled)
        gene_imp = _aggregate_gene_based(np.abs(_get_pos_class_shap(sv)).mean(axis=0) if np.asarray(sv).ndim > 2 else np.abs(sv).mean(axis=0), feature_names)
        for rank, (gid, val) in enumerate(gene_imp.head(TOP_K).items(), 1):
            gene_based_rows.append({"antibiotic": ab, "base_learner": "LogReg-gene", "rank": rank, "gene_id": gid, "mean_abs_shap": val})
        del clf, sc

        # 5. XGB-raw-chi2
        booster, top_idx = _train_xgb_raw_chi2(
            X_fit_snp, y_fit, X_es_snp, y_es,
            prefilter_k=int(cfg.prefilter_top_k),
            min_var=float(cfg.prefilter_min_variance),
            params=dict(cfg.xgb_raw_params),
        )
        X_te_top_s = X_te_snp_s[:, top_idx].astype(np.float32)
        sv = shap.TreeExplainer(booster).shap_values(xgb.DMatrix(X_te_top_s))
        gene_imp = _aggregate_raw_chi2(np.abs(_get_pos_class_shap(sv)).mean(axis=0), top_idx, snp_to_gene, gene_ids_snp)
        for rank, (gid, val) in enumerate(gene_imp.head(TOP_K).items(), 1):
            raw_chi2_rows.append({"antibiotic": ab, "base_learner": "XGB-raw-chi2", "rank": rank, "gene_id": gid, "mean_abs_shap": val})
        del booster

        print(f"   done in {time.time()-t_ab:.0f}s")

    gene_df = pd.DataFrame(gene_based_rows)
    raw_df = pd.DataFrame(raw_chi2_rows)
    gene_df.to_csv(OUTPUT_DIR / "shap_top_genes_gene_based.csv", index=False)
    raw_df.to_csv(OUTPUT_DIR / "shap_top_genes_raw_chi2.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'shap_top_genes_gene_based.csv'}")
    print(f"Saved {OUTPUT_DIR / 'shap_top_genes_raw_chi2.csv'}")

    _compute_concordance_and_summary(gene_df, raw_df, label_names)


def _compute_concordance_and_summary(gene_df: pd.DataFrame, raw_df: pd.DataFrame, label_names: list[str]) -> None:
    """Pairwise Spearman rho between the 5 base learners' FULL gene rankings
    (not just top-20), per antibiotic. Computed from the actual top-K lists
    collected above is insufficient for a real rank correlation (union of
    top-20 lists across methods is small and biased); instead this reruns
    rank comparison strictly on the union of genes that appear in ANY of the
    5 base learners' top-20 lists for that antibiotic, comparing their rank
    positions (genes absent from a list are given rank = TOP_K + 1, i.e. a
    tied worst rank) — an honest, explicitly-stated approximation given only
    top-K lists are saved, not full 20000-length importance vectors.
    """
    bases_gene = ["XGB-gene", "LGBM-gene", "RF-gene", "LogReg-gene"]
    all_bases = bases_gene + ["XGB-raw-chi2"]
    combined = pd.concat([gene_df, raw_df], ignore_index=True)

    rows = []
    for ab in label_names:
        sub = combined[combined["antibiotic"] == ab]
        rank_by_base = {}
        union_genes = set()
        for b in all_bases:
            s = sub[sub["base_learner"] == b].set_index("gene_id")["rank"]
            rank_by_base[b] = s
            union_genes |= set(s.index)
        union_genes = sorted(union_genes)
        for i in range(len(all_bases)):
            for j in range(i + 1, len(all_bases)):
                b1, b2 = all_bases[i], all_bases[j]
                r1 = [rank_by_base[b1].get(g, TOP_K + 1) for g in union_genes]
                r2 = [rank_by_base[b2].get(g, TOP_K + 1) for g in union_genes]
                if len(set(r1)) < 2 or len(set(r2)) < 2:
                    rho, p = np.nan, np.nan
                else:
                    rho, p = spearmanr(r1, r2)
                jacc = len(set(rank_by_base[b1].index) & set(rank_by_base[b2].index)) / max(
                    1, len(set(rank_by_base[b1].index) | set(rank_by_base[b2].index))
                )
                rows.append({"antibiotic": ab, "base_a": b1, "base_b": b2, "spearman_rho": rho, "jaccard_top20": jacc})

    conc_df = pd.DataFrame(rows)
    conc_df.to_csv(OUTPUT_DIR / "concordance_metrics.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'concordance_metrics.csv'}")

    # Compute the actual numbers to report BEFORE writing any prose sentence.
    valid_rho = conc_df["spearman_rho"].dropna()
    avg_rho = float(valid_rho.mean()) if len(valid_rho) else float("nan")
    avg_jaccard = float(conc_df["jaccard_top20"].mean())
    frac_positive = float((valid_rho > 0).mean()) if len(valid_rho) else float("nan")

    lines = []
    lines.append("# SHAP Interpretability — T6's 5 Classical Base Learners\n")
    lines.append(
        "SHAP was computed on the 5 classical base learners that feed TriLens "
        "T6 (XGB-gene, LGBM-gene, RF-gene, LogReg-gene, XGB-raw-chi2), refit "
        "with identical hyperparameters/features/random_split to CascadeStack "
        "v9 (single 90/10 fit/early-stop split per antibiotic, not the full "
        "5-fold OOF used for the actual reported T6 predictions — appropriate "
        "for interpretability, not for reproducing exact OOF values). SHAP was "
        "NOT computed on the neural TriLens branch (Lens A/B/C) — computationally "
        "heavy and harder to attribute meaningfully; see "
        "results/trilens_v10/attention_A/B/C_topgenes.csv for that branch's own "
        "attention-based interpretability instead.\n"
    )
    lines.append("## Concordance — computed directly from concordance_metrics.csv, verified before writing\n")
    lines.append(
        f"Across all {len(conc_df)} (antibiotic, base-learner-pair) combinations: "
        f"average Spearman rho = **{avg_rho:.4f}**, "
        f"fraction of pairs with positive rho = **{frac_positive:.2f}**, "
        f"average top-20 gene Jaccard overlap = **{avg_jaccard:.4f}**.\n"
    )
    if avg_rho >= 0.5:
        verdict = "The 5 base learners show reasonably concordant gene rankings."
    elif avg_rho > 0:
        verdict = "The 5 base learners show weak positive concordance — some shared signal but substantial method-specific disagreement."
    else:
        verdict = "The 5 base learners do NOT show meaningful concordance on gene rankings — as with the earlier pillar3_shap finding, different feature representations (gene-aggregated vs raw-chi2) and model types pick up largely different genes, and this should be reported honestly rather than claimed as convergent evidence."
    lines.append(f"**Verdict (matches the actual computed numbers above): {verdict}**\n")
    lines.append("## Files\n")
    lines.append("- `shap_top_genes_gene_based.csv` — top-20 genes per (antibiotic, base learner) for the 4 gene-aggregated-feature bases\n")
    lines.append("- `shap_top_genes_raw_chi2.csv` — top-20 genes per antibiotic for XGB-raw-chi2 (SNP-level SHAP aggregated to gene)\n")
    lines.append("- `concordance_metrics.csv` — pairwise Spearman rho / Jaccard between all 5 base learners, per antibiotic\n")

    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'SUMMARY.md'}")
    print(f"\navg_rho={avg_rho:.4f}  frac_positive={frac_positive:.2f}  avg_jaccard={avg_jaccard:.4f}")


if __name__ == "__main__":
    main()
