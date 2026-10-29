from __future__ import annotations

import gc
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_selection import chi2

from trilens.data.gene_statistics import build_gene_feature_matrix_from_pkl
from trilens.utils.seeds import set_global_seed


def _spw(y):
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return max(1.0, neg / max(pos, 1.0))


def _variance_chunked(X, chunk=10_000):
    n = X.shape[1]
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = X[:, s:e].astype(np.float32).var(axis=0)
    return out


def _chi2_chunked(X, y, chunk=10_000):
    n = X.shape[1]
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        c, _ = chi2(X[:, s:e].astype(np.float32), y)
        out[s:e] = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _xgb_gene_importance(
    X_train_flat: np.ndarray,
    y_train: np.ndarray,
    n_genes: int,
    n_stats: int,
    xgb_params: dict,
) -> np.ndarray:

    p = dict(xgb_params)
    p["scale_pos_weight"] = _spw(y_train)
    p["objective"] = "binary:logistic"
    p["eval_metric"] = "aucpr"
    p.pop("early_stopping_rounds", 0)
    n_round = p.pop("n_estimators", 1500)
    p.pop("n_jobs", None)

    dtr = xgb.DMatrix(X_train_flat, label=y_train)
    booster = xgb.train(p, dtr, num_boost_round=n_round, verbose_eval=False)

    score = booster.get_score(importance_type="gain")
    feature_imp = np.zeros(X_train_flat.shape[1], dtype=np.float32)
    for k, v in score.items():
        if k.startswith("f"):
            feature_imp[int(k[1:])] = float(v)

    per_gene = feature_imp.reshape(n_genes, n_stats).sum(axis=1)
    del booster
    gc.collect()
    return per_gene


def _xgb_raw_chi2_per_gene(
    X_train_snp: np.ndarray,
    y_train: np.ndarray,
    selected_gene_ids: list,
    gene_to_range: dict,
    prefilter_top_k: int,
    prefilter_min_variance: float,
    xgb_raw_params: dict,
) -> tuple[np.ndarray, dict]:

    var = _variance_chunked(X_train_snp)
    keep_var = np.where(var > prefilter_min_variance)[0]
    chi2_stats = _chi2_chunked(X_train_snp[:, keep_var], y_train)
    k = int(min(prefilter_top_k, len(keep_var)))
    top_local = np.argpartition(chi2_stats, -k)[-k:]
    top_idx = keep_var[top_local]
    del var, chi2_stats
    gc.collect()

    X_top = X_train_snp[:, top_idx].astype(np.float32)
    p = dict(xgb_raw_params)
    p["scale_pos_weight"] = _spw(y_train)
    p["objective"] = "binary:logistic"
    p["eval_metric"] = "aucpr"
    p.setdefault("colsample_bytree", 1.0)
    p.setdefault("colsample_bynode", 0.5)
    p.pop("early_stopping_rounds", 0)
    n_round = p.pop("n_estimators", 1500)
    p.pop("n_jobs", None)

    dtr = xgb.DMatrix(X_top, label=y_train)
    booster = xgb.train(p, dtr, num_boost_round=n_round, verbose_eval=False)

    score = booster.get_score(importance_type="gain")
    snp_imp = np.zeros(X_top.shape[1], dtype=np.float32)
    for fk, v in score.items():
        if fk.startswith("f"):
            snp_imp[int(fk[1:])] = float(v)

    n_genes = len(selected_gene_ids)
    per_gene_imp = np.zeros(n_genes, dtype=np.float32)
    snp_to_gene_imp: dict = defaultdict(dict)

    col_to_gene = -np.ones(X_train_snp.shape[1], dtype=np.int32)
    for gi, gid in enumerate(selected_gene_ids):
        rng = gene_to_range.get(str(gid))
        if rng is None:
            continue
        start, end = int(rng[0]), int(rng[1])
        col_to_gene[start:end] = gi

    for local_idx, global_col in enumerate(top_idx):
        gi = int(col_to_gene[global_col])
        if gi < 0:
            continue
        imp = float(snp_imp[local_idx])
        if imp == 0.0:
            continue
        per_gene_imp[gi] += imp
        gid = str(selected_gene_ids[gi])
        snp_to_gene_imp[gid][int(global_col)] = imp

    del booster, X_top
    gc.collect()
    return per_gene_imp, dict(snp_to_gene_imp)


def _top_k_genes(
    importance: np.ndarray,
    selected_gene_ids: list,
    top_k: int,
) -> pd.DataFrame:
    order = np.argsort(-importance)[:top_k]
    rows = []
    total = float(importance.sum() + 1e-9)
    for r, gi in enumerate(order):
        if importance[gi] == 0.0:
            break
        rows.append(
            {
                "rank": r + 1,
                "gene_idx": int(gi),
                "gene_id": str(selected_gene_ids[gi]),
                "importance": float(importance[gi]),
                "pct_total_importance": float(importance[gi] / total * 100.0),
            }
        )
    return pd.DataFrame(rows)


def _cross_ab_analysis(
    per_ab_top: dict[str, pd.DataFrame],
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    abs_list = list(per_ab_top.keys())
    sets = {ab: set(df.head(top_k)["gene_id"].tolist()) for ab, df in per_ab_top.items()}

    jacc = pd.DataFrame(index=abs_list, columns=abs_list, dtype=np.float32)
    for a in abs_list:
        for b in abs_list:
            sa, sb = sets[a], sets[b]
            jacc.loc[a, b] = len(sa & sb) / max(len(sa | sb), 1) if a != b else 1.0

    counter: dict[str, list[str]] = defaultdict(list)
    for ab, s in sets.items():
        for g in s:
            counter[g].append(ab)
    rows = []
    for g, abs_in in counter.items():
        if len(abs_in) >= 3:
            rows.append(
                {
                    "gene_id": g,
                    "n_antibiotics": len(abs_in),
                    "antibiotics": ",".join(sorted(abs_in)),
                }
            )
    multi = pd.DataFrame(rows).sort_values("n_antibiotics", ascending=False)
    return jacc, multi


def _save_heatmap_cross_ab(jacc: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(jacc.values.astype(np.float32), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(jacc.columns)))
    ax.set_xticklabels(jacc.columns)
    ax.set_yticks(range(len(jacc.index)))
    ax.set_yticklabels(jacc.index)
    for i in range(len(jacc.index)):
        for j in range(len(jacc.columns)):
            ax.text(
                j,
                i,
                f"{jacc.values[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if jacc.values[i, j] < 0.5 else "black",
                fontsize=9,
            )
    ax.set_title("Cross-antibiotic Jaccard similarity\n(top-K gene sets)")
    plt.colorbar(im, ax=ax, label="Jaccard")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_per_ab_bar(per_ab_top: dict[str, pd.DataFrame], top_k: int, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(per_ab_top)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n))
    if n == 1:
        axes = [axes]
    for ax, (ab, df) in zip(axes, per_ab_top.items()):
        d = df.head(top_k)
        ax.barh(d["gene_id"][::-1], d["importance"][::-1])
        ax.set_title(f"{ab} - top-{len(d)} genes (XGB gain)")
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "per_ab_top_genes.png", dpi=180)
    plt.close(fig)


def run(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    exp_cfg = cfg.experiment
    out_dir = Path(exp_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config_used.yaml")

    print("=" * 80)
    print("  XGB tree-based interpretability + cross-AB analysis")
    print("=" * 80)

    print("\n[1/5] Loading data...")
    pkl_path = str(cfg.paths.processed_dir).rstrip("/\\") + "/arcadia_v2_gene_filtered.pkl"
    bundle = build_gene_feature_matrix_from_pkl(
        gene_filtered_pkl_path=pkl_path,
        stats=tuple(exp_cfg.gene_stats),
    )
    X_flat = bundle["X_gene_features"].astype(np.float32)
    selected_gene_ids = [str(x) for x in bundle["gene_ids"]]
    n_genes = len(selected_gene_ids)
    n_stats = len(bundle["stats_used"])
    Y = bundle["Y"].astype(np.int32)
    label_mask = bundle["label_mask"].astype(bool)
    label_names = [str(x) for x in bundle["label_names"]]
    splits = bundle["random_split"]
    tr_idx = np.asarray(splits["train_idx"])

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    X_snp_int8 = d["X_snp_filtered"]
    gene_to_range = {str(k): v for k, v in d["gene_to_range"].items()}

    print(f"  X_flat: {X_flat.shape}, X_snp: {X_snp_int8.shape}, n_genes={n_genes}, n_train={len(tr_idx)}")

    antibiotics = list(exp_cfg.antibiotics)
    top_k = int(exp_cfg.top_k)
    do_gene_level = bool(exp_cfg.get("run_gene_level", True))
    do_snp_level = bool(exp_cfg.get("run_snp_level", True))

    gene_level_top: dict[str, pd.DataFrame] = {}
    snp_level_top: dict[str, pd.DataFrame] = {}
    plan_c_snp_detail: dict[str, dict] = {}

    if do_gene_level:
        print("\n[2/5] Gene level - XGB on gene-aggregated features (per AB)...")
        for ab in antibiotics:
            if ab not in label_names:
                continue
            ab_i = label_names.index(ab)
            m = label_mask[tr_idx, ab_i]
            valid_tr = tr_idx[m]
            y_tr = Y[valid_tr, ab_i].astype(np.int32)
            if len(np.unique(y_tr)) < 2:
                continue
            X_tr = X_flat[valid_tr]
            t0 = time.time()
            per_gene = _xgb_gene_importance(
                X_tr,
                y_tr,
                n_genes=n_genes,
                n_stats=n_stats,
                xgb_params=dict(exp_cfg.xgb_params),
            )
            gene_level_top[ab] = _top_k_genes(per_gene, selected_gene_ids, top_k)
            print(
                f"  {ab:<5}  n_train={len(y_tr)} (pos={int(y_tr.sum())})  "
                f"({time.time() - t0:.0f}s)  "
                f"top1={gene_level_top[ab]['gene_id'].iloc[0]}"
            )
            gene_level_top[ab].to_csv(out_dir / f"gain_gene_level_{ab}.csv", index=False, encoding="utf-8")

    if do_snp_level:
        print("\n[3/5] SNP level - XGB on chi²-prefiltered raw SNPs (per AB)...")
        for ab in antibiotics:
            if ab not in label_names:
                continue
            ab_i = label_names.index(ab)
            m = label_mask[tr_idx, ab_i]
            valid_tr = tr_idx[m]
            y_tr = Y[valid_tr, ab_i].astype(np.int32)
            if len(np.unique(y_tr)) < 2:
                continue
            X_tr_snp = X_snp_int8[valid_tr]
            t0 = time.time()
            per_gene_imp, snp_detail = _xgb_raw_chi2_per_gene(
                X_tr_snp,
                y_tr,
                selected_gene_ids=selected_gene_ids,
                gene_to_range=gene_to_range,
                prefilter_top_k=int(exp_cfg.prefilter_top_k),
                prefilter_min_variance=float(exp_cfg.prefilter_min_variance),
                xgb_raw_params=dict(exp_cfg.xgb_raw_params),
            )
            snp_level_top[ab] = _top_k_genes(per_gene_imp, selected_gene_ids, top_k)
            plan_c_snp_detail[ab] = snp_detail
            print(
                f"  {ab:<5}  n_train={len(y_tr)} (pos={int(y_tr.sum())})  "
                f"({time.time() - t0:.0f}s)  "
                f"top1={snp_level_top[ab]['gene_id'].iloc[0]} "
                f"(imp={snp_level_top[ab]['importance'].iloc[0]:.2f})"
            )
            snp_level_top[ab].to_csv(out_dir / f"gain_snp_level_{ab}.csv", index=False, encoding="utf-8")

    print("\n[4/5] Cross-antibiotic Jaccard similarity + multi-drug markers...")
    primary = snp_level_top if snp_level_top else gene_level_top
    jacc, multi = _cross_ab_analysis(primary, top_k=top_k)
    jacc.to_csv(out_dir / "cross_ab_jaccard.csv", encoding="utf-8")
    multi.to_csv(out_dir / "multi_drug_markers.csv", index=False, encoding="utf-8")
    _save_heatmap_cross_ab(jacc, out_dir / "cross_ab_heatmap.png")
    _save_per_ab_bar(primary, top_k=10, out_dir=out_dir)
    print(f"  Jaccard matrix saved. Multi-drug markers (>=3 ABs): {len(multi)} genes.")

    concordance_rows = []
    if gene_level_top and snp_level_top:
        for ab in antibiotics:
            if ab not in gene_level_top or ab not in snp_level_top:
                continue
            b_set = set(gene_level_top[ab].head(top_k)["gene_id"].tolist())
            c_set = set(snp_level_top[ab].head(top_k)["gene_id"].tolist())
            inter = b_set & c_set
            concordance_rows.append(
                {
                    "antibiotic": ab,
                    "shap_gene_level_size": len(b_set),
                    "shap_snp_level_size": len(c_set),
                    "intersection": len(inter),
                    "jaccard": len(inter) / max(len(b_set | c_set), 1),
                    "shared_genes": ",".join(sorted(inter)),
                }
            )
        pd.DataFrame(concordance_rows).to_csv(
            out_dir / "gene_vs_snp_concordance.csv",
            index=False,
            encoding="utf-8",
        )

    print("\n[5/5] Writing SUMMARY.md...")
    lines = [
        "# Tree-Based Interpretability\n",
        "## Methodology\n",
        "The attention pool of GeneAttentionNet gave near-uniform",
        "weightings (max deviation 15% from 1/n_genes baseline, a known limitation",
        "of softmax attention over long sequences in multi-task settings), we",
        "switched to tree-based feature importance from the two strongest base",
        "learners of the classical stack.\n",
        "**Gene level** - XGBoost on gene-aggregated features (4 stats × 5000 genes).",
        "Per-gene importance = sum of gain-based feature importance across the 4",
        "stats.\n",
        "**SNP level** - XGBoost on chi²-prefiltered raw SNPs (top 50K SNPs by chi²).",
        "Each selected SNP is mapped back to its parent gene via `gene_to_range`.",
        "Per-gene importance = sum of gain-based importance across that gene's",
        "selected SNPs. This was the classical stack's strongest base learner overall.\n",
    ]

    for tag, planned in [
        ("Gene level (XGB gene-aggregated)", gene_level_top),
        ("SNP level (XGB raw chi²-prefiltered)", snp_level_top),
    ]:
        if not planned:
            continue
        lines.append(f"## {tag}\n")
        for ab in antibiotics:
            if ab not in planned:
                continue
            df = planned[ab].head(top_k)
            lines.append(f"### {ab}\n")
            lines.append("| Rank | Gene ID | Importance | % of total |")
            lines.append("|---|---|---|---|")
            for _, r in df.iterrows():
                lines.append(
                    f"| {int(r['rank'])} | {r['gene_id']} | {r['importance']:.2f} | {r['pct_total_importance']:.2f}% |"
                )
            lines.append("")

    lines.append("## Cross-antibiotic Jaccard similarity\n")
    lines.append("Jaccard similarity between per-AB top-K gene sets:\n")
    lines.append("| | " + " | ".join(jacc.columns) + " |")
    lines.append("|" + "---|" * (len(jacc.columns) + 1))
    for ab in jacc.index:
        cells = [f"{jacc.loc[ab, c]:.2f}" for c in jacc.columns]
        lines.append(f"| {ab} | " + " | ".join(cells) + " |")
    lines.append("")

    if len(multi):
        lines.append("## Multi-drug-resistance candidate genes (in top-K of ≥3 ABs)\n")
        lines.append("| Gene ID | # ABs | Antibiotics |")
        lines.append("|---|---|---|")
        for _, r in multi.head(30).iterrows():
            lines.append(f"| {r['gene_id']} | {int(r['n_antibiotics'])} | {r['antibiotics']} |")
        lines.append("")

    if concordance_rows:
        lines.append("## Gene level vs SNP level concordance\n")
        lines.append(
            "Concordance between gene-aggregated XGB and raw-SNP XGB rankings (high overlap = robust signal):\n"
        )
        lines.append("| AB | Intersection / top-K | Jaccard |")
        lines.append("|---|---|---|")
        for r in concordance_rows:
            lines.append(f"| {r['antibiotic']} | {r['intersection']}/{top_k} | {r['jaccard']:.2f} |")
        lines.append("")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "=" * 80)
    print("  Gain-based gene importance complete")
    print(f"  Outputs -> {out_dir}")
    print("=" * 80)
