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
from scipy.stats import spearmanr
from sklearn.feature_selection import chi2

from trilens.data.gene_statistics import build_gene_feature_matrix_from_pkl
from trilens.utils.seeds import set_global_seed

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


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


def _train_xgb_and_explain(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
    shap_subsample: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    p = dict(params)
    p["scale_pos_weight"] = _spw(y_train)
    p["objective"] = "binary:logistic"
    p["eval_metric"] = "aucpr"
    p["seed"] = seed
    p.pop("early_stopping_rounds", None)
    n_round = p.pop("n_estimators", 1500)
    p.pop("n_jobs", None)

    dtr = xgb.DMatrix(X_train, label=y_train)
    booster = xgb.train(p, dtr, num_boost_round=n_round, verbose_eval=False)

    score = booster.get_score(importance_type="gain")
    gain_imp = np.zeros(X_train.shape[1], dtype=np.float32)
    for k, v in score.items():
        if k.startswith("f"):
            gain_imp[int(k[1:])] = float(v)

    rng = np.random.RandomState(seed)
    if shap_subsample and shap_subsample < len(X_train):
        idx = rng.choice(len(X_train), size=shap_subsample, replace=False)
        X_shap = X_train[idx]
        y_shap = y_train[idx]
    else:
        X_shap = X_train
        y_shap = y_train

    booster.set_param({"device": "cpu", "predictor": "cpu_predictor"})
    explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
    sv = explainer.shap_values(X_shap)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv, dtype=np.float32)
    if sv.ndim == 3:
        sv = sv[:, 1, :] if sv.shape[1] > 1 else sv[:, 0, :]

    mean_abs_shap = np.abs(sv).mean(axis=0).astype(np.float32)

    res_mask = y_shap == 1
    if res_mask.sum() >= 3:
        mean_shap_res = sv[res_mask].mean(axis=0).astype(np.float32)
    else:
        mean_shap_res = np.zeros_like(mean_abs_shap)

    del booster, dtr
    gc.collect()
    return gain_imp, mean_abs_shap, mean_shap_res


def _gene_level_per_ab(
    X_tr_flat: np.ndarray,
    y_tr: np.ndarray,
    n_genes: int,
    n_stats: int,
    xgb_params: dict,
    shap_subsample: int,
    seed: int,
) -> dict:

    gain_imp, abs_shap, res_shap = _train_xgb_and_explain(
        X_tr_flat,
        y_tr,
        xgb_params,
        shap_subsample,
        seed,
    )
    per_gene = {
        "gain": gain_imp.reshape(n_genes, n_stats).sum(axis=1),
        "shap_abs": abs_shap.reshape(n_genes, n_stats).sum(axis=1),
        "shap_res": res_shap.reshape(n_genes, n_stats).sum(axis=1),
    }
    return per_gene


def _snp_level_per_ab(
    X_tr_snp: np.ndarray,
    y_tr: np.ndarray,
    selected_gene_ids: list,
    gene_to_range: dict,
    prefilter_top_k: int,
    prefilter_min_variance: float,
    xgb_raw_params: dict,
    shap_subsample: int,
    seed: int,
) -> dict:

    var = _variance_chunked(X_tr_snp)
    keep_var = np.where(var > prefilter_min_variance)[0]
    chi2_stats = _chi2_chunked(X_tr_snp[:, keep_var], y_tr)
    k = int(min(prefilter_top_k, len(keep_var)))
    top_local = np.argpartition(chi2_stats, -k)[-k:]
    top_idx = keep_var[top_local]
    del var, chi2_stats
    gc.collect()

    X_top = X_tr_snp[:, top_idx].astype(np.float32)
    gain_imp, abs_shap, res_shap = _train_xgb_and_explain(
        X_top,
        y_tr,
        xgb_raw_params,
        shap_subsample,
        seed,
    )

    n_genes = len(selected_gene_ids)
    col_to_gene = -np.ones(X_tr_snp.shape[1], dtype=np.int32)
    for gi, gid in enumerate(selected_gene_ids):
        rng = gene_to_range.get(str(gid))
        if rng is None:
            continue
        start, end = int(rng[0]), int(rng[1])
        col_to_gene[start:end] = gi

    per_gene_gain = np.zeros(n_genes, dtype=np.float32)
    per_gene_shap_abs = np.zeros(n_genes, dtype=np.float32)
    per_gene_shap_res = np.zeros(n_genes, dtype=np.float32)
    for local_idx, global_col in enumerate(top_idx):
        gi = int(col_to_gene[global_col])
        if gi < 0:
            continue
        per_gene_gain[gi] += float(gain_imp[local_idx])
        per_gene_shap_abs[gi] += float(abs_shap[local_idx])
        per_gene_shap_res[gi] += float(res_shap[local_idx])

    del X_top
    gc.collect()
    return {
        "gain": per_gene_gain,
        "shap_abs": per_gene_shap_abs,
        "shap_res": per_gene_shap_res,
    }


def _build_ab_table(
    per_gene: dict,
    selected_gene_ids: list,
    top_k: int,
) -> tuple[pd.DataFrame, float, float]:

    gain = per_gene["gain"]
    shap_abs = per_gene["shap_abs"]
    shap_res = per_gene["shap_res"]

    ord_gain = np.argsort(-gain)
    ord_shap = np.argsort(-shap_abs)
    rank_gain = np.empty_like(ord_gain)
    rank_gain[ord_gain] = np.arange(len(ord_gain))
    rank_shap = np.empty_like(ord_shap)
    rank_shap[ord_shap] = np.arange(len(ord_shap))

    union = sorted(set(ord_gain[:top_k]).union(set(ord_shap[:top_k])))
    rows = []
    total_gain = float(gain.sum() + 1e-9)
    total_shap = float(shap_abs.sum() + 1e-9)
    in_top_g = set(ord_gain[:top_k])
    in_top_s = set(ord_shap[:top_k])
    for gi in union:
        rows.append(
            {
                "gene_idx": int(gi),
                "gene_id": str(selected_gene_ids[gi]),
                "gain": float(gain[gi]),
                "gain_pct": float(gain[gi] / total_gain * 100.0),
                "rank_gain": int(rank_gain[gi]) + 1,
                "shap_abs": float(shap_abs[gi]),
                "shap_pct": float(shap_abs[gi] / total_shap * 100.0),
                "rank_shap": int(rank_shap[gi]) + 1,
                "shap_signed_resistant": float(shap_res[gi]),
                "direction": (
                    "→ resistance" if shap_res[gi] > 0 else "→ susceptibility" if shap_res[gi] < 0 else "neutral"
                ),
                "in_top_K_gain": bool(gi in in_top_g),
                "in_top_K_shap": bool(gi in in_top_s),
                "both_methods_agree": bool(gi in in_top_g and gi in in_top_s),
            }
        )
    df = pd.DataFrame(rows).sort_values("rank_shap").reset_index(drop=True)

    if len(union) >= 3:
        rho, _ = spearmanr(rank_gain[union], rank_shap[union])
        spear_rho = float(rho) if rho is not None else 0.0
    else:
        spear_rho = 0.0

    jacc = len(in_top_g & in_top_s) / max(len(in_top_g | in_top_s), 1)

    return df, spear_rho, float(jacc)


def _cross_ab(
    per_ab: dict[str, pd.DataFrame],
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    abs_list = list(per_ab.keys())
    sets = {ab: set(df[df["in_top_K_shap"]].head(top_k)["gene_id"].tolist()) for ab, df in per_ab.items()}

    jacc = pd.DataFrame(index=abs_list, columns=abs_list, dtype=np.float32)
    for a in abs_list:
        for b in abs_list:
            sa, sb = sets[a], sets[b]
            jacc.loc[a, b] = len(sa & sb) / max(len(sa | sb), 1) if a != b else 1.0

    counter: dict[str, list[str]] = defaultdict(list)
    for ab, s in sets.items():
        for g in s:
            counter[g].append(ab)
    rows = [
        {"gene_id": g, "n_antibiotics": len(abs_in), "antibiotics": ",".join(sorted(abs_in))}
        for g, abs_in in counter.items()
        if len(abs_in) >= 3
    ]
    multi = pd.DataFrame(rows).sort_values("n_antibiotics", ascending=False)
    return jacc, multi


def _save_heatmap(jacc: pd.DataFrame, out_path: Path, title: str) -> None:
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
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Jaccard")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_per_ab_bar(per_ab: dict[str, pd.DataFrame], top_k: int, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(per_ab)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n))
    if n == 1:
        axes = [axes]
    for ax, (ab, df) in zip(axes, per_ab.items()):
        d = df.head(top_k)
        colors = ["#2ca02c" if x > 0 else "#d62728" if x < 0 else "#7f7f7f" for x in d["shap_signed_resistant"]]
        ax.barh(d["gene_id"][::-1], d["shap_abs"][::-1], color=colors[::-1])
        ax.set_title(f"{ab} - top-{len(d)} by SHAP (green=↑resistance, red=↑susceptibility)")
        ax.set_xlabel("Mean |SHAP|")
        ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def run(cfg: DictConfig) -> None:
    if not HAS_SHAP:
        raise ImportError(
            "SHAP not installed. Run:  pip install shap\n(One-time; SHAP is a standard ML interpretability library.)"
        )
    set_global_seed(cfg.seed)
    exp_cfg = cfg.experiment
    out_dir = Path(exp_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config_used.yaml")

    print("=" * 80)
    print("  SHAP + XGB-gain dual-method interpretability")
    print("=" * 80)
    print(f"  SHAP    : {shap.__version__}")

    print("\n[1/4] Loading data...")
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

    print(f"  X_flat: {X_flat.shape}, X_snp: {X_snp_int8.shape}")

    antibiotics = list(exp_cfg.antibiotics)
    top_k = int(exp_cfg.top_k)
    shap_sub = int(exp_cfg.shap_subsample)
    do_b = bool(exp_cfg.get("run_gene_level", True))
    do_c = bool(exp_cfg.get("run_snp_level", True))

    gene_level: dict[str, pd.DataFrame] = {}
    snp_level: dict[str, pd.DataFrame] = {}
    metrics_rows = []

    if do_b:
        print("\n[2/4] Gene level - XGB-gene-aggregated (per AB)...")
        for ab in antibiotics:
            if ab not in label_names:
                continue
            ab_i = label_names.index(ab)
            m = label_mask[tr_idx, ab_i]
            valid_tr = tr_idx[m]
            y_tr = Y[valid_tr, ab_i].astype(np.int32)
            if len(np.unique(y_tr)) < 2:
                continue
            t0 = time.time()
            per_gene = _gene_level_per_ab(
                X_flat[valid_tr],
                y_tr,
                n_genes=n_genes,
                n_stats=n_stats,
                xgb_params=dict(exp_cfg.xgb_params),
                shap_subsample=shap_sub,
                seed=cfg.seed,
            )
            df, rho, jacc = _build_ab_table(per_gene, selected_gene_ids, top_k)
            gene_level[ab] = df
            df.to_csv(out_dir / f"shap_gene_level_{ab}.csv", index=False, encoding="utf-8")
            metrics_rows.append(
                {
                    "level": "gene",
                    "antibiotic": ab,
                    "n_train": len(y_tr),
                    "n_pos": int(y_tr.sum()),
                    "spearman_gain_vs_shap": rho,
                    "jaccard_topK_gain_vs_shap": jacc,
                    "time_sec": time.time() - t0,
                }
            )
            print(
                f"  {ab:<5}  n_train={len(y_tr)}  "
                f"ρ_gain↔SHAP={rho:+.2f}  J={jacc:.2f}  "
                f"top1={df['gene_id'].iloc[0]}  ({time.time() - t0:.0f}s)"
            )

    if do_c:
        print("\n[3/4] SNP level - XGB-raw-chi² (per AB)...")
        for ab in antibiotics:
            if ab not in label_names:
                continue
            ab_i = label_names.index(ab)
            m = label_mask[tr_idx, ab_i]
            valid_tr = tr_idx[m]
            y_tr = Y[valid_tr, ab_i].astype(np.int32)
            if len(np.unique(y_tr)) < 2:
                continue
            t0 = time.time()
            per_gene = _snp_level_per_ab(
                X_snp_int8[valid_tr],
                y_tr,
                selected_gene_ids=selected_gene_ids,
                gene_to_range=gene_to_range,
                prefilter_top_k=int(exp_cfg.prefilter_top_k),
                prefilter_min_variance=float(exp_cfg.prefilter_min_variance),
                xgb_raw_params=dict(exp_cfg.xgb_raw_params),
                shap_subsample=shap_sub,
                seed=cfg.seed,
            )
            df, rho, jacc = _build_ab_table(per_gene, selected_gene_ids, top_k)
            snp_level[ab] = df
            df.to_csv(out_dir / f"shap_snp_level_{ab}.csv", index=False, encoding="utf-8")
            metrics_rows.append(
                {
                    "level": "snp",
                    "antibiotic": ab,
                    "n_train": len(y_tr),
                    "n_pos": int(y_tr.sum()),
                    "spearman_gain_vs_shap": rho,
                    "jaccard_topK_gain_vs_shap": jacc,
                    "time_sec": time.time() - t0,
                }
            )
            print(
                f"  {ab:<5}  n_train={len(y_tr)}  "
                f"ρ_gain↔SHAP={rho:+.2f}  J={jacc:.2f}  "
                f"top1={df['gene_id'].iloc[0]}  ({time.time() - t0:.0f}s)"
            )

    pd.DataFrame(metrics_rows).to_csv(
        out_dir / "concordance_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    print("\n[4/4] Cross-AB analyses + heatmaps + SUMMARY...")
    primary = snp_level if snp_level else gene_level
    if primary:
        jacc_c, multi_c = _cross_ab(primary, top_k=top_k)
        jacc_c.to_csv(out_dir / "cross_ab_jaccard_shap.csv", encoding="utf-8")
        multi_c.to_csv(out_dir / "multi_drug_markers_shap.csv", index=False, encoding="utf-8")
        _save_heatmap(jacc_c, out_dir / "cross_ab_heatmap_shap.png", "Cross-AB Jaccard (SHAP top-K)")
        _save_per_ab_bar(primary, top_k=10, out_path=out_dir / "per_ab_top_genes_shap.png")

    consensus_rows = []
    if gene_level and snp_level:
        for ab in antibiotics:
            if ab not in gene_level or ab not in snp_level:
                continue
            b_set = set(gene_level[ab][gene_level[ab]["in_top_K_shap"]].head(top_k)["gene_id"])
            c_set = set(snp_level[ab][snp_level[ab]["in_top_K_shap"]].head(top_k)["gene_id"])
            inter = b_set & c_set
            consensus_rows.append(
                {
                    "antibiotic": ab,
                    "shap_gene_level_size": len(b_set),
                    "shap_snp_level_size": len(c_set),
                    "intersection": len(inter),
                    "jaccard": len(inter) / max(len(b_set | c_set), 1),
                    "consensus_genes": ",".join(sorted(inter)),
                }
            )
        pd.DataFrame(consensus_rows).to_csv(
            out_dir / "gene_vs_snp_consensus.csv",
            index=False,
            encoding="utf-8",
        )

    lines = [
        "# SHAP + Gain Tree-Based Interpretability\n",
        "## Methodology\n",
        "For each antibiotic we train two XGBoost models (gene-aggregated and",
        "raw chi²-prefiltered SNPs - identical configs to the classical stack's base",
        "learners) and compute per-gene importance using both:\n",
        '1. **Gain-based importance** from `booster.get_score(importance_type="gain")`',
        "   - aggregated per gene by summing across the four gene-statistic",
        "     features (Gene level) or across all SNPs mapped back to that gene via",
        "     `gene_to_range` (SNP level).",
        "2. **SHAP** via TreeExplainer (`tree_path_dependent` backend, the",
        "   theoretically grounded interventional explainer for tree models).",
        "   Per-gene aggregation as above; mean |SHAP| over training samples is",
        "   the importance score. We also compute *directional* SHAP - mean",
        "   signed SHAP value over resistant samples only - to indicate whether",
        "   a gene drives prediction toward resistance (+) or susceptibility (-).",
        "",
        "Concordance is reported as Spearman ρ on the union of top-K genes",
        "(gain rank vs SHAP rank) and Jaccard similarity of the top-K sets.\n",
    ]

    if metrics_rows:
        lines.append("## Gain vs SHAP concordance per AB\n")
        lines.append("| Level | AB | Spearman ρ | Jaccard top-K | Time (s) |")
        lines.append("|---|---|---|---|---|")
        for r in metrics_rows:
            lines.append(
                f"| {r['level']} | {r['antibiotic']} | "
                f"{r['spearman_gain_vs_shap']:+.3f} | "
                f"{r['jaccard_topK_gain_vs_shap']:.2f} | {r['time_sec']:.0f} |"
            )
        lines.append("")
        df_m = pd.DataFrame(metrics_rows)
        avg_b = (
            df_m[df_m["level"] == "gene"]["spearman_gain_vs_shap"].mean()
            if (df_m["level"] == "gene").any()
            else float("nan")
        )
        avg_c = (
            df_m[df_m["level"] == "snp"]["spearman_gain_vs_shap"].mean()
            if (df_m["level"] == "snp").any()
            else float("nan")
        )
        lines.append(
            f"**Avg Spearman ρ - Gene level: {avg_b:+.3f}, SNP level: {avg_c:+.3f}** "
            "(high ρ → gain and SHAP agree on ranking → robust signal).\n"
        )

    for tag, planned in [
        ("Gene level (XGB gene-aggregated, SHAP-ranked)", gene_level),
        ("SNP level (XGB raw chi², SHAP-ranked)", snp_level),
    ]:
        if not planned:
            continue
        lines.append(f"## {tag}\n")
        for ab in antibiotics:
            if ab not in planned:
                continue
            df = planned[ab][planned[ab]["in_top_K_shap"]].head(top_k)
            lines.append(f"### {ab}\n")
            lines.append(
                "| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for _, r in df.iterrows():
                arrow = "↑res" if r["shap_signed_resistant"] > 0 else "↓res" if r["shap_signed_resistant"] < 0 else "-"
                lines.append(
                    f"| {int(r['rank_shap'])} | {int(r['rank_gain'])} | "
                    f"{r['gene_id']} | {r['shap_pct']:.2f}% | {r['gain_pct']:.2f}% "
                    f"| {arrow} ({r['shap_signed_resistant']:+.3f}) "
                    f"| {'✓' if r['both_methods_agree'] else ' '} |"
                )
            lines.append("")

    if primary:
        lines.append("## Cross-antibiotic Jaccard similarity (SHAP top-K)\n")
        lines.append("| | " + " | ".join(jacc_c.columns) + " |")
        lines.append("|" + "---|" * (len(jacc_c.columns) + 1))
        for ab in jacc_c.index:
            cells = [f"{jacc_c.loc[ab, c]:.2f}" for c in jacc_c.columns]
            lines.append(f"| {ab} | " + " | ".join(cells) + " |")
        lines.append("")

        if len(multi_c):
            lines.append("## Multi-drug-resistance candidate genes (in SHAP top-K of ≥3 ABs)\n")
            lines.append("| Gene | # ABs | Antibiotics |")
            lines.append("|---|---|---|")
            for _, r in multi_c.head(30).iterrows():
                lines.append(f"| {r['gene_id']} | {int(r['n_antibiotics'])} | {r['antibiotics']} |")
            lines.append("")

    if consensus_rows:
        lines.append("## Gene level vs SNP level consensus genes (in SHAP top-K of BOTH plans)\n")
        lines.append("| AB | Intersection | Jaccard | Consensus genes |")
        lines.append("|---|---|---|---|")
        for r in consensus_rows:
            lines.append(
                f"| {r['antibiotic']} | {r['intersection']}/{top_k} | {r['jaccard']:.2f} | {r['consensus_genes']} |"
            )
        lines.append("")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("  SHAP gene importance complete")
    print(f"  Outputs -> {out_dir}")
    print("=" * 80)
