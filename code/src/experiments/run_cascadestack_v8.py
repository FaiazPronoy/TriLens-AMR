# src/experiments/run_cascadestack_v8.py
"""
================================================================================
CascadeStack v8 — Streamlined K-fold stacking ensemble
================================================================================

Built on lessons from v7:
  - v7 overfit (val 0.978 → test 0.949) due to MLP meta + isotonic calibration
    on small calibration sets
  - v7 fragmented training data (60% fit / 15% ES / 15% cal per fold) → weaker
    base learners
  - v7 per-fold GAN retraining cost 6+ min/fold for marginal signal (OOF 0.847)

v8 design changes:
  - 2-way fold split: 90% fit / 10% ES inside each fold (no separate cal set;
    raw probabilities feed the meta-learner directly).
  - LogReg meta-learner with mild L2 — robust to small validation sets and to
    correlated base predictions.
  - 5 diverse base learners, four on gene-aggregated tabular features and one
    on raw chi²-prefiltered SNPs (different feature space = real diversity).
  - Per-fold chi² prefilter inside the raw-SNP branch (leak-free, recomputed
    on each fold's fit partition).
  - No per-fold deep-learning retraining (the saved GeneAttentionNet adds
    little here and would cost hours).

Targets:
  - Beat XGB gene-agg (0.830 avg PR-AUC) and the prior LogReg stack (0.843).
  - Maintain K-fold OOF, multiple bases, bootstrap CIs — the structural rigor.
================================================================================
"""
from __future__ import annotations

import gc
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.gene_aggregated_features import build_gene_feature_matrix_from_pkl
from src.experiments._shared import (
    bootstrap_mcc, bootstrap_pr_auc, compute_metrics,
)
from src.utils.seeds import set_global_seed

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

warnings.filterwarnings("ignore")


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def _spw(y: np.ndarray) -> float:
    pos = float((y == 1).sum()); neg = float((y == 0).sum())
    return max(1.0, neg / max(pos, 1.0))


def _variance_chunked(X_int8: np.ndarray, chunk: int = 10_000) -> np.ndarray:
    n = X_int8.shape[1]
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = X_int8[:, s:e].astype(np.float32).var(axis=0)
    return out


def _chi2_chunked(X_int8: np.ndarray, y: np.ndarray, chunk: int = 10_000) -> np.ndarray:
    n = X_int8.shape[1]
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        c, _ = chi2(X_int8[:, s:e].astype(np.float32), y)
        out[s:e] = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _best_threshold_mcc(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_m = 0.5, -1.0
    for t in thresholds:
        pred = (p >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        m = matthews_corrcoef(y, pred)
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)


# ───────────────────────────────────────────────────────────────────────────────
# Base learner trainers
# ───────────────────────────────────────────────────────────────────────────────
def _train_xgb_gene(X_fit, y_fit, X_es, y_es, params):
    p = dict(params)
    p["scale_pos_weight"] = _spw(y_fit)
    p["objective"] = "binary:logistic"
    p["eval_metric"] = "aucpr"
    early = p.pop("early_stopping_rounds", 150)
    n_round = p.pop("n_estimators", 1500)
    p.pop("n_jobs", None)
    dtr = xgb.DMatrix(X_fit, label=y_fit)
    des = xgb.DMatrix(X_es, label=y_es)
    return xgb.train(p, dtr, num_boost_round=n_round,
                     evals=[(des, "es")], early_stopping_rounds=early,
                     verbose_eval=False)


def _train_lgbm_gene(X_fit, y_fit, X_es, y_es, params):
    p = dict(params)
    p["scale_pos_weight"] = _spw(y_fit)
    p["random_state"] = 42
    p["verbose"] = -1
    p["n_jobs"] = -1
    early = p.pop("early_stopping_rounds", 150)
    n_round = p.pop("n_estimators", 1500)
    clf = lgb.LGBMClassifier(n_estimators=n_round, **p)
    clf.fit(X_fit, y_fit, eval_set=[(X_es, y_es)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(early, verbose=False),
                       lgb.log_evaluation(-1)])
    return clf


def _train_rf_gene(X_fit, y_fit, params):
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=int(params.get("max_depth", 12)),
        min_samples_leaf=int(params.get("min_samples_leaf", 10)),
        max_features="sqrt",                  # ~141 cols/split for 20K → ~5× faster
        class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    ).fit(X_fit, y_fit)


def _train_logreg_gene(X_fit, y_fit):
    sc = StandardScaler().fit(X_fit)
    clf = LogisticRegression(
        C=0.1, penalty="l2", solver="lbfgs",
        max_iter=3000, class_weight="balanced", random_state=42,
    ).fit(sc.transform(X_fit), y_fit)
    return clf, sc


def _train_xgb_raw_chi2(X_fit_snp, y_fit, X_es_snp, y_es,
                         prefilter_k, min_var, params):
    """Per-fold chi² prefilter on the SNP matrix, then XGB on top-K features."""
    var = _variance_chunked(X_fit_snp)
    keep_var = np.where(var > min_var)[0]
    X_var = X_fit_snp[:, keep_var]
    chi2_stats = _chi2_chunked(X_var, y_fit)
    k = int(min(prefilter_k, len(keep_var)))
    top_local = np.argpartition(chi2_stats, -k)[-k:]
    top_idx = keep_var[top_local]
    del X_var, chi2_stats, var; gc.collect()

    X_fit_top = X_fit_snp[:, top_idx].astype(np.float32)
    X_es_top = X_es_snp[:, top_idx].astype(np.float32)

    p = dict(params)
    p["scale_pos_weight"] = _spw(y_fit)
    p["objective"] = "binary:logistic"
    p["eval_metric"] = "aucpr"
    p.setdefault("colsample_bytree", 1.0)
    p.setdefault("colsample_bynode", 0.5)
    early = p.pop("early_stopping_rounds", 150)
    n_round = p.pop("n_estimators", 1500)
    p.pop("n_jobs", None)

    dtr = xgb.DMatrix(X_fit_top, label=y_fit)
    des = xgb.DMatrix(X_es_top, label=y_es)
    booster = xgb.train(p, dtr, num_boost_round=n_round,
                        evals=[(des, "es")], early_stopping_rounds=early,
                        verbose_eval=False)
    del dtr, des; gc.collect()
    return booster, top_idx


# ───────────────────────────────────────────────────────────────────────────────
# Per-antibiotic K-fold stacking
# ───────────────────────────────────────────────────────────────────────────────
def _run_one_antibiotic(
    X_flat, X_snp_int8, Y, label_mask,
    train_idx, val_idx, test_idx,
    ab_idx, ab_name, exp_cfg,
):
    n_folds = int(exp_cfg.n_folds)
    use_lgbm = bool(exp_cfg.use_lgbm) and HAS_LGB

    m = label_mask[:, ab_idx]
    tr = train_idx[m[train_idx]]
    va = val_idx[m[val_idx]]
    te = test_idx[m[test_idx]]

    y_tr = Y[tr, ab_idx].astype(np.int32)
    y_va = Y[va, ab_idx].astype(np.int32)
    y_te = Y[te, ab_idx].astype(np.int32)

    X_tr_flat = X_flat[tr]; X_va_flat = X_flat[va]; X_te_flat = X_flat[te]
    X_tr_snp = X_snp_int8[tr]; X_va_snp = X_snp_int8[va]; X_te_snp = X_snp_int8[te]

    print(f"\n  -- {ab_name} --")
    print(f"      Train={len(y_tr)} (pos={int(y_tr.sum())}, "
          f"{100*y_tr.mean():.1f}%)  Val={len(y_va)}  Test={len(y_te)}")

    base_names = ["XGB-gene"]
    if use_lgbm:
        base_names.append("LGBM-gene")
    base_names += ["RF-gene", "LogReg-gene", "XGB-raw-chi2"]
    n_base = len(base_names)

    oof_preds = np.zeros((len(y_tr), n_base), dtype=np.float32)
    val_acc = np.zeros((len(y_va), n_base), dtype=np.float32)
    test_acc = np.zeros((len(y_te), n_base), dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (fit_rel, oof_rel) in enumerate(skf.split(X_tr_flat, y_tr)):
        t_fold = time.time()
        # 90/10 fit/ES inside fold-train
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42 + fold)
        rel_fit, rel_es = next(sss.split(X_tr_flat[fit_rel], y_tr[fit_rel]))
        fit_idx = fit_rel[rel_fit]; es_idx = fit_rel[rel_es]
        X_fit, y_fit = X_tr_flat[fit_idx], y_tr[fit_idx]
        X_es, y_es = X_tr_flat[es_idx], y_tr[es_idx]
        X_fit_snp = X_tr_snp[fit_idx]; X_es_snp = X_tr_snp[es_idx]

        col = 0

        # 1. XGB on gene-aggregated
        t0 = time.time()
        b = _train_xgb_gene(X_fit, y_fit, X_es, y_es, dict(exp_cfg.xgb_params))
        oof_preds[oof_rel, col] = b.predict(xgb.DMatrix(X_tr_flat[oof_rel]))
        val_acc[:, col] += b.predict(xgb.DMatrix(X_va_flat)) / n_folds
        test_acc[:, col] += b.predict(xgb.DMatrix(X_te_flat)) / n_folds
        del b; gc.collect()
        t_xgb = time.time() - t0
        col += 1

        # 2. LightGBM on gene-aggregated
        if use_lgbm:
            t0 = time.time()
            clf = _train_lgbm_gene(X_fit, y_fit, X_es, y_es, dict(exp_cfg.lgbm_params))
            oof_preds[oof_rel, col] = clf.predict_proba(X_tr_flat[oof_rel])[:, 1]
            val_acc[:, col] += clf.predict_proba(X_va_flat)[:, 1] / n_folds
            test_acc[:, col] += clf.predict_proba(X_te_flat)[:, 1] / n_folds
            del clf; gc.collect()
            t_lgbm = time.time() - t0
            col += 1
        else:
            t_lgbm = 0.0

        # 3. RF on gene-aggregated (sqrt features for speed)
        t0 = time.time()
        clf = _train_rf_gene(X_fit, y_fit, dict(exp_cfg.rf_params))
        oof_preds[oof_rel, col] = clf.predict_proba(X_tr_flat[oof_rel])[:, 1]
        val_acc[:, col] += clf.predict_proba(X_va_flat)[:, 1] / n_folds
        test_acc[:, col] += clf.predict_proba(X_te_flat)[:, 1] / n_folds
        del clf; gc.collect()
        t_rf = time.time() - t0
        col += 1

        # 4. LogReg (scaled)
        t0 = time.time()
        clf, sc = _train_logreg_gene(X_fit, y_fit)
        oof_preds[oof_rel, col] = clf.predict_proba(sc.transform(X_tr_flat[oof_rel]))[:, 1]
        val_acc[:, col] += clf.predict_proba(sc.transform(X_va_flat))[:, 1] / n_folds
        test_acc[:, col] += clf.predict_proba(sc.transform(X_te_flat))[:, 1] / n_folds
        del clf, sc; gc.collect()
        t_lr = time.time() - t0
        col += 1

        # 5. XGB on raw chi²-prefiltered SNPs (per-fold prefilter)
        t0 = time.time()
        b, top_idx = _train_xgb_raw_chi2(
            X_fit_snp, y_fit, X_es_snp, y_es,
            prefilter_k=int(exp_cfg.prefilter_top_k),
            min_var=float(exp_cfg.prefilter_min_variance),
            params=dict(exp_cfg.xgb_raw_params),
        )
        X_oof_top = X_tr_snp[oof_rel][:, top_idx].astype(np.float32)
        X_va_top = X_va_snp[:, top_idx].astype(np.float32)
        X_te_top = X_te_snp[:, top_idx].astype(np.float32)
        oof_preds[oof_rel, col] = b.predict(xgb.DMatrix(X_oof_top))
        val_acc[:, col] += b.predict(xgb.DMatrix(X_va_top)) / n_folds
        test_acc[:, col] += b.predict(xgb.DMatrix(X_te_top)) / n_folds
        del b, X_oof_top, X_va_top, X_te_top; gc.collect()
        t_raw = time.time() - t0
        col += 1

        print(f"      Fold {fold+1}/{n_folds}  fit={len(fit_idx)} es={len(es_idx)} "
              f"oof={len(oof_rel)}  "
              f"xgb={t_xgb:.0f}s lgbm={t_lgbm:.0f}s rf={t_rf:.0f}s "
              f"lr={t_lr:.0f}s raw={t_raw:.0f}s  total={time.time() - t_fold:.0f}s")

    # Diagnostics
    print(f"\n      Base diagnostics  (OOF / Val / Test):")
    for i, bn in enumerate(base_names):
        oof_pr = float(average_precision_score(y_tr, oof_preds[:, i]))
        val_pr = float(average_precision_score(y_va, val_acc[:, i]))
        te_pr = float(average_precision_score(y_te, test_acc[:, i]))
        print(f"        {bn:<14}  OOF={oof_pr:.4f}  Val={val_pr:.4f}  Test={te_pr:.4f}")

    # Meta-learner: LogReg with mild L2 — robust to correlated bases
    meta_C = float(exp_cfg.meta_C)
    print(f"\n      Meta-learner: LogReg C={meta_C}")
    meta = LogisticRegression(
        C=meta_C, penalty="l2", solver="lbfgs",
        max_iter=2000, class_weight="balanced", random_state=42,
    ).fit(oof_preds, y_tr)
    meta_coefs = meta.coef_.ravel().tolist()
    print(f"        weights: " +
          ", ".join(f"{bn}={w:+.3f}" for bn, w in zip(base_names, meta_coefs)))

    val_meta = meta.predict_proba(val_acc)[:, 1]
    test_meta = meta.predict_proba(test_acc)[:, 1]

    thr, _ = _best_threshold_mcc(y_va, val_meta)
    metrics = compute_metrics(y_te, test_meta, threshold=thr)
    boot_n = int(exp_cfg.evaluation.bootstrap_n)
    boot_seed = int(exp_cfg.evaluation.bootstrap_seed)
    pr_lo, pr_hi = bootstrap_pr_auc(y_te, test_meta, n_boot=boot_n, seed=boot_seed)
    mcc_lo, mcc_hi = bootstrap_mcc(y_te, test_meta, n_boot=boot_n, seed=boot_seed)

    solo_test_prs = [float(average_precision_score(y_te, test_acc[:, i]))
                     for i in range(n_base)]
    best_solo_te = max(solo_test_prs)
    delta = metrics["pr_auc"] - best_solo_te
    print(f"\n      -- TEST RESULT --")
    print(f"      Ensemble PR-AUC: {metrics['pr_auc']:.4f} "
          f"[{pr_lo:.4f}, {pr_hi:.4f}]")
    print(f"      Best solo      : {best_solo_te:.4f}  "
          f"(delta_vs_best_solo: {delta:+.4f})")
    print(f"      MCC: {metrics['mcc']:.4f}  F2: {metrics['f2']:.4f}  "
          f"Sens@90: {metrics['sens_at_90_spec']:.4f}")

    return {
        "antibiotic": ab_name,
        "ensemble_pr_auc": metrics["pr_auc"],
        "ensemble_pr_auc_lo": pr_lo, "ensemble_pr_auc_hi": pr_hi,
        "ensemble_mcc": metrics["mcc"],
        "ensemble_mcc_lo": mcc_lo, "ensemble_mcc_hi": mcc_hi,
        "ensemble_f2": metrics["f2"],
        "ensemble_roc_auc": metrics["roc_auc"],
        "ensemble_sens_at_90_spec": metrics["sens_at_90_spec"],
        "best_solo_pr_auc": best_solo_te,
        "delta_vs_best_solo": delta,
        "threshold": thr,
        "n_test": len(y_te),
        **{f"solo_{bn}_test_pr_auc": v for bn, v in zip(base_names, solo_test_prs)},
        **{f"meta_w_{bn}": w for bn, w in zip(base_names, meta_coefs)},
    }, test_meta, base_names


# ───────────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────────
def run(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    exp_cfg = cfg.experiment
    out_dir = Path(exp_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config_used.yaml")

    print("=" * 80)
    print("  CascadeStack v8 — K-fold stacking, LogReg meta")
    print("=" * 80)
    print(f"  LightGBM: {'YES' if HAS_LGB else 'NO (skipping base)'}")
    print(f"  Output  : {out_dir}")

    print("\n[1/3] Loading gene-aggregated features + filtered SNP matrix...")
    pkl_path = (str(cfg.paths.processed_dir).rstrip("/\\") +
                "/arcadia_v2_gene_filtered.pkl")
    bundle = build_gene_feature_matrix_from_pkl(
        gene_filtered_pkl_path=pkl_path,
        stats=tuple(exp_cfg.gene_stats),
    )
    X_flat = bundle["X_gene_features"].astype(np.float32)
    Y = bundle["Y"].astype(np.int32)
    label_mask = bundle["label_mask"].astype(bool)
    label_names = [str(x) for x in bundle["label_names"]]
    splits = bundle["random_split"]
    tr_idx = np.asarray(splits["train_idx"])
    va_idx = np.asarray(splits["val_idx"])
    te_idx = np.asarray(splits["test_idx"])

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    X_snp_int8 = d["X_snp_filtered"]
    print(f"  X_flat: {X_flat.shape}, X_snp_int8: {X_snp_int8.shape}")

    print("\n[2/3] Running K-fold stacking per antibiotic...")
    all_rows = []
    all_test_preds: dict[str, np.ndarray] = {}
    base_names_final: list[str] = []
    for ab in exp_cfg.antibiotics:
        if ab not in label_names:
            print(f"  SKIP {ab} (not in labels)")
            continue
        ab_idx = label_names.index(ab)
        row, test_meta, base_names_final = _run_one_antibiotic(
            X_flat, X_snp_int8, Y, label_mask,
            tr_idx, va_idx, te_idx, ab_idx, ab, exp_cfg,
        )
        all_rows.append(row)
        all_test_preds[ab] = test_meta

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "v8_results.csv", index=False, encoding="utf-8")
    np.savez_compressed(
        out_dir / "v8_test_predictions.npz",
        **{f"probs_{ab}": p for ab, p in all_test_preds.items()},
        antibiotics=np.array(list(all_test_preds.keys())),
    )

    print("\n[3/3] Writing summary...")
    avg_ens = float(df["ensemble_pr_auc"].mean()) if len(df) else 0.0
    avg_solo = float(df["best_solo_pr_auc"].mean()) if len(df) else 0.0
    avg_delta = float(df["delta_vs_best_solo"].mean()) if len(df) else 0.0
    avg_mcc = float(df["ensemble_mcc"].mean()) if len(df) else 0.0

    lines = [
        "# CascadeStack v8 - K-fold stacking results\n",
        "## Average across antibiotics\n",
        f"- Ensemble PR-AUC : **{avg_ens:.4f}**",
        f"- Best solo PR-AUC: {avg_solo:.4f}",
        f"- Delta (ens-solo): **{avg_delta:+.4f}**",
        f"- Avg MCC         : {avg_mcc:.4f}\n",
        "## Per-antibiotic\n",
        "| AB | Ensemble PR-AUC | 95% CI | Best Solo | Delta | MCC | F2 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['antibiotic']} | {r['ensemble_pr_auc']:.4f} "
            f"| [{r['ensemble_pr_auc_lo']:.4f}, {r['ensemble_pr_auc_hi']:.4f}] "
            f"| {r['best_solo_pr_auc']:.4f} "
            f"| {r['delta_vs_best_solo']:+.4f} "
            f"| {r['ensemble_mcc']:.4f} | {r['ensemble_f2']:.4f} |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"  Avg ensemble PR-AUC : {avg_ens:.4f}")
    print(f"  Avg best solo       : {avg_solo:.4f}")
    print(f"  Avg delta           : {avg_delta:+.4f}")
    print(f"  Outputs -> {out_dir}")
    print("=" * 80)
