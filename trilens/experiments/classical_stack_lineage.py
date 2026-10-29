from __future__ import annotations

import gc
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from trilens.data.gene_statistics import build_gene_feature_matrix_from_pkl
from trilens.evaluation.training_helpers import bootstrap_mcc, bootstrap_pr_auc, compute_metrics
from trilens.experiments.classical_stack import _best_threshold_mcc, _run_one_antibiotic
from trilens.utils.seeds import set_global_seed

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered__mlst_holdout.pkl"
SPLIT_KEY = "mlst_lineage_holdout_split"
REQUIRE_ST_DISJOINT = True
OUTPUT_DIR = RESULTS_DIR / "classical_stack_lineage"

ANTIBIOTICS = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]
GENE_STATS = ["mean", "var", "n_nonref", "max"]
SEED = 42

XGB_PARAMS = dict(
    n_estimators=1500,
    max_depth=5,
    min_child_weight=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0.1,
    tree_method="hist",
    device="cuda",
    early_stopping_rounds=150,
)
LGBM_PARAMS = dict(
    num_leaves=63,
    max_depth=6,
    min_child_samples=20,
    learning_rate=0.05,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=150,
)
RF_PARAMS = dict(n_estimators=200, max_depth=12, min_samples_leaf=10)
XGB_RAW_PARAMS = dict(
    n_estimators=1500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=1.0,
    colsample_bynode=0.5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    device="cuda",
    early_stopping_rounds=150,
)
PREFILTER_TOP_K = 50_000
PREFILTER_MIN_VARIANCE = 1e-6
N_FOLDS = 5
META_C = 1.0
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def main() -> None:
    set_global_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 3 -- 5 classical bases (no GAN) on the MLST LINEAGE HOLDOUT split")
    print("=" * 80)

    if not GENE_FILTERED_PKL.exists():
        raise FileNotFoundError(
            f"{GENE_FILTERED_PKL} not found -- run trilens.experiments.gene_filtered_dataset_lineage first."
        )

    bundle = build_gene_feature_matrix_from_pkl(str(GENE_FILTERED_PKL), stats=tuple(GENE_STATS))
    X_flat = bundle["X_gene_features"].astype(np.float32)
    Y = bundle["Y"].astype(np.int32)
    label_mask = bundle["label_mask"].astype(bool)
    label_names = [str(x) for x in bundle["label_names"]]

    with open(GENE_FILTERED_PKL, "rb") as f:
        d = pickle.load(f)
    X_snp_int8 = d["X_snp_filtered"]
    splits = d[SPLIT_KEY]
    tr_idx = np.asarray(splits["train_idx"])
    va_idx = np.asarray(splits["val_idx"])
    te_idx = np.asarray(splits["test_idx"])

    st = np.asarray(d["st_per_genome"], dtype=object)
    st_tr = {st[i] for i in tr_idx if st[i] is not None}
    st_te = {st[i] for i in te_idx if st[i] is not None}
    st_va = {st[i] for i in va_idx if st[i] is not None}
    if REQUIRE_ST_DISJOINT:
        assert not (st_tr & st_te), "TRAIN/TEST ST LEAK"
        assert not (st_tr & st_va), "TRAIN/VAL ST LEAK"
        assert not (st_va & st_te), "VAL/TEST ST LEAK"
        print(f"  ST disjointness re-verified. train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")
    assert not (set(tr_idx) & set(te_idx)) and not (set(tr_idx) & set(va_idx)) and not (set(va_idx) & set(te_idx)), (
        "partitions overlap"
    )
    print(f"  X_flat: {X_flat.shape}, X_snp_int8: {X_snp_int8.shape}")

    exp_cfg = SimpleNamespace(
        n_folds=N_FOLDS,
        use_lgbm=True,
        xgb_params=XGB_PARAMS,
        lgbm_params=LGBM_PARAMS,
        rf_params=RF_PARAMS,
        xgb_raw_params=XGB_RAW_PARAMS,
        prefilter_top_k=PREFILTER_TOP_K,
        prefilter_min_variance=PREFILTER_MIN_VARIANCE,
        meta_C=META_C,
        evaluation=SimpleNamespace(bootstrap_n=BOOTSTRAP_N, bootstrap_seed=BOOTSTRAP_SEED),
    )

    n_ab = len(ANTIBIOTICS)

    n_cols = len(label_names)
    gan_oof_dummy = np.zeros((len(tr_idx), n_cols), dtype=np.float32)
    gan_val_dummy = np.zeros((len(va_idx), n_cols), dtype=np.float32)
    gan_test_dummy = np.zeros((len(te_idx), n_cols), dtype=np.float32)

    all_rows = []
    all_test_preds: dict[str, np.ndarray] = {}
    _oof_list, _val_list, _test_list = {}, {}, {}
    _tr_pos, _va_pos, _te_pos = {}, {}, {}
    base_names_5 = None

    for k, ab in enumerate(ANTIBIOTICS):
        if ab not in label_names:
            print(f"  SKIP {ab}")
            continue
        ab_idx = label_names.index(ab)
        print(f"\n[{k + 1}/{n_ab}] {ab}")

        (_row6, _tm6, base_names6, oof_p6, val_p6, test_p6, tr_pos, va_pos, te_pos) = _run_one_antibiotic(
            X_flat,
            X_snp_int8,
            Y,
            label_mask,
            gan_oof_dummy,
            gan_val_dummy,
            gan_test_dummy,
            tr_idx,
            va_idx,
            te_idx,
            ab_idx,
            ab,
            exp_cfg,
        )
        base_names_5 = base_names6[:5]
        oof_p5, val_p5, test_p5 = oof_p6[:, :5], val_p6[:, :5], test_p6[:, :5]

        tr, va, te = tr_idx[tr_pos], va_idx[va_pos], te_idx[te_pos]
        y_tr, y_va, y_te = Y[tr, ab_idx], Y[va, ab_idx], Y[te, ab_idx]

        meta = LogisticRegression(
            C=META_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ).fit(oof_p5, y_tr)
        meta_coefs = meta.coef_.ravel().tolist()
        val_meta = meta.predict_proba(val_p5)[:, 1]
        test_meta = meta.predict_proba(test_p5)[:, 1]

        thr, _ = _best_threshold_mcc(y_va, val_meta)
        metrics = compute_metrics(y_te, test_meta, threshold=thr)
        pr_lo, pr_hi = bootstrap_pr_auc(y_te, test_meta, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        mcc_lo, mcc_hi = bootstrap_mcc(y_te, test_meta, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        solo = [float(average_precision_score(y_te, test_p5[:, i])) for i in range(5)]

        print(
            f"  5-base classical ensemble PR-AUC: {metrics['pr_auc']:.4f} "
            f"[{pr_lo:.4f}, {pr_hi:.4f}]  best_solo={max(solo):.4f}"
        )

        all_rows.append(
            {
                "antibiotic": ab,
                "ensemble_pr_auc": metrics["pr_auc"],
                "ensemble_pr_auc_lo": pr_lo,
                "ensemble_pr_auc_hi": pr_hi,
                "ensemble_mcc": metrics["mcc"],
                "ensemble_mcc_lo": mcc_lo,
                "ensemble_mcc_hi": mcc_hi,
                "ensemble_f2": metrics["f2"],
                "best_solo_pr_auc": max(solo),
                "threshold": thr,
                "n_test": len(y_te),
                **{f"solo_{bn}_test_pr_auc": v for bn, v in zip(base_names_5, solo)},
                **{f"meta_w_{bn}": w for bn, w in zip(base_names_5, meta_coefs)},
            }
        )
        all_test_preds[ab] = test_meta
        _oof_list[ab], _val_list[ab], _test_list[ab] = oof_p5, val_p5, test_p5
        _tr_pos[ab], _va_pos[ab], _te_pos[ab] = tr_pos, va_pos, te_pos
        gc.collect()

    ab_list = list(all_test_preds.keys())
    n_abs, n_base = len(ab_list), 5
    oof_base = np.zeros((len(tr_idx), n_base, n_abs), dtype=np.float32)
    val_base = np.zeros((len(va_idx), n_base, n_abs), dtype=np.float32)
    test_base = np.zeros((len(te_idx), n_base, n_abs), dtype=np.float32)
    for i, ab in enumerate(ab_list):
        oof_base[_tr_pos[ab], :, i] = _oof_list[ab]
        val_base[_va_pos[ab], :, i] = _val_list[ab]
        test_base[_te_pos[ab], :, i] = _test_list[ab]
    test_ens = np.zeros((len(te_idx), n_abs), dtype=np.float32)
    for i, ab in enumerate(ab_list):
        test_ens[_te_pos[ab], i] = all_test_preds[ab]

    np.savez_compressed(
        str(OUTPUT_DIR / "predictions_saved.npz"),
        oof_base_probs=oof_base,
        test_base_probs=test_base,
        val_base_probs=val_base,
        test_ens_probs=test_ens,
        test_indices=te_idx.astype(np.int64),
        val_indices=va_idx.astype(np.int64),
        train_indices=tr_idx.astype(np.int64),
        base_names=np.array(base_names_5),
        antibiotic_names=np.array(ab_list),
    )
    print(f"\nSaved predictions_saved.npz -> {OUTPUT_DIR}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "classical_stack_results.csv", index=False)
    avg = float(df["ensemble_pr_auc"].mean())
    print(f"\nAverage 5-base classical ensemble PR-AUC: {avg:.4f}")
    print(df[["antibiotic", "ensemble_pr_auc", "ensemble_mcc"]].to_string(index=False))

    lines = [
        "# Classical stack, five base learners, lineage holdout split\n",
        f"**Average ensemble PR-AUC: {avg:.4f}**\n",
        "No GeneAttentionNet. Single seed=42. Gene set ranked on this split's own train.\n",
        "| AB | Ensemble PR-AUC | 95% CI | Best Solo | MCC |",
        "|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['antibiotic']} | {r['ensemble_pr_auc']:.4f} | "
            f"[{r['ensemble_pr_auc_lo']:.4f}, {r['ensemble_pr_auc_hi']:.4f}] | "
            f"{r['best_solo_pr_auc']:.4f} | {r['ensemble_mcc']:.4f} |"
        )
    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSTEP 3 COMPLETE -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
