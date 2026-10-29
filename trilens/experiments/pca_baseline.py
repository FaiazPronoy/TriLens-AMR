from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from scipy.stats import wilcoxon

from trilens.evaluation.training_helpers import (
    bootstrap_mcc,
    bootstrap_pr_auc,
    compute_metrics,
    pos_weight_from_labels,
)
from trilens.utils.seeds import set_global_seed


def _load_master_with_phylogroup(master_pkl_path: str | Path) -> dict:
    with open(master_pkl_path, "rb") as f:
        master = pickle.load(f)

    rs = master["random_split"]
    train_idx = np.asarray(rs["train_idx"])
    val_idx = np.asarray(rs["val_idx"])
    test_idx = np.asarray(rs["test_idx"])
    n_total = max(train_idx.max(), val_idx.max(), test_idx.max()) + 1

    n_feat = master["X_pca_train"].shape[1]
    X = np.zeros((n_total, n_feat), dtype=np.float32)
    X[train_idx] = master["X_pca_train"].astype(np.float32)
    X[val_idx] = master["X_pca_val"].astype(np.float32)
    X[test_idx] = master["X_pca_test"].astype(np.float32)

    phylogroup_oh = master["phylogroup_onehot"].astype(np.float32)

    return {
        "X_pca": X,
        "phylogroup_oh": phylogroup_oh,
        "Y": master["Y"].astype(np.float32),
        "label_mask": master["label_mask"].astype(bool),
        "label_names": [str(n) for n in master["label_names"]],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def _slice_for_ab(d: dict, ab: str, with_phylogroup: bool):
    ab_idx = d["label_names"].index(ab)
    mask = d["label_mask"][:, ab_idx]

    if with_phylogroup:
        X_full = np.concatenate([d["X_pca"], d["phylogroup_oh"]], axis=1)
    else:
        X_full = d["X_pca"]

    def _get(idx):
        m = mask[idx]
        valid = idx[m]
        return X_full[valid], d["Y"][valid, ab_idx].astype(np.int32)

    Xtr, ytr = _get(d["train_idx"])
    Xva, yva = _get(d["val_idx"])
    Xte, yte = _get(d["test_idx"])
    return Xtr, ytr, Xva, yva, Xte, yte


def _train_xgb(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    cfg: DictConfig,
) -> xgb.XGBClassifier:
    pw = pos_weight_from_labels(ytr.astype(np.float32))
    params = dict(cfg.experiment.xgb_params)
    early_stop = params.pop("early_stopping_rounds", 200)
    eval_metric = params.pop("eval_metric", "aucpr")

    try:
        clf = xgb.XGBClassifier(
            **params,
            scale_pos_weight=pw,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stop,
            random_state=42,
            verbosity=0,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    except Exception as ex:
        print(f"  [WARN] GPU XGBoost failed ({ex!r}); falling back to CPU.")
        params["device"] = "cpu"
        clf = xgb.XGBClassifier(
            **params,
            scale_pos_weight=pw,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stop,
            random_state=42,
            verbosity=0,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return clf


def _eval_xgb(clf: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray, cfg: DictConfig) -> dict:
    p = clf.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, p, threshold=0.5)
    pr_ci = bootstrap_pr_auc(
        y,
        p,
        n_boot=cfg.experiment.evaluation.bootstrap_n,
        seed=cfg.experiment.evaluation.bootstrap_seed,
    )
    mcc_ci = bootstrap_mcc(
        y,
        p,
        n_boot=cfg.experiment.evaluation.bootstrap_n,
        seed=cfg.experiment.evaluation.bootstrap_seed,
    )
    return {"metrics": metrics, "pr_auc_ci": pr_ci, "mcc_ci": mcc_ci, "y_true": y, "y_prob": p}


def run(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "pca_baseline_config_used.yaml")

    d = _load_master_with_phylogroup(cfg.paths.arcadia_master_pkl)

    rows: list[dict] = []
    preds: dict[str, dict] = {}

    for ab in cfg.experiment.antibiotics:
        for with_pg in [False, True]:
            tag = "with_phylogroup" if with_pg else "no_phylogroup"
            print("=" * 80)
            print(f"  XGB BASELINE - {ab} × {tag}")
            print("=" * 80)
            t0 = time.time()
            try:
                Xtr, ytr, Xva, yva, Xte, yte = _slice_for_ab(d, ab, with_pg)
                clf = _train_xgb(Xtr, ytr, Xva, yva, cfg)
                ev = _eval_xgb(clf, Xte, yte, cfg)
            except Exception as ex:
                print(f"  ERROR: {ex!r}")
                rows.append(
                    {
                        "antibiotic": ab,
                        "variant": tag,
                        "status": "error",
                        "error": repr(ex),
                    }
                )
                continue

            elapsed = time.time() - t0
            m = ev["metrics"]
            row = {
                "antibiotic": ab,
                "variant": tag,
                "status": "ok",
                "n_features": Xtr.shape[1],
                "n_train": int(len(ytr)),
                "n_pos_train": int(ytr.sum()),
                "n_test": int(len(yte)),
                "n_pos_test": int(yte.sum()),
                "test_pr_auc": m["pr_auc"],
                "test_pr_auc_lo": ev["pr_auc_ci"][0],
                "test_pr_auc_hi": ev["pr_auc_ci"][1],
                "test_mcc": m["mcc"],
                "test_mcc_lo": ev["mcc_ci"][0],
                "test_mcc_hi": ev["mcc_ci"][1],
                "test_roc_auc": m["roc_auc"],
                "test_f1": m["f1"],
                "test_f2": m["f2"],
                "test_brier": m["brier"],
                "test_sens_at_90_spec": m["sens_at_90_spec"],
                "fit_time_s": elapsed,
            }
            rows.append(row)

            preds[f"{ab}_{tag}"] = {
                "metrics": m,
                "pr_auc_ci": list(ev["pr_auc_ci"]),
                "mcc_ci": list(ev["mcc_ci"]),
            }
            pred_path = output_dir / f"xgb_{ab}_{tag}_predictions.npz"
            np.savez_compressed(pred_path, y_true=ev["y_true"], y_prob=ev["y_prob"])

            print(
                f"  PR-AUC: {m['pr_auc']:.4f}  MCC: {m['mcc']:.4f}  "
                f"F2: {m['f2']:.4f}  Sens@90: {m['sens_at_90_spec']:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "pca_baseline_results.csv", index=False)

    summary: dict = {"per_ab": preds}
    ok = df[df["status"] == "ok"]
    pivot_pr = ok.pivot_table(index="antibiotic", columns="variant", values="test_pr_auc")
    pivot_mcc = ok.pivot_table(index="antibiotic", columns="variant", values="test_mcc")

    print("\n" + "=" * 80)
    print("  XGB + PHYLOGROUP SUMMARY")
    print("=" * 80)
    if {"no_phylogroup", "with_phylogroup"}.issubset(set(pivot_pr.columns)):
        deltas_pr: list[float] = []
        deltas_mcc: list[float] = []
        for ab in pivot_pr.index:
            d_pr = pivot_pr.loc[ab, "with_phylogroup"] - pivot_pr.loc[ab, "no_phylogroup"]
            d_mcc = pivot_mcc.loc[ab, "with_phylogroup"] - pivot_mcc.loc[ab, "no_phylogroup"]
            deltas_pr.append(float(d_pr))
            deltas_mcc.append(float(d_mcc))
            print(f"  {ab:<6}  Δ PR-AUC = {d_pr:+.4f}    Δ MCC = {d_mcc:+.4f}")

        try:
            w_pr = wilcoxon(deltas_pr)
            w_mcc = wilcoxon(deltas_mcc)
            print(f"\n  Wilcoxon PR-AUC: p = {w_pr.pvalue:.4f}")
            print(f"  Wilcoxon MCC   : p = {w_mcc.pvalue:.4f}")
            summary["wilcoxon"] = {
                "pr_auc_p": float(w_pr.pvalue),
                "mcc_p": float(w_mcc.pvalue),
            }
        except Exception as ex:
            print(f"  [WARN] Wilcoxon failed: {ex!r}")

        signal_ok = float(np.mean(deltas_pr)) > 0
        verdict = "PHYLOGROUP SIGNAL" if signal_ok else "NO SIGNAL"
        print(f"\n  avg Δ PR-AUC = {float(np.mean(deltas_pr)):+.4f}  [{verdict}]")
        summary["avg_delta_pr_auc"] = float(np.mean(deltas_pr))
        summary["avg_delta_mcc"] = float(np.mean(deltas_mcc))

    with open(output_dir / "pca_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Outputs → {output_dir}")
