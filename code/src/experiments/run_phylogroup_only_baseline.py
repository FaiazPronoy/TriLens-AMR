# src/experiments/run_phylogroup_only_baseline.py
"""Phylogroup-only baseline: how much can 8-dim phylogroup one-hot alone predict?

Measures the population-structure signal in isolation (no PCA / gene features),
using the same random_split as the rest of Pillar 1/2/... For each antibiotic,
trains a LogisticRegression and an XGBoost classifier on the (6962 x 8)
phylogroup one-hot matrix and reports test PR-AUC.

Outputs:
    results/pillar_phylogroup_only/phylogroup_only_results.csv
    results/pillar_phylogroup_only/SUMMARY.md
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_PKL = PROJECT_ROOT / "data" / "processed" / "arcadia_v2_master_features.pkl"
OUTPUT_DIR = PROJECT_ROOT / "results" / "pillar_phylogroup_only"

SEED = 42


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MASTER_PKL, "rb") as f:
        master = pickle.load(f)

    phylogroup_oh = master["phylogroup_onehot"].astype(np.float32)
    Y = master["Y"].astype(np.float32)
    label_mask = master["label_mask"].astype(bool)
    label_names = [str(n) for n in master["label_names"]]

    rs = master["random_split"]
    train_idx = np.asarray(rs["train_idx"])
    val_idx = np.asarray(rs["val_idx"])
    test_idx = np.asarray(rs["test_idx"])

    rows: list[dict] = []

    for ab in label_names:
        ab_idx = label_names.index(ab)
        mask = label_mask[:, ab_idx]

        def _get(idx):
            m = mask[idx]
            valid = idx[m]
            return phylogroup_oh[valid], Y[valid, ab_idx].astype(np.int32)

        Xtr, ytr = _get(train_idx)
        Xva, yva = _get(val_idx)
        Xte, yte = _get(test_idx)

        # Combine train+val for the linear model (no early stopping needed).
        Xtr_full = np.concatenate([Xtr, Xva], axis=0)
        ytr_full = np.concatenate([ytr, yva], axis=0)

        print("=" * 70)
        print(f"  PHYLOGROUP-ONLY BASELINE — {ab}  "
              f"(n_train={len(ytr_full)}, n_test={len(yte)}, "
              f"n_pos_test={int(yte.sum())})")
        print("=" * 70)

        # --- LogisticRegression ---
        try:
            lr = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=SEED,
            )
            lr.fit(Xtr_full, ytr_full)
            p_lr = lr.predict_proba(Xte)[:, 1]
            pr_auc_lr = safe_pr_auc(yte, p_lr)
        except Exception as ex:  # noqa: BLE001
            print(f"  [WARN] LogisticRegression failed for {ab}: {ex!r}")
            pr_auc_lr = float("nan")

        # --- XGBoost ---
        try:
            pos = float((ytr_full == 1).sum())
            neg = float((ytr_full == 0).sum())
            scale_pos_weight = max(1.0, neg / max(1.0, pos))
            xgb_clf = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_weight=5,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                random_state=SEED,
                verbosity=0,
                tree_method="hist",
            )
            xgb_clf.fit(Xtr_full, ytr_full)
            p_xgb = xgb_clf.predict_proba(Xte)[:, 1]
            pr_auc_xgb = safe_pr_auc(yte, p_xgb)
        except Exception as ex:  # noqa: BLE001
            print(f"  [WARN] XGBoost failed for {ab}: {ex!r}")
            pr_auc_xgb = float("nan")

        print(f"  LogisticRegression PR-AUC: {pr_auc_lr:.4f}")
        print(f"  XGBoost            PR-AUC: {pr_auc_xgb:.4f}")

        rows.append({
            "antibiotic": ab,
            "n_train": int(len(ytr_full)),
            "n_test": int(len(yte)),
            "n_pos_test": int(yte.sum()),
            "logreg_pr_auc": pr_auc_lr,
            "xgboost_pr_auc": pr_auc_xgb,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "phylogroup_only_results.csv", index=False)

    avg_lr = df["logreg_pr_auc"].mean()
    avg_xgb = df["xgboost_pr_auc"].mean()

    lines = [
        "# Phylogroup-Only Baseline — Experiment Summary",
        "",
        "Features: `phylogroup_onehot` only (6962 x 8), random_split, seed=42.",
        "",
        "| Antibiotic | n_test | n_pos_test | LogReg PR-AUC | XGBoost PR-AUC |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['antibiotic']} | {r['n_test']} | {r['n_pos_test']} | "
            f"{r['logreg_pr_auc']:.4f} | {r['xgboost_pr_auc']:.4f} |"
        )
    lines += [
        "",
        f"**Average LogReg PR-AUC:** {avg_lr:.4f}",
        f"**Average XGBoost PR-AUC:** {avg_xgb:.4f}",
        "",
        "This isolates how much predictive signal comes from population "
        "structure (phylogroup membership) alone, with no genomic feature "
        "content whatsoever.",
    ]
    with open(OUTPUT_DIR / "SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "=" * 70)
    print(f"  Average LogReg  PR-AUC: {avg_lr:.4f}")
    print(f"  Average XGBoost PR-AUC: {avg_xgb:.4f}")
    print(f"  Outputs -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
