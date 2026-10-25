"""
src/experiments/run_tabnet_baseline.py
=======================================
TabNet baseline (Iftu Sir feedback) — a tabular deep-learning comparison
point, trained on the EXACT same gene-aggregated features (mean/var/n_nonref/
max per gene, 5000 genes -> 20,000 dims) and the same `random_split` used by
every other classical baseline in this project (stage2_gene_xgb,
CascadeStack v9's XGB-gene/LGBM-gene/RF-gene/LogReg-gene), so its per-
antibiotic PR-AUC is directly comparable — same feature set, same split, same
train/val/test partition.

Requires `pytorch-tabnet` (installed in the `amr` env for this task;
`pip install pytorch-tabnet` if missing elsewhere).

This script has NOT been run (per instructions — GPU job, to be run by the
student). Estimated runtime: ~5-15 minutes per antibiotic on an
RTX-3060/4070-class GPU (max_epochs=150, patience=20) -> roughly 35-100
minutes total for all 7 antibiotics; this is an estimate, not a measurement.

Usage:
    conda activate amr
    python -m src.experiments.run_tabnet_baseline
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

from src.data.gene_aggregated_features import build_gene_feature_matrix_from_pkl

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered.pkl"
CFG_PATH = ROOT / "configs" / "experiment" / "tabnet_baseline.yaml"
CONFIG_YAML_PATH = ROOT / "configs" / "config.yaml"


def _load_config():
    base_cfg = OmegaConf.load(CONFIG_YAML_PATH)
    if "hydra" in base_cfg:
        del base_cfg["hydra"]
    if "wandb" in base_cfg:
        del base_cfg["wandb"]
    tabnet_cfg = OmegaConf.load(CFG_PATH)
    cfg = OmegaConf.merge(base_cfg, tabnet_cfg)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))


def main() -> None:
    cfg = _load_config()
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading gene-aggregated features (same as CascadeStack v9 / stage2_gene_xgb)...")
    bundle = build_gene_feature_matrix_from_pkl(
        gene_filtered_pkl_path=str(GENE_FILTERED_PKL),
        stats=tuple(cfg.gene_stats),
    )
    X_flat = bundle["X_gene_features"].astype(np.float32)
    Y = bundle["Y"].astype(np.int64)
    label_mask = bundle["label_mask"].astype(bool)
    label_names = [str(x) for x in bundle["label_names"]]
    splits = bundle[str(cfg.split_key)]
    tr_idx = np.asarray(splits["train_idx"])
    va_idx = np.asarray(splits["val_idx"])
    te_idx = np.asarray(splits["test_idx"])

    rows = []
    t_start = time.time()
    for ab_idx, ab in enumerate(label_names):
        t_ab = time.time()
        m = label_mask[:, ab_idx]
        tr = tr_idx[m[tr_idx]]
        va = va_idx[m[va_idx]]
        te = te_idx[m[te_idx]]

        y_tr, y_va, y_te = Y[tr, ab_idx], Y[va, ab_idx], Y[te, ab_idx]
        X_tr, X_va, X_te = X_flat[tr], X_flat[va], X_flat[te]

        sc = StandardScaler().fit(X_tr)
        X_tr_s, X_va_s, X_te_s = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)

        print(f"\n-- {ab} --  train={len(y_tr)} val={len(y_va)} test={len(y_te)}")

        clf = TabNetClassifier(
            n_d=int(cfg.tabnet.n_d),
            n_a=int(cfg.tabnet.n_a),
            n_steps=int(cfg.tabnet.n_steps),
            gamma=float(cfg.tabnet.gamma),
            lambda_sparse=float(cfg.tabnet.lambda_sparse),
            seed=int(cfg.tabnet.seed),
            device_name=device,
        )
        clf.fit(
            X_tr_s, y_tr,
            eval_set=[(X_va_s, y_va)],
            eval_name=["val"],
            eval_metric=["auc"],
            max_epochs=int(cfg.training.max_epochs),
            patience=int(cfg.training.patience),
            batch_size=int(cfg.training.batch_size),
            virtual_batch_size=int(cfg.training.virtual_batch_size),
        )

        test_prob = clf.predict_proba(X_te_s)[:, 1]
        pr_auc = float(average_precision_score(y_te, test_prob))
        y_pred = (test_prob >= 0.5).astype(int)
        mcc = float(matthews_corrcoef(y_te, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0

        runtime_min = (time.time() - t_ab) / 60
        print(f"   PR-AUC={pr_auc:.4f}  MCC={mcc:.4f}  ({runtime_min:.1f} min)")
        rows.append({"antibiotic": ab, "pr_auc": pr_auc, "mcc": mcc, "runtime_min": round(runtime_min, 2)})

    df = pd.DataFrame(rows, columns=["antibiotic", "pr_auc", "mcc", "runtime_min"])
    df.to_csv(output_dir / "tabnet_results.csv", index=False)
    print(f"\nSaved {output_dir / 'tabnet_results.csv'}")

    total_min = (time.time() - t_start) / 60
    _write_summary(df, output_dir, total_min)


def _write_summary(df: pd.DataFrame, output_dir: Path, total_min: float) -> None:
    lines = []
    lines.append("# TabNet Baseline (Iftu Sir)\n")
    lines.append(
        "Trained on identical gene-aggregated features (mean/var/n_nonref/max "
        "per gene, 5000 genes) and identical `random_split` as every other "
        "classical baseline in this project (stage2_gene_xgb, CascadeStack "
        "v9's XGB-gene/LGBM-gene/RF-gene/LogReg-gene) — directly comparable "
        f"per-antibiotic PR-AUC. Total runtime: {total_min:.1f} min.\n"
    )
    lines.append("## Results\n")
    lines.append("| Antibiotic | PR-AUC | MCC | Runtime (min) |")
    lines.append("|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(f"| {r['antibiotic']} | {r['pr_auc']:.4f} | {r['mcc']:.4f} | {r['runtime_min']:.1f} |")
    lines.append("")
    lines.append(f"**Average PR-AUC: {df['pr_auc'].mean():.4f}**\n")

    # Compare against the existing gene-XGB baseline if its results are on disk.
    xgb_path = RESULTS_DIR / "stage2_gene_xgb" / "stage2_gene_xgb_results.csv"
    if xgb_path.exists():
        xgb_df = pd.read_csv(xgb_path)
        pr_col = [c for c in xgb_df.columns if "pr_auc" in c.lower() or "prauc" in c.lower()]
        lines.append("## Comparison to existing gene-XGB baseline\n")
        lines.append(f"(see `{xgb_path.relative_to(ROOT)}` for the reference numbers this was compared against)\n")
        if pr_col:
            lines.append(f"XGB-gene columns found: {pr_col} — cross-check manually against the table above.\n")

    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {output_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
