"""
src/experiments/run_tabnet_baseline_phylogeny.py
==================================================
TabNet baseline on `phylogeny_split` -- for consistency with the T6/T7
random_split-vs-phylogeny_split comparison (Iftu Sir feedback). Same gene-
aggregated features and architecture as
src/experiments/run_tabnet_baseline.py (random_split version); this is a
separate file (not a parametrized reuse of that script) to avoid editing an
already-written file, matching the pattern used elsewhere in this project
(e.g. run_pillar_cascadestack_phylogeny.py alongside run_cascadestack_v9.py).

NOT run yet (GPU job, ~5-15 min/antibiotic, same estimate as the
random_split version) -- write-only, per instructions.

Usage:
    conda activate amr
    python -m src.experiments.run_tabnet_baseline_phylogeny
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
CFG_PATH = ROOT / "configs" / "experiment" / "tabnet_baseline_phylogeny.yaml"
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

    # phylogeny_split isn't in build_gene_feature_matrix_from_pkl's passthrough
    # (only random_split is) -- load it directly from the same pkl.
    import pickle
    with open(GENE_FILTERED_PKL, "rb") as f:
        d = pickle.load(f)
    splits = d[str(cfg.split_key)]
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
    df.to_csv(output_dir / "tabnet_phylogeny_results.csv", index=False)
    print(f"\nSaved {output_dir / 'tabnet_phylogeny_results.csv'}")

    total_min = (time.time() - t_start) / 60
    _write_summary(df, output_dir, total_min)


def _write_summary(df: pd.DataFrame, output_dir: Path, total_min: float) -> None:
    lines = []
    lines.append("# TabNet Baseline — phylogeny_split\n")
    lines.append(
        "Same gene-aggregated features and TabNet hyperparameters as "
        "`results/tabnet_baseline/` (random_split version), evaluated on "
        "`phylogeny_split` instead, for a leakage-aware TabNet-vs-T6/T7 "
        f"comparison. Total runtime: {total_min:.1f} min.\n"
    )
    lines.append("## Results\n")
    lines.append("| Antibiotic | PR-AUC | MCC | Runtime (min) |")
    lines.append("|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(f"| {r['antibiotic']} | {r['pr_auc']:.4f} | {r['mcc']:.4f} | {r['runtime_min']:.1f} |")
    lines.append("")
    lines.append(f"**Average PR-AUC: {df['pr_auc'].mean():.4f}**\n")

    ref_path = RESULTS_DIR / "tabnet_baseline" / "tabnet_results.csv"
    if ref_path.exists():
        ref_df = pd.read_csv(ref_path)
        lines.append("## Comparison to TabNet on random_split\n")
        lines.append("| Antibiotic | random_split PR-AUC | phylogeny_split PR-AUC | Delta |")
        lines.append("|---|---|---|---|")
        ref_map = dict(zip(ref_df["antibiotic"], ref_df["pr_auc"]))
        for _, r in df.iterrows():
            rnd = ref_map.get(r["antibiotic"])
            if rnd is not None:
                lines.append(f"| {r['antibiotic']} | {rnd:.4f} | {r['pr_auc']:.4f} | {r['pr_auc'] - rnd:+.4f} |")
    else:
        lines.append(
            "(random_split TabNet results not found yet at "
            f"`{ref_path.relative_to(ROOT)}` -- run `run_tabnet_baseline.py` "
            "first for a full comparison.)"
        )

    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {output_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
