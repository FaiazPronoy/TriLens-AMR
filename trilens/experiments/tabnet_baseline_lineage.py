from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

from trilens.data.gene_statistics import build_gene_feature_matrix_from_pkl

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered.pkl"
MLST_SPLIT_PKL = ROOT / "data" / "processed" / "mlst_lineage_holdout_split.pkl"
OUTPUT_DIR = RESULTS_DIR / "tabnet_lineage"

GENE_STATS = ("mean", "var", "n_nonref", "max")

TABNET_PARAMS = dict(n_d=16, n_a=16, n_steps=4, gamma=1.3, lambda_sparse=1.0e-3, seed=42)
TRAINING = dict(max_epochs=150, patience=20, batch_size=256, virtual_batch_size=128)


def main() -> None:
    if not MLST_SPLIT_PKL.exists():
        raise FileNotFoundError(f"{MLST_SPLIT_PKL} not found -- run trilens.experiments.lineage_holdout_split first.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading gene-aggregated features (same pkl as the other 2 TabNet runs)...")
    bundle = build_gene_feature_matrix_from_pkl(
        gene_filtered_pkl_path=str(GENE_FILTERED_PKL),
        stats=GENE_STATS,
    )
    X_flat = bundle["X_gene_features"].astype(np.float32)
    Y = bundle["Y"].astype(np.int64)
    label_mask = bundle["label_mask"].astype(bool)
    label_names = [str(x) for x in bundle["label_names"]]

    with open(MLST_SPLIT_PKL, "rb") as f:
        sp = pickle.load(f)
    tr_idx = np.asarray(sp["train_idx"])
    va_idx = np.asarray(sp["val_idx"])
    te_idx = np.asarray(sp["test_idx"])
    st = np.asarray(sp["st_per_genome"], dtype=object)

    st_tr = {st[i] for i in tr_idx if st[i] is not None}
    st_te = {st[i] for i in te_idx if st[i] is not None}
    assert not (st_tr & st_te), "TRAIN/TEST ST LEAK"
    print(f"ST disjointness re-verified. train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

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
            n_d=TABNET_PARAMS["n_d"],
            n_a=TABNET_PARAMS["n_a"],
            n_steps=TABNET_PARAMS["n_steps"],
            gamma=TABNET_PARAMS["gamma"],
            lambda_sparse=TABNET_PARAMS["lambda_sparse"],
            seed=TABNET_PARAMS["seed"],
            device_name=device,
        )
        clf.fit(
            X_tr_s,
            y_tr,
            eval_set=[(X_va_s, y_va)],
            eval_name=["val"],
            eval_metric=["auc"],
            max_epochs=TRAINING["max_epochs"],
            patience=TRAINING["patience"],
            batch_size=TRAINING["batch_size"],
            virtual_batch_size=TRAINING["virtual_batch_size"],
        )

        test_prob = clf.predict_proba(X_te_s)[:, 1]
        pr_auc = float(average_precision_score(y_te, test_prob))
        y_pred = (test_prob >= 0.5).astype(int)
        mcc = float(matthews_corrcoef(y_te, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0

        runtime_min = (time.time() - t_ab) / 60
        print(f"   PR-AUC={pr_auc:.4f}  MCC={mcc:.4f}  ({runtime_min:.1f} min)")
        rows.append({"antibiotic": ab, "pr_auc": pr_auc, "mcc": mcc, "runtime_min": round(runtime_min, 2)})

    df = pd.DataFrame(rows, columns=["antibiotic", "pr_auc", "mcc", "runtime_min"])
    df.to_csv(OUTPUT_DIR / "tabnet_lineage_results.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR / 'tabnet_lineage_results.csv'}")

    total_min = (time.time() - t_start) / 60
    _write_summary(df, total_min)


def _write_summary(df: pd.DataFrame, total_min: float) -> None:
    lines = [
        "# TabNet Baseline -- MLST lineage holdout split\n",
        "Same gene-aggregated features (random_split-ranked, matching the other "
        "2 TabNet runs) and TabNet hyperparameters as `results/tabnet_random/` "
        "and `results/tabnet_phylogroup/` -- only the split differs. "
        f"Total runtime: {total_min:.1f} min.\n",
        "## Results\n",
        "| Antibiotic | PR-AUC | MCC | Runtime (min) |",
        "|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r['antibiotic']} | {r['pr_auc']:.4f} | {r['mcc']:.4f} | {r['runtime_min']:.1f} |")
    lines.append("")
    lines.append(f"**Average PR-AUC: {df['pr_auc'].mean():.4f}**\n")

    lines.append("## Comparison across all 3 splits\n")
    lines.append("| Antibiotic | random_split | phylogroup_split | mlst_lineage_holdout |")
    lines.append("|---|---|---|---|")
    rnd_path = RESULTS_DIR / "tabnet_random" / "tabnet_random_results.csv"
    phy_path = RESULTS_DIR / "tabnet_phylogroup" / "tabnet_phylogroup_results.csv"
    rnd_map = (
        dict(zip(pd.read_csv(rnd_path)["antibiotic"], pd.read_csv(rnd_path)["pr_auc"])) if rnd_path.exists() else {}
    )
    phy_map = (
        dict(zip(pd.read_csv(phy_path)["antibiotic"], pd.read_csv(phy_path)["pr_auc"])) if phy_path.exists() else {}
    )
    mlst_map = dict(zip(df["antibiotic"], df["pr_auc"]))
    for ab in df["antibiotic"]:
        r = f"{rnd_map[ab]:.4f}" if ab in rnd_map else "n/a"
        p = f"{phy_map[ab]:.4f}" if ab in phy_map else "n/a"
        lines.append(f"| {ab} | {r} | {p} | {mlst_map[ab]:.4f} |")

    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
