"""
holdout_validity.py
===================
The lineage-holdout numbers, read the way the external review asked (docs/novelty.md §7):
prevalence, PR-AUC, prevalence-normalised PR-AUC (nAP), AUROC and within-ST concordance per drug,
and an ST-clustered paired bootstrap CI for full stack minus classical arm.

Inputs (written by the re-run): results/pillar_mlst_final_comparison/predictions_saved.npz (stack),
results/pillar_cascadestack_classical_mlst/predictions_saved.npz (classical arm),
results/pillar_trilens_mlst/predictions_saved.npz (neural stack alone). Missing files are skipped.

    python scripts/diagnostics/holdout_validity.py [--results results] [--n-boot 2000]

Writes results/diagnostics/holdout_validity.csv and holdout_validity_deltas.csv.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.grouped import cluster_bootstrap, group_codes, summary, within_st_concordance  # noqa: E402

AB = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]
MODELS = {"stack": "pillar_mlst_final_comparison", "classical": "pillar_cascadestack_classical_mlst",
          "neural": "pillar_trilens_mlst"}


def load(results):
    with open(ROOT / "data" / "processed" / "arcadia_v2_master_features.pkl", "rb") as f:
        master = pickle.load(f)
    with open(ROOT / "data" / "processed" / "mlst_lineage_holdout_split.pkl", "rb") as f:
        split = pickle.load(f)
    Y = np.asarray(master["Y"], float)
    Y[~np.asarray(master["label_mask"]).astype(bool)] = np.nan
    test = np.asarray(split["test_idx"], int)
    st = np.asarray(split["st_per_genome"], dtype=object)
    scores = {}
    for name, d in MODELS.items():
        p = Path(results) / d / "predictions_saved.npz"
        if not p.exists():
            print(f"[skip] {name}: {p} not found")
            continue
        z = np.load(p, allow_pickle=True)
        idx = np.asarray(z["test_indices"], int)
        if not np.array_equal(np.sort(idx), np.sort(test)):
            raise ValueError(f"{p}: test strains are not the lineage holdout's")
        S = pd.DataFrame(np.nan, index=np.arange(len(Y)), columns=AB)
        S.loc[idx, AB] = np.asarray(z["test_ens_probs"], float)[:, : len(AB)]
        scores[name] = S
    Yt = pd.DataFrame(np.full_like(Y, np.nan), columns=AB)
    Yt.loc[test, AB] = Y[test]
    return Yt, st, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    Y, st, scores = load(a.results)
    rows = []
    for name, S in scores.items():
        for d in AB:
            rows.append(dict(model=name, drug=d, **summary(Y[d], S[d]),
                             **{f"within_st_{k}": v for k, v in within_st_concordance(Y[d], S[d], st).items()}))
    df = pd.DataFrame(rows)
    out = Path(a.results) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    mean = df.groupby("model")[["ap", "nap", "auroc", "within_st_macro"]].mean().reset_index().assign(drug="MEAN")
    pd.concat([df, mean], ignore_index=True).to_csv(out / "holdout_validity.csv", index=False)
    print(pd.concat([df, mean])[["model", "drug", "prevalence", "ap", "nap", "auroc", "within_st_macro",
                                 "within_st_n_st"]].round(3).to_string(index=False))
    deltas = []
    if {"stack", "classical"} <= set(scores):
        g = group_codes(st)
        for name, ds in [(d, [d]) for d in AB] + [("mean_7", AB)]:
            pt, lo, hi, ng = cluster_bootstrap(Y, scores["classical"], scores["stack"], g, ds, n_boot=a.n_boot)
            deltas.append(dict(comparison="stack - classical", endpoint=name, delta_nap=pt,
                               ci_low=lo, ci_high=hi, n_st=ng))
        dd = pd.DataFrame(deltas)
        dd.to_csv(out / "holdout_validity_deltas.csv", index=False)
        print(dd.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
