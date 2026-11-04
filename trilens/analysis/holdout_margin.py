from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from trilens.evaluation.cluster_bootstrap import cluster_bootstrap, group_codes, summary, within_st_concordance

ROOT = Path(__file__).resolve().parents[2]
AB = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]
MODELS = {"stack": "trilens_stack_lineage", "classical": "classical_stack_lineage", "neural": "trilens_neural_lineage"}


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
            rows.append(
                dict(
                    model=name,
                    drug=d,
                    **summary(Y[d], S[d]),
                    **{f"within_st_{k}": v for k, v in within_st_concordance(Y[d], S[d], st).items()},
                )
            )
    df = pd.DataFrame(rows)
    out = Path(a.results) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    mean = df.groupby("model")[["ap", "nap", "auroc", "within_st_macro"]].mean().reset_index().assign(drug="MEAN")
    pd.concat([df, mean], ignore_index=True).to_csv(out / "holdout_margin.csv", index=False)
    print(
        pd.concat([df, mean])[
            ["model", "drug", "prevalence", "ap", "nap", "auroc", "within_st_macro", "within_st_n_st"]
        ]
        .round(3)
        .to_string(index=False)
    )
    deltas = []
    if {"stack", "classical"} <= set(scores):
        g = group_codes(st)
        for name, ds in [(d, [d]) for d in AB] + [("mean_7", AB)]:
            pt, lo, hi, ng = cluster_bootstrap(Y, scores["classical"], scores["stack"], g, ds, n_boot=a.n_boot)
            deltas.append(
                dict(comparison="stack - classical", endpoint=name, delta_nap=pt, ci_low=lo, ci_high=hi, n_st=ng)
            )
        dd = pd.DataFrame(deltas)
        dd.to_csv(out / "holdout_margin_deltas.csv", index=False)
        print(dd.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
