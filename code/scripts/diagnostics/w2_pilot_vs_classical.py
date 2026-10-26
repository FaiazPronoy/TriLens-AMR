"""
W2 pilot: how much the classical arm adds above lineage lookup, on the splits whose classical
test predictions already exist (S2 phylogroup, S3 ST holdout). P/A-only pilot.

    python scripts/diagnostics/w2_pilot_vs_classical.py --classical <folder with the two
        pillar_cascadestack_classical_* folders of the outputs-classical release>

The classical predictions are from the `classical` run made before the data repair. These two
folders read the phylogroup and holdout gene sets, which the repair does not touch. XGBoost on the
GPU drifts slightly between runs. They are replaced by the re-run when it finishes.

Output: results/w2_lineage_lookup_pa_pilot/classical_minus_lookup.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.metadata import load_strain_table  # noqa: E402
from src.evaluation.shift import bootstrap_ci, holm, paired_bootstrap  # noqa: E402

PILOT = ROOT / "results" / "w2_lineage_lookup_pa_pilot"
RUNS = {"S2_phylogroup": "pillar_cascadestack_classical_phylogeny_clean",
        "S3_st_holdout": "pillar_cascadestack_classical_mlst"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical", required=True, type=Path)
    a = ap.parse_args()
    t = load_strain_table()
    lk = pd.read_csv(PILOT / "lookup_results.csv")
    sc = np.load(PILOT / "test_scores.npz", allow_pickle=False)
    rows = []
    for sid, run in RUNS.items():
        z = np.load(a.classical / run / "predictions_saved.npz", allow_pickle=False)
        names = [str(n) for n in z["antibiotic_names"]]
        cls_idx = np.asarray(z["test_indices"]).ravel()
        pos = {s: i for i, s in enumerate(cls_idx)}
        for ab in names:
            key = f"{sid}__{ab}"
            if f"{key}__test_idx" not in sc.files:
                continue
            te, y = sc[f"{key}__test_idx"], sc[f"{key}__y_test"]
            assert np.array_equal(y, t[ab].values[te].astype(int)), "labels out of order"
            cls = np.asarray(z["test_ens_probs"])[[pos[s] for s in te], names.index(ab)]
            sub = lk[(lk.split == sid) & (lk.antibiotic == ab)]
            best = sub.loc[sub.best_on_val, "baseline"].iloc[0]
            ap_c, lo_c, hi_c = bootstrap_ci(y, cls)
            for b in sub.baseline:
                d, lo, hi, p = paired_bootstrap(y, cls, sc[f"{key}__{b}"])
                rows.append(dict(split=sid, antibiotic=ab, lookup=b, lookup_is_best_on_val=(b == best),
                                 classical_pr_auc=round(ap_c, 4), classical_ci=f"[{lo_c:.4f}, {hi_c:.4f}]",
                                 lookup_pr_auc=round(average_precision_score(y, sc[f"{key}__{b}"]), 4),
                                 delta=round(d, 4), delta_lo=round(lo, 4), delta_hi=round(hi, 4), p=p))
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for (sid, b), g in df.groupby(["split", "lookup"]):
        df.loc[g.index, "p_holm"] = holm(g.p.values)
    df.insert(0, "note", "P/A-only pilot; classical predictions from the pre-re-run release")
    df.to_csv(PILOT / "classical_minus_lookup.csv", index=False)
    print(df[df.lookup_is_best_on_val].drop(columns="note").to_string(index=False))


if __name__ == "__main__":
    main()
