"""
confounder_floor.py
===================
How much of the apparent performance is reachable with NO sequence feature at all?

Trains gradient boosting on strain metadata only — assembly statistics (contigs, assembly
size, GC, N50, CDS count, tRNA count), collection year, isolation country, sequencing
platform and the source publication — and scores it on each split's test part. No SNP, no
gene presence/absence, no k-mer: nothing read from the genome sequence.

Any sequence-based model has to be read against this floor. A model that beats the random
split's floor by little is mostly re-deriving covariates that a surveillance database hands
out for free.

Honest reading of the features: assembly size and CDS count do carry biology (a larger
accessory genome means more acquired genes), so the floor is not "pure batch effect". It is
the performance attainable without variant- or gene-level features, which is exactly the
comparison a reader needs.

    python scripts/diagnostics/confounder_floor.py [--splits random,phylogeny,mlst,...]

Writes results/diagnostics/confounder_floor.csv (one row per split x feature set x
antibiotic, plus the mean rows) and prints the means.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
AB_NAMES = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]
SEED = 0

# feature blocks, so the table says which kind of covariate carries the signal
BLOCKS = {
    "assembly_qc": ["Contigs", "Size", "GC.Content", "Contig.N50", "CDS", "TRNA"],
    "year": ["Collection.Year"],
    "country": ["Isolation.Country"],
    "platform": ["Sequencing.Platform"],
    "publication": ["Publication"],
}
CATEGORICAL = {"Isolation.Country", "Sequencing.Platform", "Publication"}
FEATURE_SETS = {
    "assembly_qc_only": ["assembly_qc"],
    "publication_only": ["publication"],
    "country_year": ["country", "year"],
    "all_metadata": list(BLOCKS),
}


def load_splits(names):
    with open(PROC / "arcadia_v2_master_features.pkl", "rb") as f:
        master = pickle.load(f)
    out = {}
    for n in names:
        if n == "random":
            out[n] = master["random_split"]
        elif n == "phylogeny":
            out[n] = master["phylogeny_split"]
        else:
            p = PROC / (f"{n}_split.pkl" if n != "mlst" else "mlst_lineage_holdout_split.pkl")
            if not p.exists():
                print(f"[skip] {n}: {p.name} not found")
                continue
            with open(p, "rb") as f:
                out[n] = pickle.load(f)
    return master, out


def metadata_frame(genome_names):
    """One row per strain, in the processed data's order. No sequence is read."""
    md = pd.read_csv(RAW / "strains_metadata.csv", low_memory=False)
    if md["Genome.Name"].duplicated().any():
        raise ValueError("duplicate genome names in strains_metadata.csv")
    md = md.set_index("Genome.Name").reindex(np.asarray(genome_names).astype(str))
    cols = {}
    for block, fields in BLOCKS.items():
        for c in fields:
            if c in CATEGORICAL:
                cols[c] = pd.Categorical(md[c].astype(str)).codes.astype(float)
            else:
                cols[c] = pd.to_numeric(md[c], errors="coerce").values.astype(float)
    return pd.DataFrame(cols, index=np.arange(len(md)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="random,phylogeny,mlst")
    a = ap.parse_args()
    master, splits = load_splits([s.strip() for s in a.splits.split(",") if s.strip()])
    Y = np.nan_to_num(np.asarray(master["Y"], dtype=float))
    Mask = np.asarray(master["label_mask"]).astype(bool)
    X_all = metadata_frame(master["genome_names"])

    rows = []
    for split_name, sp in splits.items():
        tr = np.asarray(sp.get("train_idx", sp.get("train"))).ravel().astype(int)
        te = np.asarray(sp.get("test_idx", sp.get("test"))).ravel().astype(int)
        for set_name, blocks in FEATURE_SETS.items():
            X = X_all[[c for b in blocks for c in BLOCKS[b]]].to_numpy()
            for a_i, ab in enumerate(AB_NAMES):
                i_tr, i_te = tr[Mask[tr, a_i]], te[Mask[te, a_i]]
                y_tr, y_te = Y[i_tr, a_i], Y[i_te, a_i]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    continue
                clf = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
                p = clf.fit(X[i_tr], y_tr).predict_proba(X[i_te])[:, 1]
                rows.append(dict(split=split_name, feature_set=set_name, antibiotic=ab,
                                 n_train=len(i_tr), n_test=len(i_te),
                                 test_prevalence=round(float(y_te.mean()), 4),
                                 pr_auc=round(float(average_precision_score(y_te, p)), 4),
                                 roc_auc=round(float(roc_auc_score(y_te, p)), 4)))
    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing computed")
        return 1
    means = (df.groupby(["split", "feature_set"])[["pr_auc", "roc_auc"]].mean().round(4)
             .reset_index().assign(antibiotic="MEAN"))
    out = RES / "diagnostics" / "confounder_floor.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([df, means], ignore_index=True).to_csv(out, index=False)
    print(means.to_string(index=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
