from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
K = 5


def results_dir() -> Path:
    repo = os.environ.get("TRILENS_REPO_RESULTS")
    return Path(repo) if repo else ROOT / "results"


def strain_st() -> np.ndarray:
    t = pd.read_csv(results_dir() / "data_inventory" / "strain_table.csv", dtype={"st": str})
    return np.array([s if isinstance(s, str) and s not in ("", "nan", "None") else None for s in t.st], dtype=object)


def st_grouped(st, k=K):
    st = pd.Series(st, dtype=object)
    known = st.notna()
    sizes = st[known].value_counts()
    order = sorted(sizes.index, key=lambda s: (-sizes[s], int(s) if str(s).isdigit() else 10**9, str(s)))
    load, fold_of = np.zeros(k, int), {}
    for s in order:
        f = int(np.argmin(load))
        fold_of[s] = f
        load[f] += sizes[s]
    folds = [np.flatnonzero(known & st.map(fold_of).eq(f)) for f in range(k)]
    return folds, np.flatnonzero(~known)


def random_folds(n, k=K, seed=0):
    return [test for _, test in KFold(k, shuffle=True, random_state=seed).split(np.arange(n))], np.array([], int)


def write():
    st = strain_st()
    out = results_dir() / "fold_splits"
    out.mkdir(parents=True, exist_ok=True)
    specs = {
        "st_grouped_5fold": (
            st_grouped(st),
            "STs whole in one fold, largest first to the emptiest fold; unknown ST: train in every fold, never scored",
        ),
        "random_5fold": (random_folds(len(st)), "KFold(5, shuffle=True, random_state=0)"),
    }
    for name, ((folds, never), rule) in specs.items():
        d = {
            "name": name,
            "n_strains": len(st),
            "rules": rule,
            "folds": [f.tolist() for f in folds],
            "never_tested": never.tolist(),
            "fold_sizes": [len(f) for f in folds],
            "sts_per_fold": [int(pd.Series(st[f]).nunique()) for f in folds],
        }
        (out / f"{name}.json").write_text(json.dumps(d))
        print(name, d["fold_sizes"], "STs", d["sts_per_fold"], "never tested", len(never))


def load_kfold(name):
    d = json.loads((results_dir() / "fold_splits" / f"{name}.json").read_text())
    folds = [np.asarray(f, int) for f in d["folds"]]
    allidx = np.concatenate(folds + [np.asarray(d["never_tested"], int)])
    if len(allidx) != d["n_strains"] or len(np.unique(allidx)) != d["n_strains"]:
        raise ValueError(f"{name}: folds are not a disjoint cover")
    return {**d, "folds": folds, "never_tested": np.asarray(d["never_tested"], int)}


if __name__ == "__main__":
    write()
