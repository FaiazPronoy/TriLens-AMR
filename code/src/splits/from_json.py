"""
Rebuild the split pickles of the new splits from the committed index files, so a Kaggle notebook
needs no uploaded split files.

    python -m src.splits.from_json [--json-dir DIR]

DIR defaults to $TRILENS_REPO_RESULTS/splits (set by the notebook setup cell to the cloned
repository's code/results), else results/splits. Writes data/processed/<name>_split.pkl with the
same keys as src.splits.country_time; ST per strain is taken from mlst_lineage_holdout_split.pkl.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from src.data.metadata import PROCESSED, ROOT


def default_json_dir() -> Path:
    repo = os.environ.get("TRILENS_REPO_RESULTS")
    return Path(repo) / "splits" if repo else ROOT / "results" / "splits"


def split_from_json(d: dict, n_strains: int, st_per_genome) -> dict:
    parts = {k: np.asarray(d[k], dtype=np.int64) for k in ("train_idx", "val_idx", "test_idx", "unused_idx")}
    seen = np.concatenate(list(parts.values()))
    if len(seen) != n_strains or len(np.unique(seen)) != n_strains:
        raise ValueError(f"{d['name']}: partitions are not a disjoint cover of the {n_strains} strains")
    return {**parts, "st_per_genome": np.asarray(st_per_genome, dtype=object), "rules": d["rules"],
            "antibiotics": d["antibiotics"], "info": d["info"], "built_from": "results/splits json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=Path, default=None)
    a = ap.parse_args()
    src = a.json_dir or default_json_dir()
    files = sorted(src.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no split files in {src}")
    with open(PROCESSED / "mlst_lineage_holdout_split.pkl", "rb") as f:
        st = pickle.load(f)["st_per_genome"]
    for p in files:
        d = json.loads(p.read_text())
        split = split_from_json(d, len(st), st)
        out = PROCESSED / f"{d['name']}_split.pkl"
        with open(out, "wb") as f:
            pickle.dump(split, f, protocol=4)
        print(f"{d['name']:24s} train {len(split['train_idx']):5d}  val {len(split['val_idx']):4d}  "
              f"test {len(split['test_idx']):5d}  -> {out}")


if __name__ == "__main__":
    main()
