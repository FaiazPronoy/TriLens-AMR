from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
AB_NAMES = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]


def _host(common, name):
    s = common if isinstance(common, str) else name
    if not isinstance(s, str):
        return None
    return "Human" if s.lower().startswith("human") else "Non-human"


def _committed_table(master) -> pd.DataFrame:
    repo = os.environ.get("TRILENS_REPO_RESULTS")
    p = (Path(repo) if repo else ROOT / "results") / "data_inventory" / "strain_table.csv"
    if not p.exists():
        raise FileNotFoundError(f"neither data/raw/strains_metadata.csv nor {p} exists")
    t = pd.read_csv(p, dtype={"st": str, "study": str})
    Y = np.asarray(master["Y"], dtype=float)
    M = np.asarray(master["label_mask"]).astype(bool)
    for a, ab in enumerate(AB_NAMES):
        if not np.array_equal(np.where(M[:, a], Y[:, a], np.nan), t[ab].values, equal_nan=True):
            raise ValueError(f"{p}: {ab} labels differ from arcadia_v2_master_features.pkl")
    t.insert(0, "genome_name", np.asarray(master["genome_names"]).astype(str))
    t.insert(1, "sra", np.asarray(master["sra_ids"]).astype(str))
    t["year"] = [int(y) if pd.notna(y) else None for y in t.year]
    for c in ("country", "host", "st"):
        t[c] = [v if isinstance(v, str) else None for v in t[c]]
    return t.drop(columns=["strain_index"])


def load_strain_table(raw=RAW, processed=PROCESSED) -> pd.DataFrame:
    with open(Path(processed) / "arcadia_v2_master_features.pkl", "rb") as f:
        master = pickle.load(f)
    if not (Path(raw) / "strains_metadata.csv").exists():
        return _committed_table(master)
    with open(Path(processed) / "mlst_lineage_holdout_split.pkl", "rb") as f:
        mlst = pickle.load(f)
    names = np.asarray(master["genome_names"]).astype(str)
    md = pd.read_csv(Path(raw) / "strains_metadata.csv", low_memory=False)
    if md["Genome.Name"].duplicated().any():
        raise ValueError("duplicate genome names in strains_metadata.csv")
    md = md.set_index("Genome.Name").reindex(names)
    if md["SRA.Accession"].isna().any():
        raise ValueError(f"{int(md['SRA.Accession'].isna().sum())} strains have no metadata row")
    if not np.array_equal(md["SRA.Accession"].astype(str).values, np.asarray(master["sra_ids"]).astype(str)):
        raise ValueError("genome name and SRA accession joins disagree")

    country = md["Isolation.Country"].replace({"England": "United Kingdom", "Control": np.nan})
    year = md["Collection.Year"]
    study = md["Publication"].astype("string")
    study = study.where(study.notna(), "unpublished:" + country.fillna("unknown").astype(str))
    st = np.asarray(mlst["st_per_genome"], dtype=object)
    st = [None if s is None or str(s) in ("", "nan", "None") else str(s) for s in st]

    Y = np.asarray(master["Y"], dtype=float)
    M = np.asarray(master["label_mask"]).astype(bool)
    pg_names = list(master["phylogroup_to_int"])
    t = pd.DataFrame(
        {
            "genome_name": names,
            "sra": md["SRA.Accession"].values,
            "country": [c if isinstance(c, str) else None for c in country.values],
            "year": [int(y) if pd.notna(y) else None for y in year.values],
            "host": [_host(c, n) for c, n in zip(md["Host.Common.Name"].values, md["Host.Name"].values)],
            "study": study.values,
            "st": st,
            "phylogroup": [pg_names[i] for i in np.asarray(master["cluster_labels"]).astype(int)],
        }
    )
    for a, ab in enumerate(AB_NAMES):
        t[ab] = np.where(M[:, a], Y[:, a], np.nan)
    return t
