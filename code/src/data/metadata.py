"""
Strain metadata aligned to the 6,962 strains of the processed data.

Joins `data/raw/strains_metadata.csv` to `arcadia_v2_master_features.pkl` by genome
name (and checks the SRA accession agrees), never by row position, and normalises the
columns the split builders and the data inventory use:

    country   "England" merged into "United Kingdom"; missing and "Control" -> None
    year      Collection.Year as int, or None
    host      "Human", "Non-human" or None (Host.Common.Name, else Host.Name)
    study     PubMed ID of the source publication, or "unpublished:<country>"
    st        Achtman ST from the lineage-holdout split file (None if unknown)
"""
from __future__ import annotations

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


def load_strain_table(raw=RAW, processed=PROCESSED) -> pd.DataFrame:
    """One row per strain, in the order of the processed data (index 0..6961)."""
    with open(Path(processed) / "arcadia_v2_master_features.pkl", "rb") as f:
        master = pickle.load(f)
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
    t = pd.DataFrame({
        "genome_name": names,
        "sra": md["SRA.Accession"].values,
        "country": [c if isinstance(c, str) else None for c in country.values],
        "year": [int(y) if pd.notna(y) else None for y in year.values],
        "host": [_host(c, n) for c, n in zip(md["Host.Common.Name"].values, md["Host.Name"].values)],
        "study": study.values,
        "st": st,
        "phylogroup": [pg_names[i] for i in np.asarray(master["cluster_labels"]).astype(int)],
    })
    for a, ab in enumerate(AB_NAMES):
        t[ab] = np.where(M[:, a], Y[:, a], np.nan)
    return t
