"""
The gene-filter refit and the classical arm on any split file (W1: S4a, S4b, S5; later S6), by
running the reviewed ST-holdout scripts unchanged on another split:

    python -m src.experiments.run_split_pipeline rank      --split country_uk_to_norway [--device cpu]
    python -m src.experiments.run_split_pipeline build     --split country_uk_to_norway
    python -m src.experiments.run_split_pipeline classical --split country_uk_to_norway [--device cuda]

rank       XGBoost gain on chi2-prefiltered SNPs of the split's train (early stopping on its val),
           as run_xgb_gene_importance_mlst.py  -> results/w1_gene_sets/<split>/
build      the 5,000-gene filtered SNP matrix from that ranking, as build_gene_filtered_dataset_mlst.py
           -> data/processed/arcadia_v2_gene_filtered__<split>.pkl (not part of any output zip)
classical  the 5 classical learners + LogReg meta, as run_cascadestack_classical_mlst.py
           -> results/w1_classical/<split>/

Only the antibiotics that pass the split's pre-registered minimum counts (primary + secondary,
stored in the split file) are ranked and fitted. The split file is data/processed/<split>_split.pkl
(built by src.splits.country_time or src.splits.from_json).
"""
from __future__ import annotations

import argparse
import pickle

from src.data.metadata import PROCESSED, ROOT


def split_antibiotics(split: str) -> list:
    with open(PROCESSED / f"{split}_split.pkl", "rb") as f:
        sp = pickle.load(f)
    a = sp["antibiotics"]
    return list(a["primary"]) + list(a.get("secondary", []))


def configure(stage: str, split: str, device: str):
    """Point the reviewed ST-holdout module at this split; returns the module."""
    split_pkl = PROCESSED / f"{split}_split.pkl"
    if not split_pkl.exists():
        raise FileNotFoundError(f"{split_pkl} missing: run python -m src.splits.from_json first")
    abs_ = split_antibiotics(split)
    ranking_dir = ROOT / "results" / "w1_gene_sets" / split
    gf_pkl = PROCESSED / f"arcadia_v2_gene_filtered__{split}.pkl"
    if stage == "rank":
        import src.experiments.run_xgb_gene_importance_mlst as m
        m.SPLIT_PKL, m.OUT_DIR, m.ANTIBIOTICS = split_pkl, ranking_dir, abs_
        m.XGB_PARAMS = {**m.XGB_PARAMS, "device": device}
    elif stage == "build":
        import src.experiments.build_gene_filtered_dataset_mlst as m
        m.SPLIT_PKL, m.IMPORTANCE_DIR, m.OUTPUT_PKL, m.SPLIT_KEY = split_pkl, ranking_dir, gf_pkl, split
    elif stage == "classical":
        import src.experiments.run_cascadestack_classical_mlst as m
        m.GENE_FILTERED_PKL, m.SPLIT_KEY, m.ANTIBIOTICS = gf_pkl, split, abs_
        m.OUTPUT_DIR = ROOT / "results" / "w1_classical" / split
        m.REQUIRE_ST_DISJOINT = split.startswith("mlst")
        m.XGB_PARAMS = {**m.XGB_PARAMS, "device": device}
        m.XGB_RAW_PARAMS = {**m.XGB_RAW_PARAMS, "device": device}
    else:
        raise ValueError(stage)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rank", "build", "classical"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    configure(a.stage, a.split, a.device).main()


if __name__ == "__main__":
    main()
