#!/usr/bin/env python
"""Port the 30 disjoint cluster labels from the superseded
`amr_metax_dataset.pkl` (autoencoder-pca-random-split.ipynb) onto the
current Arcadia v2 sample order, by `genome_names`.

**Status: written to spec, NOT executable as of this refactor.**
`amr_metax_dataset.pkl` does not exist anywhere in this repository or the
parent thesis repository -- it only ever existed in the Kaggle session that
produced `autoencoder-pca-random-split.ipynb`. Running this script today
will fail immediately and clearly at the file-not-found check, rather than
silently doing nothing or fabricating cluster labels (Rule 2). See
docs/OPEN_QUESTIONS.md item 1.

Why this matters (Part B.1 of the refactor spec): those 30 clusters are
GENUINELY disjoint across train/val/test in the old dataset (sizes
4481/1078/1403, verified by the old notebook's own QA cell asserting empty
set intersection) -- a real lineage holdout, unlike the current
`phylogroup_stratified_split`'s 8 broad, non-exclusive phylogroups. If they
can be matched onto the current 6,962-sample order at a high rate, they are
a strictly better `groups=` argument for `splits.lineage_holdout_split` than
`groups_from_phylogroup()`.

Usage:
    python scripts/port_cluster_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.amr_data.config import load_paths  # noqa: E402
from src.amr_data.io import load_pickle, save_pickle  # noqa: E402

MIN_MATCH_RATE = 0.99


class ClusterPortingError(RuntimeError):
    pass


def port_cluster_labels(
    legacy_names: np.ndarray,
    legacy_clusters: np.ndarray,
    v2_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pure matching logic (separated from I/O so it's unit-testable without
    the legacy pkl present). Returns (ported_labels, matched_mask, match_rate).
    Unmatched positions get -1 in `ported_labels`.
    """
    name_to_cluster = dict(zip(legacy_names, legacy_clusters))
    ported = np.full(len(v2_names), -1, dtype=np.int64)
    matched = np.zeros(len(v2_names), dtype=bool)
    for i, name in enumerate(v2_names):
        if name in name_to_cluster:
            ported[i] = name_to_cluster[name]
            matched[i] = True
    return ported, matched, float(matched.mean()) if len(v2_names) else 0.0


def main() -> None:
    paths = load_paths()
    legacy_path = paths.legacy("amr_metax_dataset_pkl")
    master_path = paths.processed("master_features_pkl")
    out_path = paths.processed("ported_cluster_labels_pkl")

    if not legacy_path.exists():
        raise ClusterPortingError(
            f"amr_metax_dataset.pkl not found at {legacy_path}.\n"
            f"This file is the superseded 80,984-feature dataset described in "
            f"docs/AUDIT_FINDINGS.md (Part B). It was never present in this "
            f"repository -- it must be supplied (re-download from whatever "
            f"Kaggle dataset/notebook version produced it) before this script "
            f"can run. See docs/OPEN_QUESTIONS.md item 1 for what is also "
            f"still unknown even once the file is supplied (the clusters' "
            f"derivation method)."
        )
    if not master_path.exists():
        raise ClusterPortingError(f"arcadia_v2_master_features.pkl not found at {master_path}")

    legacy = load_pickle(legacy_path)
    master = load_pickle(master_path)

    if "cluster_labels" not in legacy:
        raise ClusterPortingError(
            "amr_metax_dataset.pkl found, but has no 'cluster_labels' key -- "
            "check this is the right file/version."
        )
    if "genome_names" not in legacy or "genome_names" not in master:
        raise ClusterPortingError("Both pkls must have 'genome_names' to align by.")

    legacy_names = np.asarray(legacy["genome_names"], dtype=str)
    legacy_clusters = np.asarray(legacy["cluster_labels"])
    v2_names = np.asarray(master["genome_names"], dtype=str)

    ported, matched, match_rate = port_cluster_labels(legacy_names, legacy_clusters, v2_names)

    print(f"Matched {matched.sum()}/{len(v2_names)} ({match_rate:.1%}) v2 samples "
          f"to a legacy cluster label by genome_names.")

    if match_rate < MIN_MATCH_RATE:
        raise ClusterPortingError(
            f"Match rate {match_rate:.1%} is below the required {MIN_MATCH_RATE:.0%}. "
            f"Refusing to save partially-matched cluster labels -- a "
            f"lineage_holdout_split built on an incomplete grouping variable "
            f"would silently treat every unmatched sample as its own "
            f"singleton group, which is not the same as a genuine failure "
            f"to port and would be easy to miss."
        )

    n_clusters = len(set(legacy_clusters.tolist()))
    result = {
        "ported_cluster_labels": ported,
        "matched_mask": matched,
        "match_rate": float(match_rate),
        "n_clusters_in_source": n_clusters,
        "source_file": str(legacy_path),
        "note": (
            "Derivation method for these clusters is UNKNOWN -- see "
            "docs/OPEN_QUESTIONS.md item 2. Verify this is acceptable as a "
            "grouping variable for lineage_holdout_split before relying on "
            "it for a thesis claim."
        ),
    }
    save_pickle(result, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
