"""Phylogroup derivation (B10 — one canonical copy).

Phase B cells 3 and 5 each contained their own copy of a 57-entry Achtman
sequence-type (ST) -> Clermont phylogroup lookup table. They were diffed
programmatically during the audit and found to be byte-for-byte identical
(57 keys, same values) -- so unlike the label encoders (B2), there was no
actual divergence to resolve here, just duplication. This module holds the
single copy.

# TODO(faiaz): unresolved -- see docs/OPEN_QUESTIONS.md.
The provenance of this specific 57-entry ST->phylogroup mapping is not
documented anywhere in the notebooks (no citation, no generation script).
It is used both for the phylogroup baseline and to derive `phylogroup_final`
wherever `Other.Typing` doesn't already contain an explicit
`phylogroup:X` tag. Needed: whether this came from ClermonTyping, EnteroBase,
a specific paper, or was hand-curated -- this determines how much weight the
phylogroup labels should be given as ground truth in the thesis.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

_PHYLOGROUP_TAG_RE = re.compile(r"phylogroup:([A-Z][12]?)")
_MLST_ST_RE = re.compile(r"MLST\.Escherichia_coli_1\.(\d+)")

# Byte-for-byte identical between Phase B cell 3 and cell 5 (verified by
# programmatic diff during the audit -- see docs/AUDIT_FINDINGS.md, B10).
ST_TO_PHYLOGROUP: dict[str, str] = {
    "131": "B2", "73": "B2", "95": "B2", "127": "B2", "12": "B2",
    "141": "B2", "144": "B2", "372": "B2", "404": "B2", "416": "B2",
    "428": "B2", "550": "B2", "1193": "B2", "14": "B2", "80": "B2",
    "498": "B2", "538": "B2", "983": "B2",
    "69": "D", "405": "D", "354": "D", "38": "D", "117": "D",
    "115": "D", "457": "D", "362": "D",
    "10": "A", "167": "A", "617": "A", "44": "A", "744": "A",
    "48": "A", "93": "A", "227": "A", "1284": "A", "168": "A",
    "752": "A", "224": "A",
    "11": "B1", "58": "B1", "155": "B1", "101": "B1", "162": "B1",
    "297": "B1", "446": "B1", "453": "B1", "517": "B1", "200": "B1",
    "62": "F", "59": "F", "522": "F", "648": "F",
    "88": "C", "410": "C", "23": "C",
    "350": "E", "57": "E",
}

assert len(ST_TO_PHYLOGROUP) == 57, "ST_TO_PHYLOGROUP must stay at 57 entries -- see docstring"


def parse_phylogroup_tag(other_typing: Optional[str]) -> Optional[str]:
    """Extract an explicit `phylogroup:X` tag from the `Other.Typing` field."""
    if pd.isna(other_typing):
        return None
    m = _PHYLOGROUP_TAG_RE.findall(str(other_typing))
    return m[0] if m else None


def parse_mlst_st(mlst_field: Optional[str]) -> Optional[str]:
    """Extract the Achtman ST number from the `MLST` field."""
    if pd.isna(mlst_field):
        return None
    m = _MLST_ST_RE.search(str(mlst_field))
    return m.group(1) if m else None


def derive_phylogroup(other_typing: pd.Series, mlst: pd.Series) -> pd.Series:
    """Reproduce Phase B's `phylogroup_final` derivation exactly.

    Priority: explicit `phylogroup:X` tag in Other.Typing, else ST->phylogroup
    lookup via MLST, else "Unknown". Identical logic in both original cells.
    """
    tag = other_typing.apply(parse_phylogroup_tag)
    st = mlst.apply(parse_mlst_st)
    return tag.fillna(st.map(ST_TO_PHYLOGROUP)).fillna("Unknown")


def build_cluster_labels(phylogroup_final: pd.Series) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Map phylogroup strings to a dense int array + name list + name->int map."""
    unique_pg = sorted(phylogroup_final.unique())
    pg_to_int = {pg: i for i, pg in enumerate(unique_pg)}
    cluster_labels = np.array([pg_to_int[pg] for pg in phylogroup_final], dtype=np.int32)
    return cluster_labels, unique_pg, pg_to_int
