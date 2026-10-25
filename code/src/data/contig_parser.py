# src/data/contig_parser.py
"""Parse Arcadia SNP position strings into contig structure.

Verified Arcadia snp_positions format:
    "CONTIG_ID:POSITION:REF>ALT1,ALT2,..."

Examples:
    'LMHECDEF_00004:12:A>T'
    'LMHECDEF_00004:30:A>G,C'   # multi-allelic
"""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np

_CONTIG_PATTERN = re.compile(r"^([^:]+):(\d+):")


def parse_snp_position(pos_str: str) -> tuple[str, int]:
    """Parse a single SNP position string.

    Args:
        pos_str: e.g. 'LMHECDEF_00004:12:A>T' or 'LMHECDEF_00004:30:A>G,C'

    Returns:
        (contig_id, position)

    Raises:
        ValueError: if pos_str cannot be parsed.
    """
    m = _CONTIG_PATTERN.match(pos_str)
    if m is None:
        raise ValueError(f"Cannot parse SNP position: {pos_str!r}")
    return m.group(1), int(m.group(2))


def parse_contig_boundaries(
    snp_positions: np.ndarray,
) -> Tuple[dict[str, tuple[int, int]], np.ndarray, np.ndarray]:
    """Parse full snp_positions array into contig structure.

    Args:
        snp_positions: 1-D array of position strings (length n_snps).

    Returns:
        contig_ranges: ``{contig_id: (start_idx_inclusive, end_idx_exclusive)}``
        contig_ids_per_snp: ``int32`` array length n_snps, contig index per SNP.
        contig_id_list: array of unique contig ids in order of first appearance.
    """
    n = len(snp_positions)
    contig_ranges: dict[str, tuple[int, int]] = {}
    contig_ids_per_snp = np.zeros(n, dtype=np.int32)
    contig_id_list: list[str] = []
    contig_to_int: dict[str, int] = {}

    current_contig: str | None = None
    current_start = 0

    for i, pos_str in enumerate(snp_positions):
        contig_id, _ = parse_snp_position(str(pos_str))

        if contig_id != current_contig:
            if current_contig is not None:
                contig_ranges[current_contig] = (current_start, i)
            current_contig = contig_id
            current_start = i
            if contig_id not in contig_to_int:
                contig_to_int[contig_id] = len(contig_id_list)
                contig_id_list.append(contig_id)

        contig_ids_per_snp[i] = contig_to_int[contig_id]

    if current_contig is not None:
        contig_ranges[current_contig] = (current_start, n)

    return contig_ranges, contig_ids_per_snp, np.array(contig_id_list)


def get_contig_size_stats(contig_ranges: dict[str, tuple[int, int]]) -> dict:
    """Summary stats for contig sizes."""
    sizes = np.array([end - start for start, end in contig_ranges.values()])
    return {
        "n_contigs": int(len(sizes)),
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "median_size": float(np.median(sizes)),
        "mean_size": float(sizes.mean()),
        "p95_size": float(np.percentile(sizes, 95)),
        "total_snps": int(sizes.sum()),
    }


def build_contig_gather_index(
    contig_ids_per_snp: np.ndarray,
    min_contig_size: int = 16,
    max_contig_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build padded gather indices for vectorised per-contig CNN.

    Args:
        contig_ids_per_snp: contig-id-per-SNP int array (length n_snps).
        min_contig_size: drop contigs smaller than this many SNPs.
        max_contig_len: if given, truncate contigs longer than this. If None,
            uses 95th percentile of kept contig sizes.

    Returns:
        gather_idx: ``int64`` array shape (n_kept_contigs, max_contig_len);
            invalid positions contain 0 (safe gather index).
        valid_mask: ``float32`` array same shape; 1.0 valid, 0.0 padding.
        kept_contig_ids: ``int32`` array of original contig ids kept.
        max_contig_len: resolved max contig length actually used.
    """
    n_contigs_raw = int(contig_ids_per_snp.max()) + 1
    contig_to_snp_idx: list[np.ndarray] = []
    sizes: list[int] = []
    for ci in range(n_contigs_raw):
        snp_idx = np.where(contig_ids_per_snp == ci)[0]
        contig_to_snp_idx.append(snp_idx)
        sizes.append(len(snp_idx))
    sizes_arr = np.array(sizes)

    kept_mask = sizes_arr >= min_contig_size
    kept_contig_ids = np.where(kept_mask)[0].astype(np.int32)
    n_kept = len(kept_contig_ids)

    if max_contig_len is None:
        kept_sizes = sizes_arr[kept_mask]
        max_contig_len = int(np.percentile(kept_sizes, 95))
        max_contig_len = max(max_contig_len, min_contig_size)

    gather_idx = np.zeros((n_kept, max_contig_len), dtype=np.int64)
    valid_mask = np.zeros((n_kept, max_contig_len), dtype=np.float32)

    for new_ci, raw_ci in enumerate(kept_contig_ids):
        snp_idx = contig_to_snp_idx[raw_ci]
        n = min(len(snp_idx), max_contig_len)
        gather_idx[new_ci, :n] = snp_idx[:n]
        valid_mask[new_ci, :n] = 1.0

    return gather_idx, valid_mask, kept_contig_ids, max_contig_len
