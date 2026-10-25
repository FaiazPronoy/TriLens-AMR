"""
src/data/pergene_snp_blocks.py
================================
Build and cache the per-gene SNP block matrix for Expert A (within-gene motif CNN).

For each of 5000 selected genes:
  - If len_g >= L_max : select the L_max columns with HIGHEST train-only variance.
  - If len_g < L_max  : use all len_g columns; LEFT-PAD to L_max with zeros.

The mask M_blocks (G, L) marks which positions contain real data (True) vs padding (False).
It is identical for all N samples (depends only on gene structure), so stored once.

Outputs written to cache_dir:
  X_blocks_{L}.npy            (N, G, L)  int8   — per-gene SNP values
  M_blocks_{L}.npy            (G, L)     bool   — True where real data, shared across samples
  pergene_snp_columns.json               dict   — gene-relative selected column indices
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Bump if the cache FILE FORMAT itself ever changes (not the data) -- lets an
# identity hash computed by an older/newer version of this module be told
# apart from a same-inputs hash computed by this version.
_CACHE_IDENTITY_VERSION = 1


def compute_cache_identity_hash(
    pkl_path: str, split_name: str, train_idx: np.ndarray
) -> str:
    """Fingerprint of (source pkl, split name, exact train indices).

    Order-invariant in train_idx (sorted before hashing) since every
    downstream computation here is order-invariant over the sample axis
    (variance, mean, etc.) -- two callers passing the same index SET in a
    different order must be treated as the same cache.
    """
    h = hashlib.sha256()
    h.update(str(_CACHE_IDENTITY_VERSION).encode("utf-8"))
    h.update(b"|")
    h.update(str(pkl_path).encode("utf-8"))
    h.update(b"|")
    h.update(str(split_name).encode("utf-8"))
    h.update(b"|")
    h.update(np.sort(np.asarray(train_idx)).astype(np.int64).tobytes())
    return h.hexdigest()


def _load_and_check_identity(
    identity_path: Path, expected_hash: str, pkl_path: str, split_name: str,
    train_idx: np.ndarray,
) -> bool:
    """True only if identity_path exists AND its stored hash matches."""
    if not identity_path.exists():
        log.info(
            f"  No cache_identity.json at {identity_path} -- cache predates "
            f"the identity check (or was never fingerprinted). Treating as "
            f"unverified: will rebuild rather than silently reuse."
        )
        return False
    try:
        with open(identity_path) as f:
            stored = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(f"  Cannot read {identity_path} ({exc}) -- rebuilding.")
        return False
    if stored.get("hash") != expected_hash:
        log.warning(
            f"  Cache identity MISMATCH at {identity_path.parent} -- this "
            f"cache was built for pkl_path={stored.get('pkl_path')!r} "
            f"split_name={stored.get('split_name')!r} "
            f"n_train={stored.get('n_train')}, but this call is for "
            f"pkl_path={pkl_path!r} split_name={split_name!r} "
            f"n_train={len(train_idx)}. Rebuilding instead of silently "
            f"reusing a cache from a different split/dataset."
        )
        return False
    return True


def _save_identity(
    identity_path: Path, hash_val: str, pkl_path: str, split_name: str,
    train_idx: np.ndarray,
) -> None:
    with open(identity_path, "w") as f:
        json.dump({
            "hash": hash_val,
            "pkl_path": str(pkl_path),
            "split_name": str(split_name),
            "n_train": int(len(train_idx)),
            "cache_identity_version": _CACHE_IDENTITY_VERSION,
        }, f, indent=2)


def build_pergene_snp_blocks(
    pkl_path:       str,
    cache_dir:      str,
    L_max:          int = 64,
    train_idx:      Optional[np.ndarray] = None,
    force_rebuild:  bool = False,
    split_name:     str = "",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build or load from cache the per-gene SNP block matrix.

    Parameters
    ----------
    pkl_path      : path to arcadia_v2_gene_filtered.pkl
    cache_dir     : directory for cached .npy files
    L_max         : target SNP block length (default 64)
    train_idx     : row indices for train split (for variance computation).
                    If None, loaded from pkl's random_split.
    force_rebuild : if True, rebuild even if cache exists
    split_name    : optional human-readable split identifier (e.g.
                    "phylogeny_split", "mlst_lineage_holdout_split"), folded
                    into the cache identity fingerprint alongside pkl_path
                    and train_idx itself. Purely a label -- train_idx is
                    already the load-bearing part of the fingerprint, this
                    just makes mismatches easier to diagnose in logs.

    Returns
    -------
    X_blocks      : (N, G, L) int8   — per-gene SNP block matrix
    M_blocks      : (G, L)    bool   — True where real data (shared across samples)
    col_indices   : dict {gene_id: [relative_col_indices]}

    Cache identity: a `cache_identity.json` sidecar (hash of pkl_path +
    split_name + sorted train_idx) is written next to the cached arrays and
    checked before ANY existing cache is trusted -- a cache built for a
    different split or a different source pkl is rebuilt, never silently
    reused. Caches written before this check existed have no sidecar and are
    therefore rebuilt on next use (this validates future runs; it does not
    retroactively alter any already-produced output file).
    """
    cache_dir  = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    blocks_path   = cache_dir / f"X_blocks_L{L_max}.npy"
    mask_path     = cache_dir / f"M_blocks_L{L_max}.npy"
    cols_path     = cache_dir / "pergene_snp_columns.json"
    identity_path = cache_dir / "cache_identity.json"

    # train_idx is needed for the identity check even on a would-be cache
    # hit, so resolve the "None -> random_split" default up front rather
    # than only inside the rebuild branch (unchanged behavior otherwise).
    if train_idx is None:
        with open(pkl_path, "rb") as f:
            _peek = pickle.load(f)
        _split = _peek["random_split"]
        train_idx = np.array(_split.get("train_idx", _split.get("train", [])))
        del _peek

    expected_hash = compute_cache_identity_hash(pkl_path, split_name, train_idx)

    cache_files_present = (blocks_path.exists() and mask_path.exists()
                           and cols_path.exists())
    identity_ok = cache_files_present and _load_and_check_identity(
        identity_path, expected_hash, pkl_path, split_name, train_idx
    )

    if not force_rebuild and identity_ok:
        log.info(f"Loading cached Expert A blocks from {cache_dir} "
                 f"(identity verified)")
        X_blocks    = np.load(str(blocks_path), mmap_mode="r")
        M_blocks    = np.load(str(mask_path))
        with open(cols_path) as f:
            col_indices = json.load(f)
        log.info(
            f"  X_blocks={X_blocks.shape}  M_blocks={M_blocks.shape}  "
            f"genes={len(col_indices)}"
        )
        return X_blocks, M_blocks, col_indices

    # Drop any stale cache files up front (e.g. a genuinely different-split
    # cache that just failed the identity check above). Best-effort: if a
    # file can't be removed (in-process reader still holding it open, a rare
    # same-process re-entry pattern none of this repo's callers use), the
    # subsequent np.save simply surfaces that as a normal write error rather
    # than silently reusing stale content.
    for p in (blocks_path, mask_path, cols_path, identity_path):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    # ── Load pkl ──────────────────────────────────────────────────────────────
    log.info(f"Building Expert A SNP blocks from {pkl_path}  (L_max={L_max})")
    t0 = time.time()
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X_snp          = data["X_snp_filtered"]          # (N, n_snps) int8
    gene_to_range  = data["gene_to_range"]            # {gene_id: (start, end)}
    selected_genes = data["selected_gene_ids"]        # list of G gene IDs
    N, n_snps      = X_snp.shape
    G              = len(selected_genes)

    # train_idx is already resolved above (before the cache-identity check,
    # which needs it) -- this is unchanged behavior, just relocated earlier.
    log.info(
        f"  Dataset: N={N}  G={G}  n_snps={n_snps}  n_train={len(train_idx)}"
    )

    # ── Output arrays ─────────────────────────────────────────────────────────
    X_blocks  = np.zeros((N, G, L_max), dtype=np.int8)
    M_blocks  = np.zeros((G, L_max),    dtype=bool)   # shared across samples
    col_indices: Dict[str, List[int]] = {}

    # ── Per-gene processing ───────────────────────────────────────────────────
    for g_idx, gene_id in enumerate(selected_genes):
        start, end = gene_to_range[gene_id]
        len_g      = int(end) - int(start)

        if len_g <= 0:
            # Degenerate gene — mask stays False
            col_indices[str(gene_id)] = []
            continue

        if len_g >= L_max:
            # Select top-L_max columns by TRAIN-ONLY variance
            gene_train = X_snp[train_idx, start:end].astype(np.float32)  # (n_tr, len_g)
            var        = gene_train.var(axis=0)                           # (len_g,)
            top_rel    = np.argsort(var)[::-1][:L_max]                   # top L by variance
            top_rel    = np.sort(top_rel)                                 # sort for reproducibility

            X_blocks[:, g_idx, :]  = X_snp[:, start + top_rel]          # (N, L_max) int8
            M_blocks[g_idx, :]     = True                                 # all positions valid
            col_indices[str(gene_id)] = top_rel.tolist()

        else:
            # LEFT-PAD to L_max: data is right-aligned
            pad = L_max - len_g
            X_blocks[:, g_idx, pad:] = X_snp[:, start:end]              # (N, len_g)
            M_blocks[g_idx, pad:]    = True                               # right part valid
            col_indices[str(gene_id)] = list(range(len_g))               # all columns

        if (g_idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            log.info(f"  {g_idx + 1}/{G} genes processed  ({elapsed:.0f}s)")

    # ── Sanity check ──────────────────────────────────────────────────────────
    n_fully_valid = (M_blocks.all(axis=1)).sum()
    n_padded      = G - n_fully_valid
    log.info(
        f"  Done in {time.time() - t0:.1f}s  |  "
        f"full genes (len≥{L_max}): {n_fully_valid}  |  "
        f"padded genes (len<{L_max}): {n_padded}"
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(str(blocks_path), X_blocks)
    np.save(str(mask_path),   M_blocks)
    with open(cols_path, "w") as f:
        json.dump(col_indices, f)
    _save_identity(identity_path, expected_hash, pkl_path, split_name, train_idx)

    log.info(
        f"Saved Expert A blocks:  "
        f"X_blocks={X_blocks.shape}  M_blocks={M_blocks.shape}  "
        f"→ {cache_dir}  (identity fingerprint saved)"
    )
    return X_blocks, M_blocks, col_indices


def verify_blocks(X_blocks: np.ndarray, M_blocks: np.ndarray) -> None:
    """Quick sanity checks on the produced block matrix."""
    N, G, L = X_blocks.shape
    assert M_blocks.shape == (G, L), f"M_blocks shape mismatch: {M_blocks.shape}"
    n_valid_per_gene = M_blocks.sum(axis=1)  # (G,)
    assert (n_valid_per_gene > 0).all(), "Some genes have zero valid positions"
    log.info(
        f"Block verification OK: N={N} G={G} L={L}  "
        f"valid_positions per gene: min={n_valid_per_gene.min()} "
        f"max={n_valid_per_gene.max()} mean={n_valid_per_gene.mean():.1f}"
    )
