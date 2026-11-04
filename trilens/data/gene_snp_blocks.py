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


_CACHE_IDENTITY_VERSION = 1


def compute_cache_identity_hash(pkl_path: str, split_name: str, train_idx: np.ndarray) -> str:

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
    identity_path: Path,
    expected_hash: str,
    pkl_path: str,
    split_name: str,
    train_idx: np.ndarray,
) -> bool:

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
    identity_path: Path,
    hash_val: str,
    pkl_path: str,
    split_name: str,
    train_idx: np.ndarray,
) -> None:
    with open(identity_path, "w") as f:
        json.dump(
            {
                "hash": hash_val,
                "pkl_path": str(pkl_path),
                "split_name": str(split_name),
                "n_train": int(len(train_idx)),
                "cache_identity_version": _CACHE_IDENTITY_VERSION,
            },
            f,
            indent=2,
        )


def build_pergene_snp_blocks(
    pkl_path: str,
    cache_dir: str,
    L_max: int = 64,
    train_idx: Optional[np.ndarray] = None,
    force_rebuild: bool = False,
    split_name: str = "",
) -> Tuple[np.ndarray, np.ndarray, Dict]:

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    blocks_path = cache_dir / f"X_blocks_L{L_max}.npy"
    mask_path = cache_dir / f"M_blocks_L{L_max}.npy"
    cols_path = cache_dir / "pergene_snp_columns.json"
    identity_path = cache_dir / "cache_identity.json"

    if train_idx is None:
        with open(pkl_path, "rb") as f:
            _peek = pickle.load(f)
        _split = _peek["random_split"]
        train_idx = np.array(_split.get("train_idx", _split.get("train", [])))
        del _peek

    expected_hash = compute_cache_identity_hash(pkl_path, split_name, train_idx)

    cache_files_present = blocks_path.exists() and mask_path.exists() and cols_path.exists()
    identity_ok = cache_files_present and _load_and_check_identity(
        identity_path, expected_hash, pkl_path, split_name, train_idx
    )

    if not force_rebuild and identity_ok:
        log.info(f"Loading cached Expert A blocks from {cache_dir} (identity verified)")
        X_blocks = np.load(str(blocks_path), mmap_mode="r")
        M_blocks = np.load(str(mask_path))
        with open(cols_path) as f:
            col_indices = json.load(f)
        log.info(f"  X_blocks={X_blocks.shape}  M_blocks={M_blocks.shape}  genes={len(col_indices)}")
        return X_blocks, M_blocks, col_indices

    for p in (blocks_path, mask_path, cols_path, identity_path):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    log.info(f"Building Expert A SNP blocks from {pkl_path}  (L_max={L_max})")
    t0 = time.time()
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X_snp = data["X_snp_filtered"]
    gene_to_range = data["gene_to_range"]
    selected_genes = data["selected_gene_ids"]
    N, n_snps = X_snp.shape
    G = len(selected_genes)

    log.info(f"  Dataset: N={N}  G={G}  n_snps={n_snps}  n_train={len(train_idx)}")

    X_blocks = np.zeros((N, G, L_max), dtype=np.int8)
    M_blocks = np.zeros((G, L_max), dtype=bool)
    col_indices: Dict[str, List[int]] = {}

    for g_idx, gene_id in enumerate(selected_genes):
        start, end = gene_to_range[gene_id]
        len_g = int(end) - int(start)

        if len_g <= 0:
            col_indices[str(gene_id)] = []
            continue

        if len_g >= L_max:
            gene_train = X_snp[train_idx, start:end].astype(np.float32)
            var = gene_train.var(axis=0)
            top_rel = np.argsort(var)[::-1][:L_max]
            top_rel = np.sort(top_rel)

            X_blocks[:, g_idx, :] = X_snp[:, start + top_rel]
            M_blocks[g_idx, :] = True
            col_indices[str(gene_id)] = top_rel.tolist()

        else:
            pad = L_max - len_g
            X_blocks[:, g_idx, pad:] = X_snp[:, start:end]
            M_blocks[g_idx, pad:] = True
            col_indices[str(gene_id)] = list(range(len_g))

        if (g_idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            log.info(f"  {g_idx + 1}/{G} genes processed  ({elapsed:.0f}s)")

    n_fully_valid = (M_blocks.all(axis=1)).sum()
    n_padded = G - n_fully_valid
    log.info(
        f"  Done in {time.time() - t0:.1f}s  |  "
        f"full genes (len≥{L_max}): {n_fully_valid}  |  "
        f"padded genes (len<{L_max}): {n_padded}"
    )

    np.save(str(blocks_path), X_blocks)
    np.save(str(mask_path), M_blocks)
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
    N, G, L = X_blocks.shape
    assert M_blocks.shape == (G, L), f"M_blocks shape mismatch: {M_blocks.shape}"
    n_valid_per_gene = M_blocks.sum(axis=1)
    assert (n_valid_per_gene > 0).all(), "Some genes have zero valid positions"
    log.info(
        f"Block verification OK: N={N} G={G} L={L}  "
        f"valid_positions per gene: min={n_valid_per_gene.min()} "
        f"max={n_valid_per_gene.max()} mean={n_valid_per_gene.mean():.1f}"
    )
