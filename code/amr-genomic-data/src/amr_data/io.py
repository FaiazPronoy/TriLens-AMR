"""Load/save helpers for pkl and npz artifacts, with checksums.

Every artifact this pipeline writes is accompanied by a `.sha256` sidecar file
so that "did this refactor change a number" (Rule 1) can be checked
mechanically: if the checksum of a regenerated artifact differs from the
checked-in sidecar, something changed, and CHANGELOG_NUMBERS.md must explain
why before the new checksum is accepted.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksum_sidecar(path: Path) -> Path:
    digest = sha256_of_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return sidecar


def verify_checksum_sidecar(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        return False
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    return sha256_of_file(path) == expected


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path, *, protocol: int = 4, checksum: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
    if checksum:
        write_checksum_sidecar(path)
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def save_npz(arrays: dict[str, np.ndarray], path: Path, *, checksum: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    if checksum:
        write_checksum_sidecar(path)
    return path
