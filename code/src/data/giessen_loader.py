# src/data/giessen_loader.py
"""Load Giessen CSV data + single-antibiotic labels for CNN training.

Verified format:
    data CSV : 'prename' + 60,936 SNP cols (X<pos>). Values 0-4 (4=missing).
    pheno CSV: 'prename' + CIP, CTX, CTZ, GEN.

Same memory strategy as arcadia_snp_loader: keep int8, cast in __getitem__.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

GIESSEN_ANTIBIOTICS: list[str] = ["CIP", "CTX", "CTZ", "GEN"]
ARCADIA_TO_GIESSEN_AB: dict[str, str] = {
    "CIP": "CIP", "CTX": "CTX", "GEN": "GEN", "CAZ": "CTZ",
}


@dataclass(frozen=True)
class GiessenSplit:
    X: np.ndarray        # (n, n_snps) int8
    y: np.ndarray        # (n,) float32
    prenames: np.ndarray


@dataclass(frozen=True)
class GiessenBundle:
    snp_column_names: list[str]
    antibiotic: str
    train: GiessenSplit
    val: GiessenSplit
    test: GiessenSplit
    n_snps: int


def load_giessen_for_antibiotic(
    data_csv_path: str | Path,
    pheno_csv_path: str | Path,
    antibiotic: str,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> GiessenBundle:
    if antibiotic not in GIESSEN_ANTIBIOTICS:
        raise ValueError(f"{antibiotic!r} not in {GIESSEN_ANTIBIOTICS}")

    data  = pd.read_csv(data_csv_path)
    pheno = pd.read_csv(pheno_csv_path)

    snp_cols = [c for c in data.columns if c != "prename"]
    merged   = data.merge(pheno[["prename", antibiotic]], on="prename", how="inner")
    merged   = merged.dropna(subset=[antibiotic]).reset_index(drop=True)

    n    = len(merged)
    rng  = np.random.RandomState(seed)
    perm = rng.permutation(n)
    te   = int(n * train_frac)
    ve   = te + int(n * val_frac)

    X        = merged[snp_cols].to_numpy(dtype=np.int8)   # keep int8
    y        = merged[antibiotic].to_numpy(dtype=np.float32)
    prenames = merged["prename"].to_numpy()

    def _s(idx: np.ndarray) -> GiessenSplit:
        return GiessenSplit(X=X[idx], y=y[idx], prenames=prenames[idx])

    return GiessenBundle(
        snp_column_names=snp_cols,
        antibiotic=antibiotic,
        train=_s(perm[:te]),
        val=_s(perm[te:ve]),
        test=_s(perm[ve:]),
        n_snps=len(snp_cols),
    )


class GiessenSingleABDataset(Dataset):
    def __init__(self, split: GiessenSplit, shuffle_perm: Optional[np.ndarray] = None):
        self.X    = split.X      # int8
        self.y    = split.y
        self.perm = shuffle_perm

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.X[i]
        if self.perm is not None:
            row = row[self.perm]
        x = torch.from_numpy(row.astype(np.int64))
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y


def arcadia_to_giessen_ab(name: str) -> str | None:
    return ARCADIA_TO_GIESSEN_AB.get(name)
