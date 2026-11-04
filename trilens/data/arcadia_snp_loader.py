from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ArcadiaRawSplit:
    X: np.ndarray
    y: np.ndarray
    sample_indices: np.ndarray


@dataclass(frozen=True)
class ArcadiaRawBundle:
    snp_positions: np.ndarray
    label_names: list[str]
    train: ArcadiaRawSplit
    val: ArcadiaRawSplit
    test: ArcadiaRawSplit
    n_snps: int

    def get_split(self, name: str) -> ArcadiaRawSplit:
        return {"train": self.train, "val": self.val, "test": self.test}[name]


def load_arcadia_raw_for_antibiotic(
    snp_pkl_path: str | Path,
    master_pkl_path: str | Path,
    antibiotic: str,
    split_type: str = "random",
) -> ArcadiaRawBundle:
    with open(snp_pkl_path, "rb") as f:
        snp = pickle.load(f)
    with open(master_pkl_path, "rb") as f:
        master = pickle.load(f)

    X_snp: np.ndarray = snp["X_snp"]
    snp_positions: np.ndarray = np.asarray(snp["snp_positions"])
    label_names = [str(n) for n in master["label_names"]]

    if antibiotic not in label_names:
        raise ValueError(f"{antibiotic!r} not in label_names={label_names}")

    ab_idx = label_names.index(antibiotic)
    Y = master["Y"].astype("float32")
    mask = master["label_mask"].astype(bool)

    split_key = f"{split_type}_split"
    if split_key not in master:
        raise KeyError(f"Master pkl missing split key {split_key!r}")
    rs = master[split_key]

    def _make_split(raw_idx: np.ndarray) -> ArcadiaRawSplit:
        m = mask[raw_idx, ab_idx]
        valid = raw_idx[m]
        return ArcadiaRawSplit(
            X=X_snp[valid],
            y=Y[valid, ab_idx].astype(np.float32),
            sample_indices=valid,
        )

    return ArcadiaRawBundle(
        snp_positions=snp_positions,
        label_names=label_names,
        train=_make_split(np.asarray(rs["train_idx"])),
        val=_make_split(np.asarray(rs["val_idx"])),
        test=_make_split(np.asarray(rs["test_idx"])),
        n_snps=X_snp.shape[1],
    )


class ArcadiaSNPSingleABDataset(Dataset):
    def __init__(
        self,
        split: ArcadiaRawSplit,
        shuffle_perm: Optional[np.ndarray] = None,
    ):
        self.X = split.X
        self.y = split.y
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
