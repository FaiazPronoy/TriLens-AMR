"""PyTorch Lightning DataModule for AMR datasets (Arcadia v2 + Giessen)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig


class AMRTensorDataset(Dataset):
    """Generic torch Dataset for AMR features + multi-label phenotypes.
    
    Handles missing labels via mask. Returns (X, Y, mask) per sample.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        mask: np.ndarray,
        indices: np.ndarray,
    ):
        """
        Args:
            X: Feature matrix (n_samples, n_features).
            Y: Label matrix (n_samples, n_antibiotics) — binary.
            mask: Valid label mask (n_samples, n_antibiotics) — bool.
            indices: Indices to include in this split.
        """
        self.X = X[indices]
        self.Y = Y[indices].astype(np.float32)
        self.mask = mask[indices].astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "X": torch.from_numpy(self.X[idx]).float(),
            "Y": torch.from_numpy(self.Y[idx]),
            "mask": torch.from_numpy(self.mask[idx]),
        }


class ArcadiaDataModule(L.LightningDataModule):
    """Lightning DataModule for Arcadia v2.
    
    Supports PCA features (X_pca) or raw SNP features (X_snp).
    Handles random_split and phylogeny_split via config.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        feature_type: Literal["pca", "snp"] = "pca",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.cfg = cfg
        self.feature_type = feature_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Filled in setup()
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.train_idx: Optional[np.ndarray] = None
        self.val_idx: Optional[np.ndarray] = None
        self.test_idx: Optional[np.ndarray] = None
        self.phylogroup_onehot: Optional[np.ndarray] = None
        self.label_names: Optional[list[str]] = None
    
    def prepare_data(self) -> None:
        """Verify data files exist."""
        master_path = Path(self.cfg.paths.master_pkl)
        if not master_path.exists():
            raise FileNotFoundError(f"Master pkl not found: {master_path}")
        
        if self.feature_type == "snp":
            snp_path = Path(self.cfg.paths.snp_matrix_pkl)
            if not snp_path.exists():
                raise FileNotFoundError(f"SNP matrix pkl not found: {snp_path}")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare data."""
        with open(self.cfg.paths.master_pkl, "rb") as f:
            master = pickle.load(f)
        
        # Reconstruct full X from per-split arrays
        rs = master[self.cfg.col_names.random_split]
        train_idx = np.array(rs["train_idx"])
        val_idx = np.array(rs["val_idx"])
        test_idx = np.array(rs["test_idx"])
        n_samples = self.cfg.n_samples
        
        if self.feature_type == "pca":
            n_feat = self.cfg.n_pca_features
            X = np.zeros((n_samples, n_feat), dtype=np.float32)
            X[train_idx] = master[self.cfg.col_names.X_train]
            X[val_idx] = master[self.cfg.col_names.X_val]
            X[test_idx] = master[self.cfg.col_names.X_test]
        else:  # snp
            with open(self.cfg.paths.snp_matrix_pkl, "rb") as f:
                snp_data = pickle.load(f)
            X = snp_data["X_snp"]
        
        self.X = X
        self.Y = master[self.cfg.col_names.Y].astype(np.int64)
        self.mask = master[self.cfg.col_names.label_mask].astype(bool)
        self.phylogroup_onehot = master[self.cfg.col_names.phylogroup].astype(np.float32)
        self.label_names = list(master[self.cfg.col_names.label_names])
        
        # Apply split strategy
        if self.cfg.split == "random":
            self.train_idx = train_idx
            self.val_idx = val_idx
            self.test_idx = test_idx
        elif self.cfg.split == "phylogeny":
            ps = master[self.cfg.col_names.phylogeny_split]
            self.train_idx = np.array(ps["train_idx"])
            self.val_idx = np.array(ps["val_idx"])
            self.test_idx = np.array(ps["test_idx"])
        else:
            raise ValueError(f"Unknown split: {self.cfg.split}")
    
    def get_dataset(self, indices: np.ndarray) -> AMRTensorDataset:
        return AMRTensorDataset(self.X, self.Y, self.mask, indices)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset(self.train_idx),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset(self.val_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset(self.test_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class GiessenDataModule(L.LightningDataModule):
    """Lightning DataModule for Giessen dataset (CSV-based)."""
    
    def __init__(
        self,
        cfg: DictConfig,
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.train_idx: Optional[np.ndarray] = None
        self.val_idx: Optional[np.ndarray] = None
        self.test_idx: Optional[np.ndarray] = None
        self.label_names: Optional[list[str]] = None
    
    def prepare_data(self) -> None:
        for p in [self.cfg.paths.snp_csv, self.cfg.paths.pheno_csv]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Giessen file not found: {p}")
    
    def setup(self, stage: Optional[str] = None) -> None:
        snp_df = pd.read_csv(self.cfg.paths.snp_csv)
        pheno_df = pd.read_csv(self.cfg.paths.pheno_csv)
        
        # Merge on prename
        id_col = self.cfg.id_column
        merged = snp_df.merge(pheno_df, on=id_col, how="inner")
        
        # Features = all columns except id and antibiotic labels
        antibiotics = list(self.cfg.antibiotics)
        feature_cols = [c for c in snp_df.columns if c != id_col]
        
        self.X = merged[feature_cols].values.astype(np.int8)
        self.Y = merged[antibiotics].values.astype(np.int64)
        self.mask = ~np.isnan(merged[antibiotics].values)  # mask missing
        # Replace NaN with 0 in Y after mask captured
        self.Y[~self.mask] = 0
        self.label_names = antibiotics
        
        # Create stratified split
        ratios = list(self.cfg.split.ratios)
        seed = self.cfg.split.seed
        stratify_col = self.cfg.split.stratify_on
        stratify_idx = antibiotics.index(stratify_col)
        
        n = len(self.X)
        indices = np.arange(n)
        stratify_labels = self.Y[:, stratify_idx]
        
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=ratios[0],
            stratify=stratify_labels,
            random_state=seed,
        )
        val_ratio = ratios[1] / (ratios[1] + ratios[2])
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio,
            stratify=stratify_labels[temp_idx],
            random_state=seed,
        )
        
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
    
    def get_dataset(self, indices: np.ndarray) -> AMRTensorDataset:
        return AMRTensorDataset(self.X, self.Y, self.mask, indices)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset(self.train_idx),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset(self.val_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset(self.test_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )