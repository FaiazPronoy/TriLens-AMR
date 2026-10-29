from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from trilens.experiments.trilens_stack import (
    AB_NAMES,
    build_expert_a_blocks,
    build_expert_c_features,
    load_all_data,
    run_kfold_and_stack,
    save_representations,
)
from trilens.utils.seeds import set_global_seed

ROOT = Path(__file__).resolve().parents[2]
CONFIG_YAML_PATH = ROOT / "configs" / "config.yaml"
TRILENS_CFG_PATH = ROOT / "configs" / "experiment" / "trilens_stack.yaml"
CLEAN_GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered__phylogeny_clean.pkl"
OUTPUT_DIR = ROOT / "results" / "trilens_neural_phylogroup"
SEED = 42


def _build_cfg():
    base_cfg = OmegaConf.load(CONFIG_YAML_PATH)
    if "hydra" in base_cfg:
        del base_cfg["hydra"]
    if "wandb" in base_cfg:
        del base_cfg["wandb"]
    exp_cfg = OmegaConf.load(TRILENS_CFG_PATH)
    exp_cfg.output_dir = str(OUTPUT_DIR)
    exp_cfg.gene_filtered_pkl = str(CLEAN_GENE_FILTERED_PKL)

    exp_cfg.classical_predictions = str(OUTPUT_DIR / "_intentionally_absent_classical_predictions.npz")

    cfg = OmegaConf.merge(base_cfg, {"experiment": exp_cfg})
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
    return cfg.experiment


def main() -> None:
    if not CLEAN_GENE_FILTERED_PKL.exists():
        raise FileNotFoundError(
            f"{CLEAN_GENE_FILTERED_PKL} not found -- run trilens.experiments.gene_filtered_dataset_phylogroup first."
        )

    cfg = _build_cfg()
    set_global_seed(SEED)

    output_dir = Path(str(cfg.output_dir))
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    OmegaConf.save(cfg, output_dir / "config_used.yaml")

    assert not Path(str(cfg.classical_predictions)).exists(), (
        "classical_predictions path must NOT exist -- it would load random_split "
        "classical predictions against phylogeny_split labels."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data (clean gene-filtered pkl)...")
    data = load_all_data(cfg)

    from trilens.experiments.trilens_stack import _get_split_indices

    train_idx_phylo, _, _ = _get_split_indices(data, "phylogeny_split")

    print("Building Expert A blocks (fit on phylogeny_split train_idx)...")
    X_blocks, M_blocks, col_indices = build_expert_a_blocks(data, cfg, cache_dir, train_idx_phylo)

    print("Building Expert C features (fit on phylogeny_split train_idx)...")
    X_gene_agg, feat_stats = build_expert_c_features(data, train_idx_phylo, cache_dir)

    print("\nRunning full_stack_phylogroup (K-fold OOF + MoE stack) on phylogeny_split, clean gene set...")
    t0 = time.time()
    result = run_kfold_and_stack(
        "full_stack_phylogroup",
        cfg,
        data,
        X_blocks,
        M_blocks,
        X_gene_agg,
        "phylogeny_split",
        SEED,
        device,
        fold_dir=output_dir / "folds",
    )
    elapsed_min = (time.time() - t0) / 60
    print(f"full_stack_phylogroup (clean gene set) done in {elapsed_min:.1f} min")

    metrics = result["metrics"]
    rows = []
    for ab in AB_NAMES:
        m = metrics.get(ab, {})
        rows.append({"antibiotic": ab, "pr_auc": m.get("prauc"), "mcc": m.get("mcc")})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "trilens_neural_phylogroup_results.csv", index=False)
    print(df)
    print(f"Average PR-AUC: {df['pr_auc'].mean():.4f}")

    np.savez(
        str(output_dir / "predictions_saved.npz"),
        test_ens_probs=result["test_probs"],
        test_base_probs=result["test_base_probs"],
        oof_base_probs=result["oof_base_probs"],
        val_ens_probs=result.get("val_probs", np.zeros((0, len(AB_NAMES)))),
        test_indices=result["test_idx"],
        val_indices=result["val_idx"],
        train_indices=result["train_idx"],
        base_names=np.array(result.get("base_names", ["TriLens"])),
    )
    print(f"\nSaved predictions_saved.npz -> {output_dir}")
    save_representations(result, output_dir / "representations.npz")


if __name__ == "__main__":
    main()
