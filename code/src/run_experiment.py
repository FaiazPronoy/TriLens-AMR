# src/run_experiment.py
"""Hydra entry-point for all experiments.

This file *replaces* the Stage 0 stub. Stage 0 helpers
(``src/utils/seeds.set_global_seed``, ``src/utils/logging.init_wandb_logger``,
the Lightning DataModules, the metrics/statistical modules) are still
imported and used by the individual experiment runners — only the
top-level dispatch is updated here.

Usage examples (run from inside the code/ folder):
    python -m src.run_experiment experiment=pillar1_exp1
    python -m src.run_experiment experiment=pillar1_exp2
    python -m src.run_experiment experiment=pillar1_exp3
    python -m src.run_experiment experiment=baseline_xgb_phylogroup
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.seeds import set_global_seed

_DISPATCH = {
    "pillar1_exp1": "src.experiments.run_pillar1_exp1",
    "pillar1_exp2": "src.experiments.run_pillar1_exp2",
    "pillar1_exp3": "src.experiments.run_pillar1_exp3",
    "baseline_xgb_phylogroup": "src.experiments.run_baseline_xgb_phylogroup",
    "stage1_5_gene_filtering": "src.experiments.run_stage1_5",
    "stage2_gene_xgb": "src.experiments.run_stage2_gene_xgb",
    "stage2_gene_attention": "src.experiments.run_stage2_gene_attention",
    "stacking_ensemble": "src.experiments.run_stacking_ensemble",
    "stage3_cascadestack_v7": "src.experiments.run_cascadestack_v7",
    "stage3_cascadestack_v8": "src.experiments.run_cascadestack_v8",
    "stage3_cascadestack_v9": "src.experiments.run_cascadestack_v9",
    "stage3_deepmeta_v11": "src.experiments.run_deepmeta_v11",
    "pillar3_interpretability": "src.experiments.run_pillar3_interpretability",
    "pillar3_xgb_importance": "src.experiments.run_pillar3_xgb_importance",
    "pillar3_shap": "src.experiments.run_pillar3_shap",
    "pillar4_giessen_transfer": "src.experiments.run_pillar4_giessen_transfer",
    "gene_attention_v2": "src.experiments.run_gene_attention_v2",
    "posthoc_analysis": "src.experiments.run_posthoc_analysis",
    "phylogeny_split_eval": "src.experiments.run_phylogeny_split_eval",
    "trilens_v10": "src.experiments.run_trilens_v10",
}

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)

    exp_name = cfg.experiment.name
    print("\n" + "=" * 80)
    print(f"  HYDRA DISPATCH — experiment.name = {exp_name!r}")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg.experiment))

    if exp_name not in _DISPATCH:
        raise ValueError(
            f"Unknown experiment {exp_name!r}. "
            f"Available: {sorted(_DISPATCH)}"
        )

    mod_name = _DISPATCH[exp_name]
    import importlib
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, "run"):
        raise AttributeError(f"{mod_name} has no ``run(cfg)`` function.")
    mod.run(cfg)


if __name__ == "__main__":
    main()
