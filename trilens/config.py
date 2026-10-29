from __future__ import annotations

from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_config(experiment: str, overrides: Sequence[str] = ()) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=[f"experiment={experiment}", *overrides])


def show(cfg: DictConfig) -> None:
    print(f"experiment: {cfg.experiment.name}")
    print(OmegaConf.to_yaml(cfg.experiment))
