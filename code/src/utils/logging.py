"""WandB logging utilities."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from omegaconf import DictConfig, OmegaConf

import wandb
from lightning.pytorch.loggers import WandbLogger


def init_wandb_logger(cfg: DictConfig, experiment_name: str) -> WandbLogger:
    """Initialize WandB logger with full config tracking.
    
    Args:
        cfg: Hydra config object.
        experiment_name: Unique name for this run.
    
    Returns:
        Configured WandbLogger instance.
    """
    # Build run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_{timestamp}"
    
    # Tags for filtering
    tags = [
        experiment_name,
        cfg.data.name,
        cfg.get("model", {}).get("type", "unknown"),
        cfg.data.get("split", "random"),
    ]
    
    # Convert OmegaConf to dict for WandB
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        tags=tags,
        config=config_dict,
        mode=cfg.wandb.mode,
        log_model=cfg.wandb.log_model,
        save_dir=str(Path(cfg.paths.logs_dir)),
    )
    
    return logger


def log_metrics_dict(metrics: dict[str, float], prefix: str = "") -> None:
    """Log a dictionary of metrics to active WandB run.
    
    Args:
        metrics: {metric_name: value} dictionary.
        prefix: Prefix to prepend to metric names (e.g., 'test_').
    """
    if wandb.run is None:
        return
    prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}
    wandb.log(prefixed)