"""Config loading and path resolution.

Single source of truth for every filesystem path (B11). Nothing else in this
package should contain a literal path string.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PATHS_YAML = _REPO_ROOT / "config" / "paths.yaml"
_PARAMS_YAML = _REPO_ROOT / "config" / "params.yaml"


def detect_env() -> str:
    """Return "kaggle", "colab", or "local".

    Override with the AMR_ENV environment variable if this ever guesses
    wrong (e.g. a Kaggle-like mount point on a local machine).
    """
    override = os.environ.get("AMR_ENV")
    if override in ("kaggle", "colab", "local"):
        return override
    if Path("/kaggle/input").exists():
        return "kaggle"
    if Path("/content").exists():
        return "colab"
    return "local"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Paths:
    """Resolved absolute paths for the detected environment."""

    env: str
    project_root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    results_dir: Path
    raw_files: dict[str, str]
    interim_files: dict[str, str]
    processed_files: dict[str, str]
    legacy_files: dict[str, str]
    giessen_files: dict[str, str]
    card_files: dict[str, str]

    def raw(self, key: str) -> Path:
        return self.raw_dir / self.raw_files[key]

    def interim(self, key: str) -> Path:
        return self.interim_dir / self.interim_files[key]

    def processed(self, key: str) -> Path:
        return self.processed_dir / self.processed_files[key]

    def giessen(self, key: str) -> Path:
        p = Path(self.giessen_files[key])
        return p if p.is_absolute() else self.project_root / p

    def legacy(self, key: str) -> Path:
        p = Path(self.legacy_files[key])
        return p if p.is_absolute() else self.project_root / p


def load_paths(paths_yaml: Path | None = None) -> Paths:
    """Load and resolve config/paths.yaml for the current environment."""
    cfg = _load_yaml(paths_yaml or _PATHS_YAML)
    env = cfg.get("env", "auto")
    if env == "auto":
        env = detect_env()

    root_cfg = cfg["roots"][env]
    project_root = Path(root_cfg["project_root"])
    if not project_root.is_absolute():
        project_root = (_REPO_ROOT / project_root).resolve()

    def _resolve(key: str) -> Path:
        p = Path(root_cfg[key])
        return p if p.is_absolute() else (_REPO_ROOT / p).resolve()

    return Paths(
        env=env,
        project_root=project_root,
        raw_dir=_resolve("raw_dir"),
        interim_dir=_resolve("interim_dir"),
        processed_dir=_resolve("processed_dir"),
        results_dir=_resolve("results_dir"),
        raw_files=cfg["raw_files"],
        interim_files=cfg["interim_files"],
        processed_files=cfg["processed_files"],
        legacy_files=cfg.get("legacy_files", {}),
        giessen_files=cfg["giessen_files"],
        card_files=cfg.get("card_files", {}),
    )


def load_params(params_yaml: Path | None = None) -> dict[str, Any]:
    """Load config/params.yaml (locked pipeline parameters)."""
    return _load_yaml(params_yaml or _PARAMS_YAML)


def ensure_dirs(paths: Paths) -> None:
    """Create interim/processed/results directories if missing (not raw_dir --
    raw data is an input, this package never creates it)."""
    for d in (paths.interim_dir, paths.processed_dir, paths.results_dir):
        d.mkdir(parents=True, exist_ok=True)
