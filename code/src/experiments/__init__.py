# src/experiments/__init__.py
"""Pillar 1 experiment runners.

Each module exposes a single ``run(cfg)`` entry-point that consumes a Hydra
config and writes results under ``${paths.results_dir}/...``.
"""
