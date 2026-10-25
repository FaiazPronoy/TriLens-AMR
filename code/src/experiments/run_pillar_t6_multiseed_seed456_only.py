"""
src/experiments/run_pillar_t6_multiseed_seed456_only.py
==========================================================
Recovery script: seed=123 of run_pillar_t6_multiseed.py already completed
and saved (results/pillar_t6_multiseed/SEED_123/, full ablation_results.csv
+ predictions_saved.npz intact) before the machine hung and was force-
powered-off. seed=456 was mid-training (Fold 5/5, epoch 23/50) and lost no
saved output. Re-running seed=123 would waste ~2h40min of already-correct,
already-saved work, so this only runs seed=456, then calls the same
_aggregate() the original script would have -- which now finds both
SEED_123 and SEED_456 ablation_results.csv present and produces the
identical combined t6_multiseed_results.csv / SUMMARY.md as if both seeds
had run in one invocation.

Does not modify run_pillar_t6_multiseed.py -- imports run_one_seed() and
_aggregate() directly, unchanged.

Usage:
    conda activate amr
    python -m src.experiments.run_pillar_t6_multiseed_seed456_only
"""
from __future__ import annotations

from src.experiments.run_pillar_t6_multiseed import OUTPUT_DIR, _aggregate, run_one_seed


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed123_path = OUTPUT_DIR / "SEED_123" / "ablation_results.csv"
    if not seed123_path.exists():
        raise FileNotFoundError(
            f"{seed123_path} not found -- seed=123 was expected to already be "
            f"complete. If it is genuinely missing, use the original "
            f"run_pillar_t6_multiseed.py (both seeds) instead."
        )
    print(f"seed=123 already complete ({seed123_path}) -- skipping, running seed=456 only.")
    run_one_seed(456)
    _aggregate()


if __name__ == "__main__":
    main()
