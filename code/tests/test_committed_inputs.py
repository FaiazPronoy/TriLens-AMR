"""Files the notebooks read from the cloned repository must be tracked by git, not just present on
disk (a .gitignore rule once dropped pd_amr_joined.tsv.gz silently)."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NEEDED = ["results/data_inventory/genome_ids.csv", "results/data_inventory/strain_table.csv",
          "results/data_inventory/labels16.csv", "results/data_inventory/covariates.csv",
          "results/data_inventory/pd_amr_joined.tsv.gz", "results/data_inventory/pd_amr_joined.json",
          "results/splits_kfold/st_grouped_5fold.json", "results/splits_kfold/random_5fold.json",
          "results/splits/country_uk_to_norway.json", "results/splits/country_norway_to_uk.json",
          "results/splits/time_norway_2013.json"]


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT.parent / ".git").exists(), reason="not a git checkout")
def test_notebook_inputs_are_tracked():
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split())
    missing = [f for f in NEEDED if f not in tracked]
    assert not missing, f"not committed (check .gitignore): {missing}"
