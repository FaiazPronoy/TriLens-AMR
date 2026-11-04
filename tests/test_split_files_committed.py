import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_committed_split_files_are_train_val_test():
    files = sorted((ROOT / "results" / "splits").glob("*.json"))
    assert files
    for p in files:
        d = json.loads(p.read_text())
        assert {"name", "train_idx", "val_idx", "test_idx", "unused_idx", "rules"} <= set(d), p.name
