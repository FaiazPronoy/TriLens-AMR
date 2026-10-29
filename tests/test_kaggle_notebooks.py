import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "kaggle" / "notebooks").glob("*.ipynb"))
CHOICE_FLAGS = ("--part", "--split", "--model", "--attention")


def commands():
    found = set()
    for nb in NOTEBOOKS:
        cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
        text = "".join("".join(c["source"]) for c in cells)
        for m in re.finditer(r"python -m (pipeline\.\w+)([^\"'\n]*)", text):
            found.add((m.group(1), m.group(2).strip()))
    return sorted(found)


def test_every_notebook_is_valid_json():
    assert NOTEBOOKS
    for nb in NOTEBOOKS:
        data = json.loads(nb.read_text(encoding="utf-8"))
        assert data["cells"]


def test_generator_output_is_up_to_date(tmp_path):
    before = {p.name: p.read_text(encoding="utf-8") for p in NOTEBOOKS}
    subprocess.run(
        [sys.executable, str(ROOT / "kaggle" / "make_notebooks.py")], check=True, capture_output=True, cwd=ROOT
    )
    after = {p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "kaggle" / "notebooks").glob("*.ipynb"))}
    assert before == after


@pytest.mark.parametrize("module,args", commands())
def test_notebook_command_uses_a_real_step_and_choice(module, args):
    assert (ROOT / (module.replace(".", "/") + ".py")).exists()
    help_text = subprocess.run(
        [sys.executable, "-m", module, "--help"], capture_output=True, text=True, cwd=ROOT
    ).stdout
    tokens = args.split()
    for i, tok in enumerate(tokens[:-1]):
        if tok in CHOICE_FLAGS:
            value = tokens[i + 1]
            assert re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", help_text), (module, tok, value)
