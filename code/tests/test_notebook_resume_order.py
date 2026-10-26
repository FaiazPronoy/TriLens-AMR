import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[2] / "kaggle" / "notebooks").glob("*.ipynb"))


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("_DATA_REPAIRED.txt", "repaired")
        z.writestr("_COMPLETE.txt", "failed steps:\n")
        z.writestr("_PA_FIXED.txt", "fixed")
        for name, text in files.items():
            z.writestr(name, text)
    return path


def _run_unpack(nb, tmp, releases, inputs=()):
    src = "".join(json.loads(nb.read_text(encoding="utf-8"))["cells"][1]["source"])
    start, end = src.index("def _target"), src.index("os.chdir(WORK)")
    gh = types.ModuleType("github_sync")
    gh.download_asset = lambda repo, token, tag, name, dest: releases.get(tag[len("outputs-"):])
    inp = tmp / "input"
    inp.mkdir()
    for z in inputs:
        z.rename(inp / z.name)
    res = tmp / "results"
    ns = {"Path": Path, "sys": sys, "zipfile": zipfile, "RES": res, "LENSB": "fixed",
          "FIXED_TRILENS": ("trilens_random", "trilens_splits"), "INPUT": inp,
          "NAME": "analysis", "NEEDS": ["trilens_random"], "GITHUB_REPO": "x/y", "TOKEN": None,
          "BASE": tmp, "WORK": tmp}
    saved = sys.modules.get("github_sync")
    sys.modules["github_sync"] = gh
    try:
        exec(src[start:end], ns)
    finally:
        if saved is None:
            sys.modules.pop("github_sync", None)
        else:
            sys.modules["github_sync"] = saved
    return res


NBS = [n for n in NOTEBOOKS if "def _target" in n.read_text(encoding="utf-8")]


@pytest.mark.parametrize("nb", NBS, ids=lambda n: n.stem)
def test_newer_upstream_release_beats_own_copy(nb, tmp_path):
    own = _zip(tmp_path / "analysis_outputs.zip", {"trilens_attnfix/a.csv": "old", "diagnostics/b.csv": "mine"})
    up = _zip(tmp_path / "trilens_random_outputs.zip", {"trilens_attnfix/a.csv": "new"})
    res = _run_unpack(nb, tmp_path, {"analysis": own, "trilens_random": up})
    assert (res / "trilens_attnfix" / "a.csv").read_text() == "new"
    assert (res / "diagnostics" / "b.csv").read_text() == "mine"


@pytest.mark.parametrize("nb", NBS, ids=lambda n: n.stem)
def test_attached_upstream_beats_own_release(nb, tmp_path):
    up = _zip(tmp_path / "trilens_random_outputs.zip", {"trilens_attnfix/a.csv": "new"})
    own = _zip(tmp_path / "own.zip", {"trilens_attnfix/a.csv": "old"})
    res = _run_unpack(nb, tmp_path, {"analysis": own}, inputs=[up])
    assert (res / "trilens_attnfix" / "a.csv").read_text() == "new"


@pytest.mark.parametrize("nb", NBS, ids=lambda n: n.stem)
def test_unfinished_upstream_is_refused(nb, tmp_path):
    up = tmp_path / "trilens_random_outputs.zip"
    with zipfile.ZipFile(up, "w") as z:                      # still running: no _COMPLETE.txt
        z.writestr("_DATA_REPAIRED.txt", "repaired")
        z.writestr("trilens_attnfix/a.csv", "partial")
    with pytest.raises(RuntimeError, match="not finished yet"):
        _run_unpack(nb, tmp_path, {"trilens_random": up})


@pytest.mark.parametrize("nb", NBS, ids=lambda n: n.stem)
def test_upstream_on_old_presence_matrix_is_refused(nb, tmp_path):
    up = tmp_path / "trilens_random_outputs.zip"
    with zipfile.ZipFile(up, "w") as z:                      # finished, but before pa_fix existed
        z.writestr("_DATA_REPAIRED.txt", "repaired")
        z.writestr("_COMPLETE.txt", "failed steps:\n")
        z.writestr("trilens_attnfix/a.csv", "old matrix")
    with pytest.raises(RuntimeError, match="old presence/absence matrix"):
        _run_unpack(nb, tmp_path, {"trilens_random": up})


@pytest.mark.parametrize("nb", NBS, ids=lambda n: n.stem)
def test_rebuilt_presence_matrix_replaces_the_dataset_copy(nb, tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (tmp_path / "dataset.pkl").write_text("old")
    (proc / "arcadia_v2_gene_presence_absence.pkl").symlink_to(tmp_path / "dataset.pkl")
    up = _zip(tmp_path / "pa_fix_outputs.zip", {"pa_fix/arcadia_v2_gene_presence_absence.pkl": "rebuilt"})
    src = "".join(json.loads(nb.read_text(encoding="utf-8"))["cells"][1]["source"])
    if '"pa_fix" in NEEDS' not in src:
        pytest.skip("notebook made before pa_fix")
    start, end = src.index("def _target"), src.index("os.chdir(WORK)")
    gh = types.ModuleType("github_sync")
    gh.download_asset = lambda repo, token, tag, name, dest: {"pa_fix": up}.get(tag[len("outputs-"):])
    inp = tmp_path / "input"
    inp.mkdir()
    ns = {"Path": Path, "sys": sys, "zipfile": zipfile, "RES": tmp_path / "results", "LENSB": "fixed",
          "FIXED_TRILENS": ("trilens_random", "trilens_splits"), "INPUT": inp, "NAME": "trilens_random",
          "NEEDS": ["pa_fix"], "GITHUB_REPO": "x/y", "TOKEN": None, "BASE": tmp_path, "WORK": tmp_path,
          "proc": proc, "RESUME_OWN": False}
    saved = sys.modules.get("github_sync")
    sys.modules["github_sync"] = gh
    try:
        exec(src[start:end], ns)
    finally:
        if saved is None:
            sys.modules.pop("github_sync", None)
        else:
            sys.modules["github_sync"] = saved
    assert (proc / "arcadia_v2_gene_presence_absence.pkl").read_text() == "rebuilt"
    assert ns["PA_STATE"] == "fixed"
