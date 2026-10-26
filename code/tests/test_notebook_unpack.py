import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[2] / "kaggle" / "notebooks").glob("*.ipynb"))


def _unpack_helpers(nb, res, lensb):
    src = "".join(json.loads(nb.read_text(encoding="utf-8"))["cells"][1]["source"])
    start, end = src.index("def _target"), src.index("for z in sorted(INPUT.rglob")
    ns = {"Path": Path, "RES": res, "LENSB": lensb, "NAME": "analysis",
          "FIXED_TRILENS": ("trilens_random", "trilens_splits")}
    exec(src[start:end], ns)
    return ns


@pytest.mark.parametrize("nb", [n for n in NOTEBOOKS if "USES_RANDOM_GF" in n.read_text(encoding="utf-8")],
                         ids=lambda n: n.stem)
@pytest.mark.parametrize("lensb", ["fixed", "standard"])
def test_pre_repair_outputs_are_dropped_with_or_without_prefix(nb, lensb, tmp_path):
    ns = _unpack_helpers(nb, tmp_path, lensb)
    ns["_pre_repair"].add("standard_random")
    take = lambda rel: ns["_take"]("standard_random", rel, lambda: b"x")
    assert take("standard/trilens_attnfix/ablation_results.csv") == 0
    assert take("trilens_attnfix/ablation_results.csv") == 0
    assert take("standard/pillar_trilens_mlst/trilens_mlst_results.csv") == 1
