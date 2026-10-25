"""
Proof that Lens B's attention cannot depend on the sample.
==========================================================

In models/trilens.py, ExpertB.forward computes

    weighted = E_pa.unsqueeze(0) * X_pa.unsqueeze(-1)
    e        = attn_w( tanh(attn_V(weighted)) * sigmoid(attn_U(weighted)) )

X_pa is binary. For a gene that IS present, weighted[b,k,:] == E_pa[k,:], which
does not depend on b. So the pre-softmax attention score of every present gene is
the same number for every strain. The softmax only renormalises over whichever
genes that strain happens to carry.

Consequence: the lens cannot rank genes differently for different strains. It has
one global ranking, filtered by presence. That is not what gated attention-MIL
(Ilse et al. 2018) does, and it explains why the attention is near-uniform, why
the top genes are nearly identical across all seven antibiotics, and why those
genes carry no lineage and no resistance information.

This script demonstrates it on an untrained module in a few seconds. It needs no
data and no checkpoint - the property is structural, not learned.

    conda activate amr
    python scripts/diagnostics/prove_lens_b_invariance.py
"""

import sys
from pathlib import Path

import torch

# Import EXACTLY the way run_trilens_v10.py does, so this tests the same file
# that training will use:   from src.models.trilens import AB_NAMES, TriLens
ROOT = Path(__file__).resolve().parents[2]   # the code/ folder
if not (ROOT / "src" / "models" / "trilens.py").exists():
    sys.exit(f"Expected {ROOT / 'src' / 'models' / 'trilens.py'} - keep this script in code/scripts/diagnostics/.")
sys.path.insert(0, str(ROOT))

import src.models.trilens as _tl            # noqa: E402
from src.models.trilens import ExpertB      # noqa: E402

print("=" * 74)
print("WHICH FILE IS ACTUALLY LOADED?")
print("=" * 74)
print(f"  {_tl.__file__}")

torch.manual_seed(0)

G_PA, D_TOKEN, D_OUT, ATTN = 2000, 32, 64, 32
expert = ExpertB(G_pa=G_PA, d_token=D_TOKEN, d_out=D_OUT, attn_dim=ATTN).eval()

# --------------------------------------------------------------------------
# Simulate a TRAINED network.
#
# At random init this test cannot distinguish the two versions, because the
# strain context is a mean over ~800 zero-mean Xavier vectors, so it collapses
# to almost exactly 0 and the context term contributes nothing. After training,
# E_pa is structured and the context projections have learned weights.
#
# We reproduce that by giving E_pa structure along the gene axis, so that
# strains carrying different gene sets genuinely have different contexts, and
# by scaling the context projections to a realistic magnitude.
# --------------------------------------------------------------------------
with torch.no_grad():
    ramp = torch.linspace(-1.0, 1.0, G_PA).unsqueeze(1)      # (G_PA, 1)
    expert.E_pa.copy_(expert.E_pa * 0.3 + ramp)              # gene-dependent structure
    if hasattr(expert, "attn_Vc"):
        expert.attn_Vc.weight.mul_(5.0)
        expert.attn_Uc.weight.mul_(5.0)

has_fix = hasattr(expert, "attn_Vc") and hasattr(expert, "attn_Uc")
print(f"  Has attn_Vc / attn_Uc (the fix): {has_fix}")
if not has_fix:
    print("""
  ^^^ THE FIX IS NOT IN THE FILE BEING IMPORTED.

  You edited one copy of trilens.py and Python is loading a different one.
  The path printed above is the file that actually ran. Either edit that file,
  or delete the stale copy. Common causes:

    1. A second copy at code/models/trilens.py, which this script checks
       BEFORE code/src/models/trilens.py
    2. The project pip-installed non-editably, so site-packages has a frozen
       copy. Fix with:   pip install -e .
    3. A stale __pycache__ - delete code/src/models/__pycache__/

  Nothing below this line is meaningful until that path points at your edit.
""")
print()

# Two very different strains that nevertheless share some genes.
X = torch.zeros(2, G_PA)
X[0, :800] = 1.0                 # strain 0 carries genes 0-799
X[1, 400:1400] = 1.0             # strain 1 carries genes 400-1399
shared = torch.arange(400, 800)  # genes both strains carry

with torch.no_grad():
    _, a, _ = expert(X)

print("=" * 74)
print("LENS B ATTENTION - IS IT SAMPLE-DEPENDENT?")
print("=" * 74)
print(f"Strain 0 carries {int(X[0].sum())} genes, strain 1 carries {int(X[1].sum())}.")
print(f"They share {len(shared)} genes (indices 400-799).\n")

a0 = a[0, shared]
a1 = a[1, shared]

# The softmax denominators differ, so compare the RANKING and the ratios.
r0 = torch.argsort(a0, descending=True)
r1 = torch.argsort(a1, descending=True)
same_order = bool(torch.equal(r0, r1))

ratio = (a0 / a1)
ratio_spread = float(ratio.max() - ratio.min())

print(f"Ranking of the shared genes identical for both strains : {same_order}")
print(f"Spread of the per-gene attention ratio a0/a1           : {ratio_spread:.3e}")
print(f"   (exactly 0 means every shared gene was rescaled by the")
print(f"    same constant, i.e. only the softmax denominator changed)\n")

# NOTE: we deliberately do NOT reimplement the score here. Reimplementing it was
# a bug in an earlier version of this script - it hard-coded the old formula, so
# it reported the old behaviour no matter what the model actually did. The ratio
# spread above is computed from the module's own output and is the real test:
#
#   sample-INVARIANT scores  ->  a0[k]/a1[k] = Z1/Z0 for every shared gene k,
#                                i.e. one constant, so the spread is ~0
#   sample-DEPENDENT scores  ->  the ratio varies gene by gene, spread >> 0
diff = ratio_spread

print("\n" + "-" * 74)
# A spread below ~1e-3 is float32 noise on attention values of order 1e-3.
# Anything above that means the two strains weight the same gene differently.
if ratio_spread < 1e-3:
    print("BUG PRESENT — attention is sample-invariant.")
    print()
    print(f"Every shared gene was rescaled by the same constant "
          f"(ratio spread {ratio_spread:.2e}).")
    print("Lens B has one global gene ranking, filtered by presence.")
    print("It cannot express per-sample importance.")
else:
    print("FIX VERIFIED — attention is sample-dependent.")
    print()
    print(f"The same gene receives materially different weight in the two "
          f"strains\n(ratio spread {ratio_spread:.2e}, ranking differs: {not same_order}).")
    print("Lens B can now rank genes differently for different samples.")
print("-" * 74)

print("""
WHY THIS MATTERS

  The thesis reports that Lens B's attention recovered no resistance gene, was
  near-uniform for five of seven drugs, and selected genes carrying neither
  lineage nor resistance information. All three follow from the property above.
  It is a structural limitation of multiplying a learned embedding by a binary
  presence indicator, not a failure of attention pooling in general - Lens A and
  Lens C, whose inputs vary per sample, do not have it.

MINIMAL FIX (for future work, NOT before the defence)

  Condition the attention on the strain, e.g.

      ctx      = weighted.sum(1) / X_pa.sum(1, keepdim=True).clamp(min=1)
      q        = torch.cat([weighted, ctx.unsqueeze(1).expand_as(weighted)], -1)
      e        = attn_w(tanh(attn_V(q)) * sigmoid(attn_U(q))).squeeze(-1)

  and widen attn_V / attn_U to accept 2 * d_token. Alternatively replace the
  softmax with entmax15, which results/gene_attention_v2/attention_sparsity.csv
  already shows produces genuinely sparse attention on this data (E2: 21 genes
  above 1%, entropy 5.21 against a maximum of 8.52).
""")
