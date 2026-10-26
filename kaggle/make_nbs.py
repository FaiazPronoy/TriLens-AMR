"""Generates the Kaggle notebooks of the full re-run.

    python kaggle/make_nbs.py kaggle/notebooks

Edit the notebooks here, never by hand: every notebook shares the same setup cell.
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])

H = "python -m src.run_experiment experiment="
M = "python -m src.experiments."
D = "python scripts/diagnostics/"

STAGES = {
    "pa_fix": dict(
        title="Gene presence/absence: rebuild with the correct orientation",
        lensb="fixed", needs=[], gpu="None (CPU is enough; it saves GPU quota)",
        about="The dataset's `arcadia_v2_gene_presence_absence.pkl` marks the minority state of each "
              "pan-genome locus, not its presence (code/docs/KNOWN_ISSUES.md, issue 12). This notebook "
              "rebuilds it from Arcadia's PLINK files using the allele letters in the .bim file, and "
              "refuses if anything but the orientation would change. Every notebook that reads the "
              "matrix (the four TriLens notebooks and analysis) takes the rebuilt copy from here. "
              "Optional input: the raw-download dataset with presence_absence_data/; without it the "
              "release is downloaded from Zenodo (2.4 GB, a few minutes).",
        writes=["pa_fix"],
        steps=[
            ("Find Arcadia's PLINK presence/absence files",
             "python scripts/fetch_gene_pa_source.py --dest /tmp/trilens/pa_source",
             "pa_fix/arcadia_v2_gene_presence_absence.pkl"),
            ("Rebuild the matrix (refuses if more than the orientation changes)",
             "python scripts/repair_gene_pa.py --source /tmp/trilens/pa_source/paths.json "
             "--old data/processed/arcadia_v2_gene_presence_absence.pkl "
             "--out results/pa_fix/arcadia_v2_gene_presence_absence.pkl --report results/pa_fix/pa_repair.json",
             "pa_fix/arcadia_v2_gene_presence_absence.pkl"),
        ]),
    "classical": dict(
        title="Classical arm, baselines, Giessen and SHAP",
        lensb="fixed", needs=[], gpu="GPU T4 x2 (XGBoost runs on the GPU)",
        about="The five classical base learners on all three splits (the classical half of the "
              "stack), every external baseline, the Giessen transfer and the SHAP/importance runs. "
              "Every other notebook needs this one's output.",
        writes=["stage3_cascadestack_v9", "pillar_cascadestack_classical_phylogeny_clean",
                "pillar_cascadestack_classical_mlst", "baselines", "pillar_phylogroup_only",
                "tabnet_baseline", "pillar_tabnet_phylogeny", "pillar_tabnet_mlst",
                "pillar4_giessen_transfer", "pillar3_xgb_importance", "pillar3_shap", "pillar_shap_t6"],
        steps=[
            ("Classical stack, random split (also the 6th learner, GeneAttentionNet)",
             H + "stage3_cascadestack_v9", "stage3_cascadestack_v9/predictions_saved.npz"),
            ("Classical stack, phylogroup split", M + "run_cascadestack_classical_phylogeny_clean",
             "pillar_cascadestack_classical_phylogeny_clean/classical5_results.csv"),
            ("Classical stack, lineage holdout", M + "run_cascadestack_classical_mlst",
             "pillar_cascadestack_classical_mlst/classical5_results.csv"),
            ("Baseline: XGBoost on 1,024 PCA components", H + "baseline_xgb_phylogroup",
             "baselines/xgb_phylogroup_results.csv"),
            ("Baseline: phylogroup label only", M + "run_phylogroup_only_baseline",
             "pillar_phylogroup_only/phylogroup_only_results.csv"),
            ("Baseline: TabNet, random split", M + "run_tabnet_baseline", "tabnet_baseline/tabnet_results.csv"),
            ("Baseline: TabNet, phylogroup split", M + "run_tabnet_baseline_phylogeny",
             "pillar_tabnet_phylogeny/tabnet_phylogeny_results.csv"),
            ("Baseline: TabNet, lineage holdout", M + "run_tabnet_baseline_mlst",
             "pillar_tabnet_mlst/tabnet_mlst_results.csv"),
            ("External validation: Giessen", H + "pillar4_giessen_transfer",
             "pillar4_giessen_transfer/pillar4_per_ab_metrics.csv"),
            ("Interpretability: XGBoost gain importance", H + "pillar3_xgb_importance",
             "pillar3_xgb_importance/multi_drug_markers.csv"),
            ("Interpretability: SHAP", H + "pillar3_shap", "pillar3_shap/multi_drug_markers_shap.csv"),
            ("Interpretability: SHAP on the five stack learners", M + "run_pillar_shap_t6",
             "pillar_shap_t6/concordance_metrics.csv"),
        ]),
    "trilens_random": dict(
        title="TriLens, random split, and the layer ablation",
        lensb="fixed", needs=["pa_fix", "classical"], gpu="GPU T4 x2",
        about="The delivered model (strain-conditioned Lens B) on the random split: every headline "
              "number of the paper. Then the three-stage layer ablation on the same run.",
        writes=["trilens_attnfix", "pillar2_layer_ablation"],
        steps=[
            ("TriLens-MoE-Stack, random split", H + "trilens_v10", "trilens_attnfix/predictions_saved.npz"),
            ("Layer ablation, stage 1 (lenses without the gate)", M + "run_pillar2_stage1_expert_only",
             "pillar2_layer_ablation/stage1_expert_only_preds.npz"),
            ("Layer ablation, table", M + "run_pillar2_layer_ablation",
             "pillar2_layer_ablation/layer_ablation_results.csv"),
        ]),
    "trilens_splits": dict(
        title="TriLens, phylogroup split and lineage holdout",
        lensb="fixed", needs=["pa_fix", "classical"], gpu="GPU T4 x2",
        about="The neural half of the model on the two harder splits. The analysis notebook stacks "
              "it with the classical half.",
        writes=["pillar_trilens_t7_clean_gene_set", "pillar_trilens_mlst"],
        steps=[
            ("TriLens, phylogroup split", M + "run_trilens_t7_clean_gene_set",
             "pillar_trilens_t7_clean_gene_set/t7_clean_results.csv"),
            ("TriLens, lineage holdout", M + "run_trilens_mlst", "pillar_trilens_mlst/trilens_mlst_results.csv"),
        ]),
    "standard_random": dict(
        title="Strain-conditioning ablation: standard Lens B, random split",
        lensb="standard", needs=["pa_fix", "classical"], gpu="GPU T4 x2",
        about="The same model with standard gated attention in Lens B (TRILENS_LENSB=standard). "
              "Everything else is identical. This is the 'Standard' column of tab:attnfix.",
        writes=["trilens_attnfix"],
        steps=[("TriLens with standard Lens B, random split", H + "trilens_v10",
                "trilens_attnfix/predictions_saved.npz")]),
    "standard_splits": dict(
        title="Strain-conditioning ablation: standard Lens B, phylogroup split and lineage holdout",
        lensb="standard", needs=["pa_fix", "classical"], gpu="GPU T4 x2",
        about="Standard Lens B on the two harder splits.",
        writes=["pillar_trilens_t7_clean_gene_set", "pillar_trilens_mlst"],
        steps=[
            ("Standard Lens B, phylogroup split", M + "run_trilens_t7_clean_gene_set",
             "pillar_trilens_t7_clean_gene_set/t7_clean_results.csv"),
            ("Standard Lens B, lineage holdout", M + "run_trilens_mlst",
             "pillar_trilens_mlst/trilens_mlst_results.csv"),
        ]),
    "shuffle_test": dict(
        title="Locality check: position-shuffling test",
        lensb="fixed", needs=[], gpu="GPU T4 x2",
        about="The plain 1D CNN on the whole pan-genome, with and without shuffled variant order "
              "(the paper's locality argument for the Motif Pattern Encoder). Independent of the rest.",
        writes=["pillar1"],
        steps=[(f"1D CNN, {d} {o}", f"EXP3_ONLY={d}_CIP_{o} " + H + "pillar1_exp3",
                f"pillar1/exp3/{d}_CIP_{o}_run.json")
               for d in ("giessen", "arcadia") for o in ("orig", "shuf")]
              + [("Ordered against shuffled: summary", H + "pillar1_exp3", "pillar1/exp3/exp3_results.csv")]),
    "analysis": dict(
        title="Analysis: stacking, diagnostics, ablation table, figures, paper numbers",
        lensb="fixed", needs=["pa_fix", "classical", "trilens_random", "trilens_splits", "standard_random",
                              "standard_splits", "shuffle_test"],
        gpu="None (CPU is enough here; it saves GPU quota)",
        about="Reads the outputs of all other notebooks. Nothing is trained here except the "
              "per-split meta-learners. The last cell is the table of every paper number, old "
              "against new.",
        writes=["*"],
        steps=[
            ("Phylogroup split: stack neural + classical", M + "diagnose_t7_clean_final_comparison",
             "pillar_t7_clean_final_comparison/three_way_comparison_FINAL.csv"),
            ("Lineage holdout: stack neural + classical", M + "diagnose_mlst_final_comparison",
             "pillar_mlst_final_comparison/three_way_comparison_FINAL.csv"),
            ("Prevalence-matched holdout", D + "matched_prevalence_check.py", "diagnostics/matched_prevalence.csv"),
            ("Clinical error rates under the holdout", D + "holdout_clinical_metrics.py",
             "diagnostics/holdout_clinical_metrics.csv"),
            ("Threshold tuning", D + "threshold_tuning.py", "diagnostics/threshold_tuning.csv"),
            ("Near-duplicate check, random split", M + "run_pillar_dedup_eval", "pillar_dedup_eval/dedup_eval_results.csv"),
            ("Near-duplicate check, lineage holdout", M + "run_pillar_dedup_eval_mlst",
             "pillar_dedup_eval/dedup_eval_results_mlst.csv"),
            ("Full metric suite (F1, F2, accuracy)", M + "run_pillar_full_metric_suite",
             "pillar_metrics_full/full_metric_suite.csv"),
            ("Prevalence-invariant metrics", M + "diagnose_prevalence_invariant_metrics",
             "pillar_prevalence_invariant_metrics/prevalence_invariant_metrics.csv"),
            ("Gene annotation of the top genes", M + "run_pillar_gene_annotation",
             "pillar_gene_annotation/annotated_markers.csv"),
            ("Lineage-marker test", D + "lineage_marker_test.py", "diagnostics/lineage_marker_test.csv"),
            ("Lens B prefix base rate", D + "prefix_baserate_lens_b.py", "diagnostics/prefix_baserate_lens_b.csv"),
            ("Attention diagnosis", D + "attention_diagnosis.py", "diagnostics/attention_diagnosis.csv"),
            ("Pillar 1 summary", M + "aggregate_pillar1_results --paper-outline /tmp/outline.md",
             "pillar1/pillar1_combined.csv"),
            ("Leakage and split checks (a FAIL stops the paper numbers being used)",
             D + "check_leakage.py", None),
        ]),
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


SETUP = r'''
# 1. Set up: code, data, and the outputs of the earlier notebooks
GITHUB_REPO = "FaiazPronoy/TriLens-AMR"   # where the code lives and the outputs go
CODE_REF    = "main"                       # branch, tag or commit to run
RESUME_OWN  = {resume_own}                       # continue from this notebook's own release

NAME   = "{name}"
LENSB  = "{lensb}"          # "fixed" = delivered model, "standard" = ablation model
WRITES = {writes}
NEEDS  = {needs}
NEEDS_GPU = {needs_gpu}

import os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

INPUT   = Path("/kaggle/input")
BASE    = Path("/tmp/trilens")           # outside /kaggle/working: only the zip is saved as output
WORK    = BASE / "code"
RES     = WORK / "results"
OUT_ZIP = Path(f"/kaggle/working/{{NAME}}_outputs.zip")
PREFIX  = "standard/" if LENSB == "standard" else ""
STANDARD_STAGES = ("standard_random", "standard_splits")
FIXED_TRILENS   = ("trilens_random", "trilens_splits")

def find_one(name):
    hits = sorted(INPUT.rglob(name))
    return hits[0] if hits else None

def _secret(key):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key)
    except Exception:
        return os.environ.get(key)

TOKEN = _secret("GITHUB_TOKEN")      # Add-ons -> Secrets -> GITHUB_TOKEN, switched on for this notebook
print("github :", GITHUB_REPO, "(token found)" if TOKEN else "(no token: nothing is uploaded)")

# code: the dataset trilens-code if attached, otherwise a clone of the repository
entry = find_one("run_experiment.py")
if entry is not None:
    code_src, CODE_VERSION = entry.parent.parent, "dataset trilens-code"
else:
    repo_dir = BASE / "repo"
    if not repo_dir.exists():
        url = f"https://github.com/{{GITHUB_REPO}}.git"
        auth = []        # the token goes in a header for this one command, never into .git/config
        if TOKEN:
            import base64
            auth = ["-c", "http.extraheader=AUTHORIZATION: basic "
                    + base64.b64encode(f"x-access-token:{{TOKEN}}".encode()).decode()]
        r = subprocess.run(["git", *auth, "clone", "--quiet", url, str(repo_dir)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("git clone failed: " + (r.stderr or "").replace(TOKEN or "\0", "***"))
        subprocess.run(["git", "checkout", "--quiet", CODE_REF], cwd=repo_dir, check=True)
    code_src = repo_dir / "code"
    CODE_VERSION = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                                  capture_output=True, text=True).stdout.strip()
print("version:", CODE_VERSION)
if not WORK.exists():
    def _skip(d, names):
        top = {{"data", "results", "experiments", "outputs"}} if Path(d) == code_src else set()
        return [n for n in names if n in top or n == "__pycache__"]
    shutil.copytree(code_src, WORK, ignore=_skip)
print("code   :", code_src)

# data (dataset trilens-data): symlinks, never copies
master = find_one("arcadia_v2_master_features.pkl")
if master is None:
    raise FileNotFoundError("add the dataset trilens-data (Add Input)")
proc, giessen = WORK / "data" / "processed", WORK / "data" / "Giessen_dataset"
proc.mkdir(parents=True, exist_ok=True); giessen.mkdir(parents=True, exist_ok=True)
for f in sorted(master.parent.iterdir()):
    if f.is_file():
        dst = (giessen if f.name.startswith("cip_ctx_ctz_gen") else proc) / f.name
        if not dst.exists():
            dst.symlink_to(f)
for n in ("cip_ctx_ctz_gen_multi_data.csv", "cip_ctx_ctz_gen_pheno.csv"):
    f = find_one(n)
    if f is not None and not (giessen / n).exists():
        (giessen / n).symlink_to(f)
print("data   :", master.parent, f"({{len(list(proc.iterdir()))}} files)")

# The random-split gene-filtered file has 42 columns with values outside the SNP codes
# 0..3 (found by data_check). Re-read those columns from the SNP matrix; the script
# refuses if anything else in the file differs from the matrix. Report: results/diagnostics/.
gfp = proc / "arcadia_v2_gene_filtered.pkl"
fixed = BASE / "fixed" / gfp.name
DATA_STATE = "repaired" if (gfp.is_symlink() and Path(os.readlink(gfp)) == fixed) else "unchecked"
if gfp.is_symlink() and Path(os.readlink(gfp)) != fixed:
    r = subprocess.run([sys.executable, str(WORK / "scripts" / "repair_gene_filtered.py"),
                        "--src", str(gfp.resolve()), "--snp", str((proc / "arcadia_v2_snp_matrix.pkl").resolve()),
                        "--out", str(fixed), "--report", str(WORK / "results" / "diagnostics" / "data_repair.json")],
                       capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if '"status"' in l or "columns" in l))
    if r.returncode == 0:
        gfp.unlink(); gfp.symlink_to(fixed)
        DATA_STATE = "repaired"
        print("data   : arcadia_v2_gene_filtered.pkl repaired ->", fixed)
    elif r.returncode == 2:
        DATA_STATE = "clean"
    else:
        raise RuntimeError("repair_gene_filtered refused the file - read the report above:\n" + r.stdout[-2000:] + r.stderr[-2000:])

# outputs of earlier notebooks: every *_outputs.zip under /kaggle/input
# (a notebook output, or a dataset you uploaded it to). The standard-model notebooks
# keep their files under results/standard/; inside those notebooks they are unpacked
# to results/ itself, because there the standard model IS the main run.
RES.mkdir(parents=True, exist_ok=True)

def _target(stage, member):
    if LENSB == "standard":
        if stage in FIXED_TRILENS:
            return None                       # never mix the two models
        if member.startswith("standard/"):
            member = member[len("standard/"):]
    return RES / member

# Runs that read the random-split gene-filtered file. Their outputs from a run made
# before the repair (no _DATA_REPAIRED.txt in the zip) are thrown away and recomputed.
USES_RANDOM_GF = {{"stage3_cascadestack_v9", "tabnet_baseline", "pillar_tabnet_phylogeny", "pillar_tabnet_mlst",
                  "pillar3_xgb_importance", "pillar3_shap", "pillar_shap_t6", "trilens_attnfix",
                  "pillar2_layer_ablation", "pillar_dedup_eval", "pillar_metrics_full"}}
_pre_repair = set()
_unfinished = set()      # stages whose output has no _COMPLETE.txt (still running, or stopped)
_pa_old = set()          # stages whose output was made with the old presence/absence matrix

def _note(stage, names):
    if "_DATA_REPAIRED.txt" not in names:
        _pre_repair.add(stage)
    if "_COMPLETE.txt" not in names:
        _unfinished.add(stage)
    if "_PA_FIXED.txt" not in names:
        _pa_old.add(stage)

def _take(stage, rel, read):
    t = _target(stage, rel)
    if t is None or rel.split("/")[0].startswith("_"):
        return 0
    if stage == NAME and t.exists():
        return 0                          # a newer copy from an earlier stage is already here
    if stage in _pre_repair and rel.removeprefix("standard/").split("/")[0] in USES_RANDOM_GF:
        return 0
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_bytes(read())
    return 1

for z in sorted(INPUT.rglob("*_outputs.zip")):
    stage = z.name[: -len("_outputs.zip")]
    with zipfile.ZipFile(z) as zf:
        _note(stage, zf.namelist())
        n = sum(_take(stage, m.filename, lambda m=m: zf.read(m))
                for m in zf.infolist() if not m.is_dir())
    print(f"earlier: {{z.name:32s}} {{n}} files")
for marker in sorted(INPUT.rglob("_STAGE_*.txt")):      # an output Kaggle already unpacked
    stage, top = marker.stem[len("_STAGE_"):], marker.parent
    _note(stage, [f.name for f in top.iterdir()])
    n = sum(_take(stage, f.relative_to(top).as_posix(), f.read_bytes)
            for f in top.rglob("*") if f.is_file())
    print(f"earlier: {{stage:32s}} {{n}} files (unpacked)")

# earlier stages not attached as inputs: download them from the GitHub releases.
# Also this notebook's own release, so a stopped run continues where it stopped. It goes
# first: a newer output of an earlier stage must overwrite the copy kept in it.
# (analysis does not resume: everything it computes is cheap and derived from upstream.)
sys.path.insert(0, str(WORK / "scripts"))
import github_sync as gh
have = {{z.name[: -len("_outputs.zip")] for z in INPUT.rglob("*_outputs.zip")}}
have |= {{m.stem[len("_STAGE_"):] for m in INPUT.rglob("_STAGE_*.txt")}}
own = [NAME] if globals().get("RESUME_OWN", True) else []
for stage in [s for s in own + NEEDS if s not in have]:
    z = gh.download_asset(GITHUB_REPO, TOKEN, f"outputs-{{stage}}", f"{{stage}}_outputs.zip", BASE / "inputs")
    if z is None:
        if stage != NAME:
            print(f"earlier: {{stage}} NOT FOUND - run that notebook first (or attach its output)")
            _unfinished.add(stage)
        continue
    with zipfile.ZipFile(z) as zf:
        _note(stage, zf.namelist())
        n = sum(_take(stage, m.filename, lambda m=m: zf.read(m))
                for m in zf.infolist() if not m.is_dir())
    print(f"earlier: {{stage:32s}} {{n}} files (GitHub release)"
          + ("  [made before the data repair: runs on the random-split file are redone]" if stage in _pre_repair else ""))

# An earlier notebook that is still running (or stopped) has no _COMPLETE.txt yet.
# Starting now would build on its partial output, so stop here instead.
waiting = sorted(set(NEEDS) & _unfinished)
if waiting:
    raise RuntimeError(f"not finished yet (or not found): {{waiting}}. Run this notebook after "
                       + ", ".join(waiting) + " says 'All steps finished'.")

# The gene presence/absence matrix in the dataset marks the minority state of each locus,
# not presence (KNOWN_ISSUES, issue 12). Notebooks that read it use the copy rebuilt by pa_fix.
PA_STAGES = ("trilens_random", "trilens_splits", "standard_random", "standard_splits")
PA_STATE = "original"
if "pa_fix" in NEEDS:
    pa_fixed = RES / "pa_fix" / "arcadia_v2_gene_presence_absence.pkl"
    if not pa_fixed.exists():
        raise RuntimeError("pa_fix finished without the rebuilt matrix: read its pa_repair.json")
    pap = proc / pa_fixed.name
    if pap.exists() or pap.is_symlink():
        pap.unlink()
    pap.symlink_to(pa_fixed)
    PA_STATE = "fixed"
    print("data   : gene presence/absence -> rebuilt copy from pa_fix")
old_pa = sorted(set(NEEDS) & set(PA_STAGES) & _pa_old)
if old_pa:
    raise RuntimeError(f"{{old_pa}} ran on the old presence/absence matrix: run them again first.")

os.chdir(WORK)
os.environ.update(TRILENS_ROOT=str(WORK), TRILENS_MAIN_RUN="trilens_attnfix", TRILENS_LENSB=LENSB,
                  WANDB_MODE="disabled", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], check=True)
import torch
ENV_TEXT = "\n".join([
    f"python {{sys.version.split()[0]}}",
    f"torch {{torch.__version__}} cuda {{torch.version.cuda}}",
    f"gpu {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}}",
    "", subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout])
print("\nGPU    :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("Lens B :", LENSB)
if NEEDS_GPU and not torch.cuda.is_available():
    raise RuntimeError("No GPU. Settings -> Accelerator -> GPU T4 x2, then Save & Run All again.")
'''

HELPERS = r'''
# 2. Helpers: run a step (skipped if its output exists), save the zip after every step
FIG_DIR = BASE / "thesis" / "images"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def _keep(f):
    parts = set(f.parts)
    return not (parts & {"cache", "ckpts", "__pycache__"} or f.suffix == ".ckpt")

def package(complete=False):
    tmp = OUT_ZIP.with_suffix(".tmp")
    dirs = ([d for d in RES.iterdir() if d.is_dir() and d.name != "pa_fix"] if WRITES == ["*"]
            else [RES / d for d in WRITES])
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"_STAGE_{NAME}.txt", NAME)
        z.writestr("_CODE_VERSION.txt", f"{NAME}\n{GITHUB_REPO}\n{CODE_VERSION}\n")
        if DATA_STATE in ("repaired", "clean"):
            z.writestr("_DATA_REPAIRED.txt", DATA_STATE)
        if PA_STATE == "fixed" or NAME == "pa_fix":
            z.writestr("_PA_FIXED.txt", "presence/absence matrix rebuilt by pa_fix")
        z.writestr("_ENV.txt", ENV_TEXT)            # exact package versions and GPU of this run
        if complete:                                # the notebook reached its last cell
            z.writestr("_COMPLETE.txt", "failed steps:\n" + "\n".join(FAILED))
        for d in dirs:
            for f in d.rglob("*") if d.exists() else []:
                if f.is_file() and _keep(f.relative_to(RES)):
                    z.write(f, PREFIX + f.relative_to(RES).as_posix())
        for f in sorted(FIG_DIR.glob("*.pdf")):
            z.write(f, "figures/" + f.name)
    tmp.replace(OUT_ZIP)
    print(f"[saved] {OUT_ZIP.name}  {OUT_ZIP.stat().st_size / 1e6:.1f} MB")

_last_sync = [0.0]

def sync(force=False):
    """Send the zip to the release and the text results to their branch (at most every 10 min)."""
    if not TOKEN or (not force and time.time() - _last_sync[0] < 600):
        return
    gh.upload_asset(GITHUB_REPO, TOKEN, f"outputs-{NAME}", OUT_ZIP)
    gh.push_text_results(gh.remote(GITHUB_REPO), f"results/{NAME}", RES,
                         message=f"{NAME} outputs, code {CODE_VERSION}", token=TOKEN)
    _last_sync[0] = time.time()

FAILED = []

def run(cmd, marker=None, cwd=None, env=None):
    root = Path(cwd) if cwd else WORK
    if marker and (root / "results" / marker).exists():
        print(f"[skip] {cmd}\n       (results/{marker} already exists)")
        return
    print(f"\n$ {cmd}\n", flush=True)
    t = time.time()
    p = subprocess.Popen(cmd, shell=True, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env={**os.environ, **(env or {})})
    last_save = time.time()
    for line in p.stdout:
        print(line, end="")
        if time.time() - last_save > 1200:        # a long step: save what it has so far
            package()                              # (finished folds, cached runs) every 20 min,
            sync()                                 # so a session that ends at 12 h loses little
            last_save = time.time()
    p.wait()
    print(f"\n[{'ok' if p.returncode == 0 else 'FAILED'}] {cmd}  ({(time.time() - t) / 60:.1f} min)")
    package()
    sync()
    if p.returncode != 0:
        FAILED.append(cmd)      # keep going: later steps that do not need this one still run
'''

STD_ROOT = r'''
# Standard Lens B: the same diagnostics, in a second project folder whose results/
# points at results/standard/ (written there) and at the shared classical runs (read only).
STD = Path("/tmp/trilens_std/code")
std_res = RES / "standard"
if (std_res / "trilens_attnfix").exists():
    if not STD.exists():
        shutil.copytree(WORK, STD, ignore=lambda d, n: [x for x in n if Path(d) == WORK and x in ("results", "data", "experiments")])
        (STD / "data").symlink_to(WORK / "data")
        (STD / "results").mkdir()
        own = ["trilens_attnfix", "pillar_trilens_t7_clean_gene_set", "pillar_trilens_mlst",
               "pillar_t7_clean_final_comparison", "pillar_mlst_final_comparison", "diagnostics"]
        shared = ["stage3_cascadestack_v9", "pillar_cascadestack_classical_phylogeny_clean",
                  "pillar_cascadestack_classical_mlst", "pillar3_shap"]
        for d in own:
            (std_res / d).mkdir(parents=True, exist_ok=True)
            (STD / "results" / d).symlink_to(std_res / d)
        for d in shared:
            (STD / "results" / d).symlink_to(RES / d)
    env = dict(TRILENS_ROOT=str(STD), TRILENS_LENSB="standard")
    for title, cmd, marker in [
        ("phylogroup split", "python -m src.experiments.diagnose_t7_clean_final_comparison",
         "pillar_t7_clean_final_comparison/three_way_comparison_FINAL.csv"),
        ("lineage holdout", "python -m src.experiments.diagnose_mlst_final_comparison",
         "pillar_mlst_final_comparison/three_way_comparison_FINAL.csv"),
        ("clinical", "python scripts/diagnostics/holdout_clinical_metrics.py", "diagnostics/holdout_clinical_metrics.csv"),
        ("lineage markers", "python scripts/diagnostics/lineage_marker_test.py", "diagnostics/lineage_marker_test.csv"),
        ("prefix base rate", "python scripts/diagnostics/prefix_baserate_lens_b.py", "diagnostics/prefix_baserate_lens_b.csv"),
        ("attention", "python scripts/diagnostics/attention_diagnosis.py", "diagnostics/attention_diagnosis.csv"),
    ]:
        print(f"\n--- standard Lens B: {title}")
        run(cmd, marker, cwd=STD, env=env)
else:
    print("results/standard/ not found - attach the standard_random and standard_splits outputs")

run("python scripts/diagnostics/strain_ablation_table.py")
'''

FIGS = r'''
# Figures drawn from the new arrays (the table-based figures are redrawn after the
# paper numbers are updated). They land in figures/ inside the output zip.
run("python scripts/figures/make_results_figures.py")
run('python -c "import sys; sys.path.insert(0, \'scripts/figures\'); import make_all_figures as m; m.fig_pr_curves()"')
run("python scripts/diagnostics/verify_thesis_numbers.py")
'''

CHECK = r'''
# {n}. Check: every paper number this run has produced so far, old against new
run("python scripts/diagnostics/collect_paper_numbers.py")
import pandas as pd
t = pd.read_csv(RES / "diagnostics" / "paper_numbers.csv")
print(t[t.status != "MISSING"].drop(columns="source").to_string(index=False))
'''

PA_CHECK = r'''
# {n}. Check: what the rebuild changed
import json
r = json.loads((RES / "pa_fix" / "pa_repair.json").read_text())
for k in ("status", "present_letter", "allele_order_counts", "loci_flipped", "loci_unchanged", "loci_inconsistent",
          "missing_cells", "core_median_prevalence", "spearman_prevalence_vs_ecor", "per_strain_loci",
          "max_prevalence_old", "max_prevalence_new", "named_core_genes"):
    print(f"{{k:30s}} {{r.get(k)}}")
'''

FINAL = r'''
# {n}. Save (and send to GitHub)
package(complete=True)
sync(force=True)
print("Download {name}_outputs.zip from the Output tab (right side, or Data -> Output).")
if FAILED:
    print("\nThese steps FAILED (scroll up to their output):")
    for c in FAILED:
        print("  ", c)
    raise RuntimeError(f"{{len(FAILED)}} step(s) failed")
print("\nAll steps finished.")
'''


def notebook(name, st):
    needs = st["needs"]
    intro = [f"# {st['title']}", "", st["about"], "", "**Before you start**", "",
             f"1. Settings → Accelerator → **{st['gpu']}**. Settings → Internet → **On**.",
             "2. Add Input → your dataset **trilens-data**. (The code is cloned from GitHub.)",
             "3. Add-ons → Secrets → **GITHUB_TOKEN** → switch it on for this notebook.",
             "4. **Save Version → Save & Run All (Commit)**. You can close the tab.",
             ""]
    if needs:
        intro.append("The outputs of " + ", ".join(f"**{n}**" for n in needs) +
                     " are downloaded from their GitHub releases. Run those notebooks first.")
        intro.append("")
    intro += [f"Output: `{name}_outputs.zip`. After every step it is sent to the GitHub release "
              f"`outputs-{name}`, and its text results to the branch `results/{name}`. If the "
              "session stops (12-hour limit), just run it again: it downloads its own release "
              "and skips the finished steps.",
              ]
    if st["lensb"] == "standard":
        intro += ["", "This notebook trains the **ablation** model. Do not attach the "
                  "trilens_random or trilens_splits outputs here."]
    cells = [md("\n".join(intro)),
             code(SETUP.format(name=name, lensb=st["lensb"], writes=json.dumps(st["writes"]),
                         needs=json.dumps(st["needs"]), resume_own=str(name != "analysis"),
                         needs_gpu=str(name not in ("analysis", "pa_fix")))),
             code(HELPERS)]
    n = 3
    for title, cmd, marker in st["steps"]:
        cells.append(md(f"## {n}. {title}"))
        cells.append(code(f'run("{cmd}",\n    "{marker}")' if marker else f'run("{cmd}")'))
        n += 1
    if name == "analysis":
        cells.append(md(f"## {n}. Strain-conditioning ablation (tab:attnfix)"))
        cells.append(code(STD_ROOT)); n += 1
        cells.append(md(f"## {n}. Figures and the thesis-number check"))
        cells.append(code(FIGS)); n += 1
    check = CHECK.format(n=n) if name != "pa_fix" else PA_CHECK.format(n=n)
    if st["lensb"] == "standard":
        check += '\nprint("\\n(these are STANDARD Lens B values; analysis.ipynb puts them side by side)")\n'
    cells.append(md(f"## {n}. Check"))
    cells.append(code(check))
    n += 1
    cells.append(code(FINAL.format(n=n, name=name)))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


OUT.mkdir(parents=True, exist_ok=True)
for name, st in STAGES.items():
    nb = notebook(name, st)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), name, "exec")
    (OUT / f"{name}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", name)
