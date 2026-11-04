import json
import sys
from pathlib import Path

STEP = "python -m pipeline."
SPLITS_ALL = "st_grouped_5fold,random_5fold,country_uk_to_norway,country_norway_to_uk,time_norway_2013"
FLOOR_SPLITS = "random,mlst,country_uk_to_norway,country_norway_to_uk,time_norway_2013"
GPU = "GPU T4 x2"
CPU = "None (CPU is enough)"


def step(title, command, marker=None):
    return (title, STEP + command, marker)


DETERMINANTS = step(
    "Strain by determinant matrix (AMRFinderPlus 4.2.7, database 2026-03-24.1)",
    "step22_build_determinant_matrix build",
    "determinant_matrix/coverage.json",
)

STAGES = {
    "stage01_gene_presence_matrix": dict(
        title="Gene presence/absence matrix with the right orientation",
        about="Downloads the PLINK presence/absence files of the Arcadia release and builds the matrix from "
        "the allele letters, so that 1 means present for every locus. Every later notebook that reads "
        "the matrix takes this copy.",
        gpu=CPU,
        needs=[],
        writes=["gene_presence_rebuild"],
        steps=[
            step(
                "Find the PLINK files",
                "step04_build_gene_presence_matrix --part download --dest /tmp/trilens/pa_source",
                "gene_presence_rebuild/arcadia_v2_gene_presence_absence.pkl",
            ),
            step(
                "Build the matrix (refuses if more than the orientation changes)",
                "step04_build_gene_presence_matrix --part build --source /tmp/trilens/pa_source/paths.json "
                "--old data/processed/arcadia_v2_gene_presence_absence.pkl "
                "--out results/gene_presence_rebuild/arcadia_v2_gene_presence_absence.pkl "
                "--report results/gene_presence_rebuild/gene_presence_rebuild_report.json",
                "gene_presence_rebuild/arcadia_v2_gene_presence_absence.pkl",
            ),
        ],
    ),
    "stage02_classical_models": dict(
        title="Classical stack, baselines, Giessen transfer and gene importance",
        about="The five classical learners on the three splits, the external baselines, the retrained stack on "
        "the Giessen collection, and gain and SHAP importance.",
        gpu=GPU + " (XGBoost runs on the GPU)",
        needs=[],
        writes=[
            "classical_stack_random",
            "classical_stack_phylogroup",
            "classical_stack_lineage",
            "pca_baseline",
            "phylogroup_only_baseline",
            "tabnet_random",
            "tabnet_phylogroup",
            "tabnet_lineage",
            "giessen_transfer",
            "gene_importance_gain",
            "gene_importance_shap",
            "base_learner_shap",
        ],
        steps=[
            step(
                "Classical stack, random split",
                "step09_train_classical_stack --split random",
                "classical_stack_random/predictions_saved.npz",
            ),
            step(
                "Classical stack, phylogroup split",
                "step09_train_classical_stack --split phylogroup",
                "classical_stack_phylogroup/classical_stack_results.csv",
            ),
            step(
                "Classical stack, lineage holdout",
                "step09_train_classical_stack --split lineage",
                "classical_stack_lineage/classical_stack_results.csv",
            ),
            step(
                "Baseline: XGBoost on 1,024 PCA components",
                "step10_train_baselines --model pca",
                "pca_baseline/pca_baseline_results.csv",
            ),
            step(
                "Baseline: phylogroup label only",
                "step10_train_baselines --model phylogroup_only",
                "phylogroup_only_baseline/phylogroup_only_results.csv",
            ),
            step(
                "Baseline: TabNet, random split",
                "step10_train_baselines --model tabnet_random",
                "tabnet_random/tabnet_random_results.csv",
            ),
            step(
                "Baseline: TabNet, phylogroup split",
                "step10_train_baselines --model tabnet_phylogroup",
                "tabnet_phylogroup/tabnet_phylogroup_results.csv",
            ),
            step(
                "Baseline: TabNet, lineage holdout",
                "step10_train_baselines --model tabnet_lineage",
                "tabnet_lineage/tabnet_lineage_results.csv",
            ),
            step("Giessen collection", "step18_transfer_to_giessen", "giessen_transfer/per_drug_metrics.csv"),
            step("Gain importance", "step19_explain_models --part gain", "gene_importance_gain/multi_drug_markers.csv"),
            step(
                "SHAP importance",
                "step19_explain_models --part shap",
                "gene_importance_shap/multi_drug_markers_shap.csv",
            ),
            step(
                "SHAP on the five stack learners",
                "step19_explain_models --part base_learner_shap",
                "base_learner_shap/concordance_metrics.csv",
            ),
        ],
    ),
    "stage03_trilens_random": dict(
        title="TriLens stack on the random split, and its component ablation",
        about="The three neural branches with the gate, stacked with the classical learners, then the model "
        "with one part removed at a time.",
        gpu=GPU,
        needs=["stage01_gene_presence_matrix", "stage02_classical_models"],
        writes=["trilens_stack_random", "component_ablation"],
        steps=[
            step(
                "TriLens stack, random split",
                "step11_train_trilens_stack --split random",
                "trilens_stack_random/predictions_saved.npz",
            ),
            step(
                "Branches averaged, no gate",
                "step12_ablate_trilens_components --part branches_averaged",
                "component_ablation/branches_averaged_predictions.npz",
            ),
            step(
                "Ablation table",
                "step12_ablate_trilens_components --part table",
                "component_ablation/component_ablation_results.csv",
            ),
        ],
    ),
    "stage04_trilens_other_splits": dict(
        title="TriLens neural branches on the phylogroup split and the lineage holdout",
        about="The neural branches retrained on the two other splits, with the gene sets fitted on each split.",
        gpu=GPU,
        needs=["stage01_gene_presence_matrix", "stage02_classical_models"],
        writes=["trilens_neural_phylogroup", "trilens_neural_lineage"],
        steps=[
            step(
                "Phylogroup split",
                "step11_train_trilens_stack --split phylogroup",
                "trilens_neural_phylogroup/trilens_neural_phylogroup_results.csv",
            ),
            step(
                "Lineage holdout",
                "step11_train_trilens_stack --split lineage",
                "trilens_neural_lineage/trilens_neural_lineage_results.csv",
            ),
        ],
    ),
    "stage05_standard_attention_random": dict(
        title="Ablation model: standard gated attention, random split",
        about="The same TriLens stack with the strain conditioning of the gene attention branch removed.",
        gpu=GPU,
        needs=["stage01_gene_presence_matrix", "stage02_classical_models"],
        attention="standard",
        writes=["trilens_stack_random"],
        steps=[
            step(
                "Standard attention, random split",
                "step11_train_trilens_stack --split random --attention standard",
                "trilens_stack_random/predictions_saved.npz",
            )
        ],
    ),
    "stage06_standard_attention_other_splits": dict(
        title="Ablation model: standard gated attention, phylogroup split and lineage holdout",
        about="The standard-attention model on the two other splits.",
        gpu=GPU,
        needs=["stage01_gene_presence_matrix", "stage02_classical_models"],
        attention="standard",
        writes=["trilens_neural_phylogroup", "trilens_neural_lineage"],
        steps=[
            step(
                "Phylogroup split",
                "step11_train_trilens_stack --split phylogroup --attention standard",
                "trilens_neural_phylogroup/trilens_neural_phylogroup_results.csv",
            ),
            step(
                "Lineage holdout",
                "step11_train_trilens_stack --split lineage --attention standard",
                "trilens_neural_lineage/trilens_neural_lineage_results.csv",
            ),
        ],
    ),
    "stage07_position_shuffle": dict(
        title="Position shuffle test",
        about="A plain convolutional network on ciprofloxacin, with the variant columns in order and shuffled, "
        "on both collections.",
        gpu=GPU,
        needs=[],
        writes=["position_shuffle_test"],
        steps=[
            *[
                (
                    f"{d}, {v}",
                    f"SHUFFLE_ONLY={d}_CIP_{v} {STEP}step17_test_position_shuffle --part run",
                    f"position_shuffle_test/runs/{d}_CIP_{v}_run.json",
                )
                for d in ("giessen", "arcadia")
                for v in ("orig", "shuf")
            ],
            step(
                "All four runs together",
                "step17_test_position_shuffle --part run",
                "position_shuffle_test/runs/shuffle_results.csv",
            ),
            step("Summary", "step17_test_position_shuffle --part summary", "position_shuffle_test/verdict.json"),
        ],
    ),
    "stage08_analysis": dict(
        title="Analysis of the three splits",
        about="Stacks the neural and classical arms on the other splits, corrects the holdout for prevalence, "
        "measures clinical errors, audits the splits, explains the models and builds the strain "
        "conditioning table. Nothing here trains a large model.",
        gpu=CPU,
        cpu=True,
        needs=[
            "stage01_gene_presence_matrix",
            "stage02_classical_models",
            "stage03_trilens_random",
            "stage04_trilens_other_splits",
            "stage05_standard_attention_random",
            "stage06_standard_attention_other_splits",
            "stage07_position_shuffle",
        ],
        writes=["*"],
        resume=False,
        steps=[
            step(
                "Phylogroup split: stack neural and classical",
                "step13_stack_neural_and_classical --split phylogroup",
                "trilens_stack_phylogroup/split_comparison.csv",
            ),
            step(
                "Lineage holdout: stack neural and classical",
                "step13_stack_neural_and_classical --split lineage",
                "trilens_stack_lineage/split_comparison.csv",
            ),
            step(
                "Prevalence-matched holdout",
                "step14_correct_holdout_for_prevalence --part matching",
                "diagnostics/matched_prevalence.csv",
            ),
            step(
                "Clinical errors under the holdout",
                "step15_measure_clinical_errors --part holdout",
                "diagnostics/holdout_clinical_errors.csv",
            ),
            step(
                "Threshold tuning",
                "step15_measure_clinical_errors --part thresholds",
                "diagnostics/threshold_tuning.csv",
            ),
            step(
                "Near duplicates, random split",
                "step16_audit_splits_and_duplicates --part duplicates_random",
                "near_duplicate_audit/near_duplicate_results.csv",
            ),
            step(
                "Near duplicates, lineage holdout",
                "step16_audit_splits_and_duplicates --part duplicates_lineage",
                "near_duplicate_audit/near_duplicate_results_lineage.csv",
            ),
            step(
                "Full metric table",
                "step15_measure_clinical_errors --part metrics",
                "full_metric_suite/full_metric_suite.csv",
            ),
            step(
                "Prevalence-free metrics",
                "step14_correct_holdout_for_prevalence --part invariant_metrics",
                "prevalence_invariant_metrics/prevalence_invariant_metrics.csv",
            ),
            step(
                "Marker gene annotation",
                "step19_explain_models --part annotation",
                "marker_gene_annotation/annotated_markers.csv",
            ),
            step(
                "Lineage marker test",
                "step19_explain_models --part lineage_markers",
                "diagnostics/lineage_marker_test.csv",
            ),
            step(
                "Source assembly enrichment",
                "step19_explain_models --part provenance",
                "diagnostics/assembly_provenance_enrichment.csv",
            ),
            step(
                "Attention diagnosis", "step19_explain_models --part attention", "diagnostics/attention_diagnosis.csv"
            ),
            step(
                "Position shuffle summary",
                "step17_test_position_shuffle --part summary",
                "position_shuffle_test/verdict.json",
            ),
            step(
                "Holdout margin of the stack over the classical arm",
                "step20_test_seed_stability --part margin",
                "diagnostics/holdout_margin.csv",
            ),
            step("Split integrity checks", "step16_audit_splits_and_duplicates --part integrity"),
        ],
    ),
    "stage09_seed_stability": dict(
        title="Seed stability of the holdout margin",
        about="Retrains the neural branch on the lineage holdout with two more seeds and recomputes the margin "
        "of the full stack over the classical arm.",
        gpu=GPU,
        needs=["stage01_gene_presence_matrix", "stage02_classical_models", "stage04_trilens_other_splits"],
        writes=["seed_stability"],
        steps=[
            step(
                "Seed 123",
                "step20_test_seed_stability --part seeds --seeds 123 --train-only",
                "seed_stability/seed_123/trilens_neural_lineage/predictions_saved.npz",
            ),
            step(
                "Seed 456",
                "step20_test_seed_stability --part seeds --seeds 456 --train-only",
                "seed_stability/seed_456/trilens_neural_lineage/predictions_saved.npz",
            ),
            step(
                "Stack and read every seed",
                "step20_test_seed_stability --part seeds --seeds 123,456",
                "seed_stability/verdict.json",
            ),
        ],
    ),
    "stage10_metadata_floor": dict(
        title="Metadata and assembly floor",
        about="Models on metadata and assembly statistics alone, on every split.",
        gpu=CPU,
        cpu=True,
        needs=[],
        writes=["diagnostics"],
        steps=[
            step("Split files", "step08_build_country_time_and_fold_splits --part load"),
            step(
                "Metadata-only models",
                f"step21_measure_metadata_floor --splits {FLOOR_SPLITS}",
                "diagnostics/metadata_floor.csv",
            ),
        ],
    ),
    "stage11_determinant_rescue": dict(
        title="Determinant calls on unseen lineages",
        about="Adds each strain's AMRFinderPlus calls to the pan-genome catalogue, on every split, with two "
        "sensitivity analyses.",
        gpu=CPU,
        cpu=True,
        needs=["stage01_gene_presence_matrix"],
        writes=["determinant_matrix", "determinant_matrix_all_databases", "determinant_rescue"],
        steps=[
            DETERMINANTS,
            step(
                "Primary: intermediate counted as resistant",
                f"step23_rescue_with_determinant_calls --drugs paper --labels ri --splits {SPLITS_ALL}",
                "determinant_rescue/paper_ri/verdict.json",
            ),
            step(
                "Sensitivity: intermediate excluded",
                "step23_rescue_with_determinant_calls --drugs paper --labels no_intermediate",
                "determinant_rescue/paper_no_intermediate/verdict.json",
            ),
            step(
                "Sensitivity: every database version",
                "step22_build_determinant_matrix build --db all --out results/determinant_matrix_all_databases",
                "determinant_matrix_all_databases/coverage.json",
            ),
            step(
                "Sensitivity: the test on that matrix",
                "step23_rescue_with_determinant_calls --drugs paper --labels ri "
                "--determinant-matrix results/determinant_matrix_all_databases/acquired.csv "
                "--out results/determinant_rescue/paper_ri_all_databases",
                "determinant_rescue/paper_ri_all_databases/verdict.json",
            ),
        ],
    ),
    "stage12_assembly_size": dict(
        title="Assembly size within lineages",
        about="Assembly statistics against acquired-gene count and resistance inside sequence types.",
        gpu=CPU,
        cpu=True,
        needs=["stage01_gene_presence_matrix"],
        writes=["determinant_matrix", "diagnostics"],
        steps=[
            DETERMINANTS,
            step(
                "Within-lineage assembly signal",
                "step24_test_assembly_size_proxy --part within_lineage",
                "diagnostics/assembly_size_within_lineage.json",
            ),
        ],
    ),
    "stage13_external_cohort_screen": dict(
        title="External cohort screen",
        about="Finds the NCBI Pathogen Detection isolates with submitted susceptibility results. Internet must be on.",
        gpu=CPU,
        cpu=True,
        needs=[],
        writes=["external_cohort_screen"],
        steps=[
            step(
                "Count external candidates",
                "step25_screen_external_cohort --out results/external_cohort_screen",
                "external_cohort_screen/columns.json",
            )
        ],
    ),
    "stage14_external_cohort": dict(
        title="Frozen models on the external cohort",
        about="Trains once on the training strains and applies the models unchanged to the external isolates.",
        gpu=CPU,
        cpu=True,
        needs=["stage13_external_cohort_screen"],
        writes=["determinant_matrix", "external_cohort"],
        steps=[
            DETERMINANTS,
            step(
                "Primary",
                "step26_validate_on_external_cohort --part transfer --db primary --labels ri",
                "external_cohort/primary_ri/verdict.json",
            ),
            step(
                "Every database version",
                "step26_validate_on_external_cohort --part transfer --db all --labels ri",
                "external_cohort/all_ri/verdict.json",
            ),
            step(
                "Intermediate excluded",
                "step26_validate_on_external_cohort --part transfer --db primary --labels no_intermediate",
                "external_cohort/primary_no_intermediate/verdict.json",
            ),
        ],
    ),
    "stage15_coverage_and_rule": dict(
        title="Coverage check and model against rule",
        about="Whether the rescue holds in the strains without calls, and a learned model against the rule.",
        gpu=CPU,
        cpu=True,
        needs=["stage01_gene_presence_matrix", "stage11_determinant_rescue"],
        writes=["determinant_matrix", "coverage_and_rule"],
        steps=[
            DETERMINANTS,
            step(
                "Coverage strata and the rule",
                "step27_compare_model_with_rule --rescue-dir results/determinant_rescue/paper_ri "
                "--out results/coverage_and_rule",
                "coverage_and_rule/verdict.json",
            ),
        ],
    ),
    "stage16_external_robustness": dict(
        title="External robustness and the assembly mechanism",
        about="Leave one BioProject out on the external cohort, and whether the assembly signal is batch or biology.",
        gpu=CPU,
        cpu=True,
        needs=["stage13_external_cohort_screen", "stage14_external_cohort"],
        writes=["determinant_matrix", "external_robustness", "assembly_size_mechanism"],
        steps=[
            DETERMINANTS,
            step(
                "Leave one BioProject out",
                "step26_validate_on_external_cohort --part robustness "
                "--external-dir results/external_cohort/primary_ri "
                "--out results/external_robustness",
                "external_robustness/verdict.json",
            ),
            step(
                "Batch or biology",
                "step24_test_assembly_size_proxy --part mechanism --out results/assembly_size_mechanism",
                "assembly_size_mechanism/verdict.json",
            ),
        ],
    ),
    "stage17_species_screen": dict(
        title="Second species screen",
        about="Applies the pre-registered gate to every organism in NCBI Pathogen Detection. Internet must be on.",
        gpu=CPU,
        cpu=True,
        needs=[],
        writes=["species_screen"],
        steps=[
            step(
                "Every organism",
                "step29_test_second_species --part screen --out results/species_screen",
                "species_screen/cluster_source.json",
            )
        ],
    ),
}


SETUP = r"""
GITHUB_REPO = "FaiazPronoy/TriLens-AMR"
CODE_REF = "{code_ref}"
NAME = "{name}"
ATTENTION = "{attention}"
WRITES = {writes}
NEEDS = {needs}
NEEDS_GPU = {needs_gpu}
RESUME_OWN = {resume_own}

import os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
BASE = Path("/tmp/trilens")
WORK = BASE / "work"
RES = WORK / "results"
OUT_ZIP = Path(f"/kaggle/working/{{NAME}}_outputs.zip")
PREFIX = "standard_attention/" if ATTENTION == "standard" else ""
STANDARD = ("stage05_standard_attention_random", "stage06_standard_attention_other_splits")
CONDITIONED = ("stage03_trilens_random", "stage04_trilens_other_splits")


def find_one(name):
    hits = sorted(INPUT.rglob(name))
    return hits[0] if hits else None


def secret(key):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key)
    except Exception:
        return os.environ.get(key)


TOKEN = secret("GITHUB_TOKEN")
print("github :", GITHUB_REPO, "(token found)" if TOKEN else "(no token: nothing is uploaded)")

repo = BASE / "repo"
if not repo.exists():
    auth = []
    if TOKEN:
        import base64
        auth = ["-c", "http.extraheader=AUTHORIZATION: basic "
                + base64.b64encode(f"x-access-token:{{TOKEN}}".encode()).decode()]
    r = subprocess.run(["git", *auth, "clone", "--quiet", f"https://github.com/{{GITHUB_REPO}}.git", str(repo)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git clone failed: " + (r.stderr or "").replace(TOKEN or "\0", "***"))
    subprocess.run(["git", "checkout", "--quiet", CODE_REF], cwd=repo, check=True)
CODE_VERSION = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
print("version:", CODE_VERSION)
if not WORK.exists():
    shutil.copytree(repo, WORK, ignore=lambda d, n: [x for x in n if (Path(d) == repo and x in
                    ("results", "data", "paper", "thesis", ".git")) or x == "__pycache__"])

master = find_one("arcadia_v2_master_features.pkl")
if master is None:
    raise FileNotFoundError("add the dataset trilens-data (Add Input)")
proc, giessen = WORK / "data" / "processed", WORK / "data" / "Giessen_dataset"
proc.mkdir(parents=True, exist_ok=True)
giessen.mkdir(parents=True, exist_ok=True)
for f in sorted(master.parent.iterdir()):
    if f.is_file():
        dst = (giessen if f.name.startswith("cip_ctx_ctz_gen") else proc) / f.name
        if not dst.exists():
            dst.symlink_to(f)
for rel in ("raw/strains_metadata.csv", "raw/pangenome/pangenome_cds.fa", "raw/pangenome/gene_presence_absence.csv"):
    f = find_one(Path(rel).name)
    dst = WORK / "data" / rel
    if f is not None and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(f)

gene_filtered = proc / "arcadia_v2_gene_filtered.pkl"
repaired = BASE / "repaired" / gene_filtered.name
if gene_filtered.is_symlink() and Path(os.readlink(gene_filtered)) != repaired:
    r = subprocess.run([sys.executable, "-m", "pipeline.step06_repair_snp_gene_blocks",
                        "--src", str(gene_filtered.resolve()),
                        "--snp", str((proc / "arcadia_v2_snp_matrix.pkl").resolve()),
                        "--out", str(repaired), "--report", str(RES / "diagnostics" / "data_repair.json")],
                       cwd=WORK, capture_output=True, text=True)
    if r.returncode == 0:
        gene_filtered.unlink()
        gene_filtered.symlink_to(repaired)
        print("data   : gene-filtered SNP blocks repaired")
    elif r.returncode != 2:
        raise RuntimeError("the SNP block repair refused the file:\n" + r.stdout[-2000:] + r.stderr[-2000:])

RES.mkdir(parents=True, exist_ok=True)
unfinished = set()


def target(stage, member):
    if ATTENTION == "standard":
        if stage in CONDITIONED:
            return None
        if member.startswith("standard_attention/"):
            member = member[len("standard_attention/"):]
    return RES / member


def take(stage, rel, read):
    t = target(stage, rel)
    if t is None or rel.split("/")[0].startswith("_") or (stage == NAME and t.exists()):
        return 0
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_bytes(read())
    return 1


def unpack(stage, zf):
    names = zf.namelist()
    if "_COMPLETE.txt" not in names:
        unfinished.add(stage)
    return sum(take(stage, m.filename, lambda m=m: zf.read(m)) for m in zf.infolist() if not m.is_dir())


have = set()
for z in sorted(INPUT.rglob("*_outputs.zip")):
    stage = z.name[: -len("_outputs.zip")]
    have.add(stage)
    with zipfile.ZipFile(z) as zf:
        print(f"earlier: {{stage:42s}} {{unpack(stage, zf)}} files")

sys.path.insert(0, str(WORK))
from trilens.utils import github_sync as gh
for stage in [s for s in ([NAME] if RESUME_OWN else []) + NEEDS if s not in have]:
    z = gh.download_asset(GITHUB_REPO, TOKEN, f"outputs-{{stage}}", f"{{stage}}_outputs.zip", BASE / "inputs")
    if z is None:
        if stage != NAME:
            print(f"earlier: {{stage}} NOT FOUND - run that notebook first or attach its output")
            unfinished.add(stage)
        continue
    with zipfile.ZipFile(z) as zf:
        print(f"earlier: {{stage:42s}} {{unpack(stage, zf)}} files (GitHub release)")

waiting = sorted(set(NEEDS) & unfinished)
if waiting:
    raise RuntimeError(f"not finished yet: {{waiting}}. Run this notebook after they say 'All steps finished'.")

if "stage01_gene_presence_matrix" in NEEDS:
    rebuilt = RES / "gene_presence_rebuild" / "arcadia_v2_gene_presence_absence.pkl"
    if not rebuilt.exists():
        raise RuntimeError("stage01_gene_presence_matrix finished without the rebuilt matrix")
    link = proc / rebuilt.name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(rebuilt)
    print("data   : gene presence/absence -> rebuilt copy")

os.chdir(WORK)
os.environ.update(TRILENS_ROOT=str(WORK), TRILENS_ATTENTION=ATTENTION, WANDB_MODE="disabled",
                  PYTHONUNBUFFERED="1", MPLBACKEND="Agg", TRILENS_REPO_RESULTS=str(repo / "results"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], check=True)
import torch
ENV_TEXT = "\n".join([f"python {{sys.version.split()[0]}}", f"torch {{torch.__version__}} cuda {{torch.version.cuda}}",
                      f"gpu {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}}", "",
                      subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout])
print("GPU    :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("model  :", "standard attention" if ATTENTION == "standard" else "strain-conditioned attention")
if NEEDS_GPU and not torch.cuda.is_available():
    raise RuntimeError("No GPU. Settings -> Accelerator -> GPU T4 x2, then Save & Run All again.")
"""

HELPERS = r"""
FAILED = []
last_sync = [0.0]


def keep(f):
    return not (set(f.parts) & {"cache", "ckpts", "__pycache__"} or f.suffix == ".ckpt")


def package(complete=False):
    tmp = OUT_ZIP.with_suffix(".tmp")
    dirs = [d for d in RES.iterdir() if d.is_dir()] if WRITES == ["*"] else [RES / d for d in WRITES]
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"_STAGE_{NAME}.txt", NAME)
        z.writestr("_CODE_VERSION.txt", f"{NAME}\n{GITHUB_REPO}\n{CODE_VERSION}\n")
        z.writestr("_ENV.txt", ENV_TEXT)
        if complete:
            z.writestr("_COMPLETE.txt", "failed steps:\n" + "\n".join(FAILED))
        for d in dirs:
            for f in (d.rglob("*") if d.exists() else []):
                if f.is_file() and keep(f.relative_to(RES)):
                    z.write(f, PREFIX + f.relative_to(RES).as_posix())
    tmp.replace(OUT_ZIP)
    print(f"[saved] {OUT_ZIP.name}  {OUT_ZIP.stat().st_size / 1e6:.1f} MB")


def sync(force=False):
    if not TOKEN or (not force and time.time() - last_sync[0] < 600):
        return
    gh.upload_asset(GITHUB_REPO, TOKEN, f"outputs-{NAME}", OUT_ZIP)
    last_sync[0] = time.time()


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
        if time.time() - last_save > 1200:
            package()
            sync()
            last_save = time.time()
    p.wait()
    print(f"\n[{'ok' if p.returncode == 0 else 'FAILED'}] {cmd}  ({(time.time() - t) / 60:.1f} min)")
    package()
    sync()
    if p.returncode != 0:
        FAILED.append(cmd)
"""

STANDARD_ATTENTION = r"""
STD = Path("/tmp/trilens_standard/work")
std_res = RES / "standard_attention"
if (std_res / "trilens_stack_random").exists():
    if not STD.exists():
        shutil.copytree(WORK, STD, ignore=lambda d, n: [x for x in n if Path(d) == WORK and x in ("results", "data")])
        (STD / "data").symlink_to(WORK / "data")
        (STD / "results").mkdir()
        own = ["trilens_stack_random", "trilens_neural_phylogroup", "trilens_neural_lineage",
               "trilens_stack_phylogroup", "trilens_stack_lineage", "diagnostics"]
        shared = ["classical_stack_random", "classical_stack_phylogroup", "classical_stack_lineage",
                  "gene_importance_shap"]
        for d in own:
            (std_res / d).mkdir(parents=True, exist_ok=True)
            (STD / "results" / d).symlink_to(std_res / d)
        for d in shared:
            (STD / "results" / d).symlink_to(RES / d)
    env = dict(TRILENS_ROOT=str(STD), TRILENS_ATTENTION="standard")
    for title, cmd, marker in [
        ("phylogroup split", "python -m pipeline.step13_stack_neural_and_classical --split phylogroup",
         "trilens_stack_phylogroup/split_comparison.csv"),
        ("lineage holdout", "python -m pipeline.step13_stack_neural_and_classical --split lineage",
         "trilens_stack_lineage/split_comparison.csv"),
        ("clinical errors", "python -m pipeline.step15_measure_clinical_errors --part holdout",
         "diagnostics/holdout_clinical_errors.csv"),
        ("lineage markers", "python -m pipeline.step19_explain_models --part lineage_markers",
         "diagnostics/lineage_marker_test.csv"),
        ("source assembly enrichment", "python -m pipeline.step19_explain_models --part provenance",
         "diagnostics/assembly_provenance_enrichment.csv"),
        ("attention", "python -m pipeline.step19_explain_models --part attention",
         "diagnostics/attention_diagnosis.csv"),
    ]:
        print(f"\n--- standard attention: {title}")
        run(cmd, marker, cwd=STD, env=env)
else:
    print("results/standard_attention/ not found: attach the outputs of the two standard-attention notebooks")
run("python -m pipeline.step19_explain_models --part strain_conditioning")
"""

FINAL = r"""
package(complete=True)
sync(force=True)
print("Download {name}_outputs.zip from the Output tab.")
if FAILED:
    print("\nThese steps FAILED (scroll up to their output):")
    for c in FAILED:
        print("  ", c)
    raise RuntimeError(f"{{len(FAILED)}} step(s) failed")
print("\nAll steps finished.")
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def notebook(name, st):
    intro = [
        f"# {st['title']}",
        "",
        st["about"],
        "",
        "**Before you start**",
        "",
        f"1. Settings, Accelerator: **{st['gpu']}**. Settings, Internet: **On**.",
        "2. Add Input: the dataset **trilens-data**. The code is cloned from GitHub.",
        "3. Add-ons, Secrets: **GITHUB_TOKEN**, switched on for this notebook.",
        "4. **Save Version, Save & Run All (Commit)**. You can close the tab.",
        "",
    ]
    if st["needs"]:
        intro += [
            "It reads the outputs of "
            + ", ".join(f"**{n}**" for n in st["needs"])
            + " from their GitHub releases, so run those first.",
            "",
        ]
    intro += [
        f"Output: `{name}_outputs.zip`, also sent to the GitHub release `outputs-{name}` after every step. "
        "If the session stops, run it again: finished steps are skipped."
    ]
    cells = [
        md("\n".join(intro)),
        code(
            SETUP.format(
                name=name,
                attention=st.get("attention", "conditioned"),
                code_ref=st.get("code_ref", "main"),
                writes=json.dumps(st["writes"]),
                needs=json.dumps(st["needs"]),
                needs_gpu=str(not st.get("cpu", False)),
                resume_own=str(st.get("resume", True)),
            )
        ),
        code(HELPERS),
    ]
    for title, cmd, marker in st["steps"]:
        cells.append(md(f"## {title}"))
        cells.append(code(f'run("{cmd}",\n    "{marker}")' if marker else f'run("{cmd}")'))
    if name == "stage08_analysis":
        cells.append(md("## Strain conditioning table"))
        cells.append(code(STANDARD_ATTENTION))
    cells.append(md("## Save"))
    cells.append(code(FINAL.format(name=name)))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "notebooks"
    out.mkdir(parents=True, exist_ok=True)
    for name, st in STAGES.items():
        nb = notebook(name, st)
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                compile("".join(c["source"]), name, "exec")
        (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
