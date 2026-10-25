"""One-off generator for the three thin notebooks (Part D). Not part of the
pipeline itself -- run once to (re)materialize notebooks/*.ipynb from the
cell contents below, so the notebooks stay trivially regeneratable/diffable
rather than hand-edited as opaque JSON.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


NOTEBOOK_01 = nb([
    md("# 01 — Build Arcadia v2 Dataset\n\n"
       "Phase A (VCF -> SNP matrix) + Phase B (labels, phylogroups, splits, "
       "PCA baseline). Thin by design: every cell below calls into "
       "`src/amr_data`; no business logic lives in this notebook. See "
       "`docs/PIPELINE.md` for the full stage diagram."),
    code("import sys\nsys.path.insert(0, '..')\n\n"
         "from src.amr_data.config import load_paths, load_params\n"
         "from src.amr_data.io import load_pickle, save_pickle\n\n"
         "paths = load_paths()\n"
         "params = load_params()\n"
         "print(f'Environment: {paths.env}')"),
    md("## Phase A — VCF to SNP matrix\n\n"
       "Requires the raw VCF (Zenodo 12692732, see `docs/DATA_PROVENANCE.md`). "
       "Skip this cell if `arcadia_v2_snp_matrix.pkl` already exists at "
       "`paths.processed('snp_matrix_pkl')`."),
    code("from src.amr_data.vcf_to_snp import stream_vcf_to_snp_matrix, VcfToSnpConfig\n\n"
         "snp_pkl_path = paths.processed('snp_matrix_pkl')\n"
         "if snp_pkl_path.exists():\n"
         "    print(f'Already built: {snp_pkl_path}')\n"
         "else:\n"
         "    result = stream_vcf_to_snp_matrix(\n"
         "        vcf_path=paths.raw('vcf_filtered'),\n"
         "        sra_map_path=paths.raw('sra_to_genome_map'),\n"
         "        pheno_path=paths.raw('phenotype_matrix'),\n"
         "        cfg=VcfToSnpConfig(**params['vcf_to_snp']),\n"
         "    )\n"
         "    save_pickle(vars(result), snp_pkl_path)\n"
         "    print(f'Saved: {snp_pkl_path}')"),
    md("## Phase B — labels, phylogroups, splits\n\n"
       "Requires `strains_metadata.csv` and the phenotype matrix (same "
       "Zenodo record as the VCF)."),
    code("import pandas as pd\n"
         "from src.amr_data.labels import build_label_matrix\n"
         "from src.amr_data.phylogroup import derive_phylogroup, build_cluster_labels\n"
         "from src.amr_data.splits import (\n"
         "    random_split, phylogroup_stratified_split, lineage_holdout_split,\n"
         "    groups_from_phylogroup, save_split_diagnostics,\n"
         ")\n\n"
         "snp_data = load_pickle(snp_pkl_path)\n"
         "pheno = pd.read_csv(paths.raw('phenotype_matrix'))\n"
         "metadata = pd.read_csv(paths.raw('strains_metadata'))\n\n"
         "samples_df = pd.DataFrame({'Genome.Name': snp_data['selected_genome_names']})\n"
         "samples_df = samples_df.merge(pheno, on='Genome.Name', how='left')\n"
         "samples_df = samples_df.merge(metadata[['Genome.Name', 'MLST', 'Other.Typing']],\n"
         "                               on='Genome.Name', how='left')\n\n"
         "Y, label_mask = build_label_matrix(samples_df)\n"
         "phylogroup_final = derive_phylogroup(samples_df['Other.Typing'], samples_df['MLST'])\n"
         "cluster_labels, phylogroup_names, pg_to_int = build_cluster_labels(phylogroup_final)\n\n"
         "amp_idx = params['labels']['antibiotics'].index('AMP')\n"
         "splits = {\n"
         "    'random_split': random_split(Y, amp_idx, seed=params['seed']),\n"
         "    'phylogroup_stratified_split': phylogroup_stratified_split(cluster_labels, seed=params['seed']),\n"
         "    'lineage_holdout_split': lineage_holdout_split(groups_from_phylogroup(cluster_labels), seed=params['seed']),\n"
         "}\n"
         "diag = save_split_diagnostics(\n"
         "    splits, Y, label_mask, params['labels']['antibiotics'],\n"
         "    paths.results_dir / 'splits' / 'split_diagnostics.csv',\n"
         ")\n"
         "diag"),
    md("## PCA-1024 baseline (B8: sparse, memory-safe)"),
    code("from src.amr_data.baselines import build_pca_features\n\n"
         "rs = splits['random_split']\n"
         "Xtr, Xva, Xte, pca, scaler = build_pca_features(\n"
         "    snp_data['X_snp'], rs['train_idx'], rs['val_idx'], rs['test_idx'],\n"
         ")\n"
         "print(f'Variance retained: {pca.explained_variance_ratio_.sum():.4f}')"),
])

NOTEBOOK_02 = nb([
    md("# 02 — QA and Baselines\n\n"
       "Verification checks (`qa.py`) against whatever processed artifacts "
       "already exist, plus the PCA dimension search (manifest E7)."),
    code("import sys\nsys.path.insert(0, '..')\n\n"
         "from src.amr_data.config import load_paths\n"
         "from src.amr_data.io import load_pickle\n"
         "from src.amr_data import qa\n\n"
         "paths = load_paths()"),
    md("## Gene presence/absence QA (B4/B5/B6/B9)"),
    code("gf = load_pickle(paths.processed('gene_filtered_legacy_pkl'))\n"
         "pa = load_pickle(paths.processed('gene_pa_pkl'))\n"
         "report = qa.verify_gene_pa(pa, gf, genome_names_reference=gf['genome_names'])\n"
         "report"),
    code("qa.export_unmatched_strains(pa, paths.results_dir / 'qa' / 'unmatched_strains.csv')"),
    md("## PCA dimension search (reproduces manifest E7)"),
    code("from src.amr_data.baselines import pca_dim_search\n\n"
         "snp_data = load_pickle(paths.processed('snp_matrix_pkl'))\n"
         "master = load_pickle(paths.processed('master_features_pkl'))\n"
         "rs = master['random_split']\n\n"
         "results = pca_dim_search(\n"
         "    snp_data['X_snp'], rs['train_idx'], rs['val_idx'], rs['test_idx'],\n"
         "    master['Y'].astype('float64'), master['label_mask'], list(master['label_names']),\n"
         ")\n"
         "for dim, r in results.items():\n"
         "    print(f\"dim={dim}: var={r['variance_retained']:.3f}  avg_pr_auc={r['avg_pr_auc']:.4f}\")"),
])

NOTEBOOK_03 = nb([
    md("# 03 — Giessen External Validation\n\n"
       "Loads the Giessen dataset (canonical antibiotic naming applied), "
       "builds the split, and (if TensorFlow is available) runs the "
       "AMR-EnsembleNet reproduction. See `docs/OPEN_QUESTIONS.md` #7 for "
       "why this notebook's split is NOT the same as thesis Table 5.10's."),
    code("import sys\nsys.path.insert(0, '..')\n\n"
         "from src.amr_data.config import load_paths\n"
         "from src.amr_data.giessen import load_giessen_raw, build_giessen_split\n"
         "from src.amr_data import qa\n\n"
         "paths = load_paths()\n"
         "ds = load_giessen_raw(paths)\n"
         "qa.verify_giessen(ds)"),
    code("split = build_giessen_split(ds, seed=42)\n"
         "print({k: len(v) if hasattr(v, '__len__') else v for k, v in split.items()})"),
    md("## AMR-EnsembleNet reproduction (requires `tensorflow`)\n\n"
       "Attribution: Siddiqui & Tarannum (2025), arXiv:2509.23552, "
       "github.com/Saiful185/AMR-EnsembleNet. See "
       "`src/amr_data/amr_ensemblenet.py` module docstring -- NOT executed "
       "as part of this refactor (no TensorFlow in that environment)."),
    code("from src.amr_data import amr_ensemblenet as ae\n\n"
         "try:\n"
         "    import tensorflow  # noqa: F401\n"
         "    HAS_TF = True\n"
         "except ModuleNotFoundError:\n"
         "    HAS_TF = False\n"
         "    print('TensorFlow not installed -- skipping CNN reproduction. '\n"
         "          'See docs/OPEN_QUESTIONS.md #10.')\n\n"
         "if HAS_TF:\n"
         "    for ab in ds['label_names']:\n"
         "        j = ds['antibiotic_to_index'][ab]\n"
         "        m = ds['label_mask'][:, j]\n"
         "        tr = split['train_idx'][m[split['train_idx']]]\n"
         "        va = split['val_idx'][m[split['val_idx']]]\n"
         "        te = split['test_idx'][m[split['test_idx']]]\n"
         "        results = ae.run_one_antibiotic(\n"
         "            ds['X'][tr].astype('int32'), ds['Y'][tr, j].astype('int32'),\n"
         "            ds['X'][va].astype('int32'), ds['Y'][va, j].astype('int32'),\n"
         "            ds['X'][te].astype('int32'), ds['Y'][te, j].astype('int32'),\n"
         "            ab,\n"
         "        )\n"
         "        print(ab, results['Ensemble (soft vote, official)'])"),
])


if __name__ == "__main__":
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, notebook in [
        ("01_build_arcadia_dataset.ipynb", NOTEBOOK_01),
        ("02_qa_and_baselines.ipynb", NOTEBOOK_02),
        ("03_giessen_external.ipynb", NOTEBOOK_03),
    ]:
        out = NB_DIR / name
        with open(out, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1)
        print(f"Wrote {out}")
