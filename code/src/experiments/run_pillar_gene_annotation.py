"""
src/experiments/run_pillar_gene_annotation.py
================================================
Gene annotation of top marker genes using data/processed/cds_eggNog.tsv
(an eggNOG-mapper output supplied for this project, covering 72 distinct
Prokka locus-tag prefixes / 15,056 CDS entries -- NOT a single-genome
annotation).

data/processed/genes.gff INVESTIGATION (per instructions, checked before use):
its `##sequence-region` lines declare locus tags across many prefixes too
(ABCDEHGJ, AIHHOGBN, BCDBBOOO, BIKPOGCH, DHJNCGMO, ...) -- same general
multi-genome coverage as the eggNOG file. However, this script does NOT use
genes.gff for annotation: eggNOG's `Description`/`Preferred_name` columns
already give a directly usable functional label per locus tag, while the GFF
only gives coordinates/contig boundaries (no functional description) -- it
would add a coordinate cross-check but not additional annotation content, so
using eggNOG alone is sufficient and avoids an unnecessary GFF-parsing step.
(This differs from a past attempt, per the task's own note, where a GFF's
locus-tag scheme reportedly did not match this project's gene IDs at all;
that is NOT the case here -- both files use recognizable Prokka-style tags
overlapping this project's `selected_gene_ids`, confirmed by a direct
intersection check: 3326/5000 selected genes match an eggNOG query ID
verbatim before even resolving `Cluster_...` compound IDs.)

Two gene_id naming schemes appear in this project's top-marker-gene files:
  1. Plain Prokka locus tag, e.g. "LMHECDEF_04343" -- looked up directly.
  2. Pan-genome cluster compound name, e.g.
     "Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_
     KBHDICFF_04013_+_+_CO_F" -- one or more embedded locus tags are
     extracted via regex and each tried against the eggNOG lookup in order;
     the first match found is used (documented in the `matched_locus_tag`
     column so this is auditable, not hidden).

Canonical-AMR-gene check: the matched eggNOG `Preferred_name` AND
`Description` are checked against an explicit, fixed list of canonical AMR
gene-name regex patterns (beta-lactamases, quinolone-resistance targets,
aminoglycoside-modifying enzymes, sulfonamide/trimethoprim resistance,
tetracycline/macrolide/chloramphenicol resistance, efflux pumps, mcr).
No name is inferred or guessed -- unmatched genes are explicitly marked
"no match", never given a fabricated name.

Usage:
    conda activate amr
    python -m src.experiments.run_pillar_gene_annotation
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "pillar_gene_annotation"
EGGNOG_TSV = ROOT / "data" / "processed" / "cds_eggNog.tsv"

LOCUS_TAG_RE = re.compile(r"[A-Za-z]{6,10}_\d{4,6}")

# Canonical AMR gene-name check, split into two DELIBERATELY STRICT checks to
# avoid substring false positives (e.g. a naive `tet` substring search matches
# "Tetratricopeptide repeats" -- a completely unrelated protein domain name --
# caught and fixed during this script's own development, not a hypothetical):
#   1. `Preferred_name` (eggNOG's clean short gene-symbol column) must be an
#      EXACT, field-anchored match to a canonical AMR gene symbol pattern.
#   2. `Description` is only checked against specific multi-word phrases that
#      cannot plausibly collide with unrelated protein names.
_NAME_PATTERNS = (
    r"^(bla[a-z0-9_-]*|gyr[ab]|par[ce]|aac\(.*|aph\(.*|ant\(.*|sul[123]|"
    r"dfr[a-z0-9]*|tet[a-z]|mcr-?\d*|qnr[a-z]|flor|cat\d?|erm[a-z]|mph[a-z]|"
    r"oqx[ab]|acr[ab]|tolc|ampc|marr|mexy|vana|fosa|emr[ab]|mdt[a-z]|cusa)$"
)
NAME_REGEX = re.compile(_NAME_PATTERNS, re.IGNORECASE)
DESC_PHRASES = (
    r"beta-lactamase|quinolone resistance|"
    r"aminoglycoside.{0,20}(transferase|phosphotransferase|acetyltransferase|nucleotidyltransferase)|"
    r"multidrug efflux|tetracycline resistance|chloramphenicol acetyltransferase|"
    r"sulfonamide resistance|trimethoprim resistance|macrolide resistance"
)
DESC_REGEX = re.compile(DESC_PHRASES, re.IGNORECASE)

# the main TriLens run (delivered strain-conditioned model); was trilens_v10 (pre-fix)
MAIN_RUN_DIR = RESULTS_DIR / os.environ.get("TRILENS_MAIN_RUN", "trilens_attnfix")

TOP_GENE_SOURCES = [
    ("TriLens_LensA_attention", MAIN_RUN_DIR / "attention_A_topgenes.csv", "gene_id"),
    ("TriLens_LensB_attention", MAIN_RUN_DIR / "attention_B_topgenes.csv", "gene_id"),
    ("TriLens_LensC_attention", MAIN_RUN_DIR / "attention_C_topgenes.csv", "gene_id"),
    ("T6_SHAP_gene_based", RESULTS_DIR / "pillar_shap_t6" / "shap_top_genes_gene_based.csv", "gene_id"),
    ("T6_SHAP_raw_chi2", RESULTS_DIR / "pillar_shap_t6" / "shap_top_genes_raw_chi2.csv", "gene_id"),
    ("pillar3_SHAP_xgb_standalone", RESULTS_DIR / "pillar3_shap" / "multi_drug_markers_shap.csv", "gene_id"),
    ("pillar3_gain_xgb_standalone", RESULTS_DIR / "pillar3_xgb_importance" / "multi_drug_markers.csv", "gene_id"),
]


def _load_eggnog_lookup() -> dict:
    df = pd.read_csv(EGGNOG_TSV, sep="\t", low_memory=False)
    lookup = {}
    for _, r in df.iterrows():
        lookup[str(r["query"])] = {
            "Description": r.get("Description", "-"),
            "Preferred_name": r.get("Preferred_name", "-"),
            "PFAMs": r.get("PFAMs", "-"),
            "COG_category": r.get("COG_category", "-"),
        }
    return lookup


def _resolve_gene_annotation(gene_id: str, lookup: dict):
    """Returns (matched_locus_tag_or_None, annotation_dict_or_None)."""
    if gene_id in lookup:
        return gene_id, lookup[gene_id]
    for candidate in LOCUS_TAG_RE.findall(gene_id):
        if candidate in lookup:
            return candidate, lookup[candidate]
    return None, None


def _is_canonical_amr(ann: dict) -> bool:
    name = str(ann.get("Preferred_name", "") or "").strip()
    desc = str(ann.get("Description", "") or "").strip()
    if name and name != "-" and NAME_REGEX.match(name):
        return True
    if DESC_REGEX.search(desc):
        return True
    return False


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading eggNOG annotation table: {EGGNOG_TSV}")
    lookup = _load_eggnog_lookup()
    print(f"  {len(lookup)} annotated locus tags loaded")

    rows = []
    for source_name, path, gene_col in TOP_GENE_SOURCES:
        if not path.exists():
            print(f"  SKIP (file not found): {path}")
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            gene_id = str(r[gene_col])
            matched_tag, ann = _resolve_gene_annotation(gene_id, lookup)
            if ann is None:
                rows.append({
                    "source": source_name,
                    "gene_id": gene_id,
                    "matched_locus_tag": "no match",
                    "description": "no match",
                    "preferred_name": "no match",
                    "pfams": "no match",
                    "is_canonical_amr_gene": "❌ (no match)",
                    **{k: r[k] for k in df.columns if k != gene_col},
                })
            else:
                is_amr = _is_canonical_amr(ann)
                rows.append({
                    "source": source_name,
                    "gene_id": gene_id,
                    "matched_locus_tag": matched_tag,
                    "description": ann["Description"],
                    "preferred_name": ann["Preferred_name"],
                    "pfams": ann["PFAMs"],
                    "is_canonical_amr_gene": "✅" if is_amr else "❌",
                    **{k: r[k] for k in df.columns if k != gene_col},
                })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_DIR / "annotated_markers.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'annotated_markers.csv'}  ({len(out_df)} rows)")

    _write_summary(out_df, n_eggnog=len(lookup))


def _write_summary(df: pd.DataFrame, n_eggnog: int) -> None:
    total = len(df)
    matched = int((df["matched_locus_tag"] != "no match").sum())
    coverage_pct = 100.0 * matched / total if total else 0.0
    amr_hits = int((df["is_canonical_amr_gene"] == "✅").sum())

    lines = ["# Gene Annotation — Top Marker Genes (eggNOG)\n"]
    lines.append(
        f"Source: `data/processed/cds_eggNog.tsv` ({n_eggnog} "
        f"annotated locus tags, 72 distinct Prokka prefixes). "
        f"`data/processed/genes.gff` was inspected but not used for annotation "
        f"(no functional description column, only coordinates) -- see script "
        f"docstring for the full investigation.\n"
    )
    lines.append(f"**Overall coverage: {matched}/{total} rows matched ({coverage_pct:.1f}%).**\n")
    lines.append(f"**Canonical AMR gene name pattern hits: {amr_hits}/{total} rows ({100*amr_hits/total:.1f}%).**\n")

    lines.append("## Coverage by source\n")
    lines.append("| Source | Total rows | Matched | Coverage % | Canonical AMR hits |")
    lines.append("|---|---|---|---|---|")
    for source in df["source"].unique():
        sub = df[df["source"] == source]
        m = int((sub["matched_locus_tag"] != "no match").sum())
        a = int((sub["is_canonical_amr_gene"] == "✅").sum())
        lines.append(f"| {source} | {len(sub)} | {m} | {100*m/len(sub):.1f}% | {a} |")
    lines.append("")

    lines.append("## Canonical AMR gene matches found\n")
    amr_df = df[df["is_canonical_amr_gene"] == "✅"]
    if len(amr_df):
        lines.append("| Source | Gene ID | Matched locus tag | Preferred name | Description |")
        lines.append("|---|---|---|---|---|")
        for _, r in amr_df.drop_duplicates(subset=["matched_locus_tag"]).iterrows():
            lines.append(f"| {r['source']} | {r['gene_id'][:40]} | {r['matched_locus_tag']} | {r['preferred_name']} | {str(r['description'])[:80]} |")
    else:
        lines.append("None found among the top marker genes checked.")
    lines.append("")

    lines.append(
        "## Honest caveat\n\n"
        "A `no match` result means the eggNOG file (72 genome prefixes) did "
        "not contain that specific locus tag -- it does NOT mean the gene is "
        "unannotatable in general, only that this particular reference "
        "genome's annotation wasn't in the supplied eggNOG file for that "
        "specific pan-genome cluster representative. No gene name was "
        "inferred or guessed for unmatched rows."
    )

    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'SUMMARY.md'}")
    print(f"\nCoverage: {matched}/{total} ({coverage_pct:.1f}%)  |  Canonical AMR hits: {amr_hits}")


if __name__ == "__main__":
    main()
