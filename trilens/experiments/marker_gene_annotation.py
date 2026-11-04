from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "marker_gene_annotation"
EGGNOG_TSV = ROOT / "data" / "processed" / "cds_eggNog.tsv"

LOCUS_TAG_RE = re.compile(r"[A-Za-z]{6,10}_\d{4,6}")


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


MAIN_RUN_DIR = RESULTS_DIR / "trilens_stack_random"

TOP_GENE_SOURCES = [
    ("TriLens_LensA_attention", MAIN_RUN_DIR / "attention_A_topgenes.csv", "gene_id"),
    ("TriLens_LensB_attention", MAIN_RUN_DIR / "attention_B_topgenes.csv", "gene_id"),
    ("TriLens_LensC_attention", MAIN_RUN_DIR / "attention_C_topgenes.csv", "gene_id"),
    ("full_stack_SHAP_gene_based", RESULTS_DIR / "base_learner_shap" / "shap_top_genes_gene_based.csv", "gene_id"),
    ("full_stack_SHAP_raw_chi2", RESULTS_DIR / "base_learner_shap" / "shap_top_genes_raw_chi2.csv", "gene_id"),
    ("shap_xgb_standalone", RESULTS_DIR / "gene_importance_shap" / "multi_drug_markers_shap.csv", "gene_id"),
    ("gain_xgb_standalone", RESULTS_DIR / "gene_importance_gain" / "multi_drug_markers.csv", "gene_id"),
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
                rows.append(
                    {
                        "source": source_name,
                        "gene_id": gene_id,
                        "matched_locus_tag": "no match",
                        "description": "no match",
                        "preferred_name": "no match",
                        "pfams": "no match",
                        "is_canonical_amr_gene": "no match",
                        **{k: r[k] for k in df.columns if k != gene_col},
                    }
                )
            else:
                is_amr = _is_canonical_amr(ann)
                rows.append(
                    {
                        "source": source_name,
                        "gene_id": gene_id,
                        "matched_locus_tag": matched_tag,
                        "description": ann["Description"],
                        "preferred_name": ann["Preferred_name"],
                        "pfams": ann["PFAMs"],
                        "is_canonical_amr_gene": "yes" if is_amr else "no",
                        **{k: r[k] for k in df.columns if k != gene_col},
                    }
                )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_DIR / "annotated_markers.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'annotated_markers.csv'}  ({len(out_df)} rows)")

    _write_summary(out_df, n_eggnog=len(lookup))


def _write_summary(df: pd.DataFrame, n_eggnog: int) -> None:
    total = len(df)
    matched = int((df["matched_locus_tag"] != "no match").sum())
    coverage_pct = 100.0 * matched / total if total else 0.0
    amr_hits = int((df["is_canonical_amr_gene"] == "yes").sum())

    lines = ["# Gene Annotation - Top Marker Genes (eggNOG)\n"]
    lines.append(
        f"Source: `data/processed/cds_eggNog.tsv` ({n_eggnog} "
        f"annotated locus tags, 72 distinct Prokka prefixes). "
        f"`data/processed/genes.gff` was inspected but not used for annotation "
        f"(no functional description column, only coordinates) -- see script "
        f"docstring for the full investigation.\n"
    )
    lines.append(f"**Overall coverage: {matched}/{total} rows matched ({coverage_pct:.1f}%).**\n")
    lines.append(
        f"**Canonical AMR gene name pattern hits: {amr_hits}/{total} rows ({100 * amr_hits / total:.1f}%).**\n"
    )

    lines.append("## Coverage by source\n")
    lines.append("| Source | Total rows | Matched | Coverage % | Canonical AMR hits |")
    lines.append("|---|---|---|---|---|")
    for source in df["source"].unique():
        sub = df[df["source"] == source]
        m = int((sub["matched_locus_tag"] != "no match").sum())
        a = int((sub["is_canonical_amr_gene"] == "yes").sum())
        lines.append(f"| {source} | {len(sub)} | {m} | {100 * m / len(sub):.1f}% | {a} |")
    lines.append("")

    lines.append("## Canonical AMR gene matches found\n")
    amr_df = df[df["is_canonical_amr_gene"] == "yes"]
    if len(amr_df):
        lines.append("| Source | Gene ID | Matched locus tag | Preferred name | Description |")
        lines.append("|---|---|---|---|---|")
        for _, r in amr_df.drop_duplicates(subset=["matched_locus_tag"]).iterrows():
            lines.append(
                f"| {r['source']} | {r['gene_id'][:40]} | {r['matched_locus_tag']} | "
                f"{r['preferred_name']} | {str(r['description'])[:80]} |"
            )
    else:
        lines.append("None found among the top marker genes checked.")
    lines.append("")

    lines.append(
        "## Caveat\n\n"
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
