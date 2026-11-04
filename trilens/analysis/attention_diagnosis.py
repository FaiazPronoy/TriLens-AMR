import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)
TRILENS = ROOT / "results" / "trilens_stack_random"
EGGNOG = ROOT / "data" / "processed" / "cds_eggNog.tsv"
SPARSITY = ROOT / "results" / "gene_attention_v2" / "attention_sparsity.csv"
OUT = DIAG / "attention_diagnosis.csv"

TOP_N = 100


AMR_PAT = re.compile(
    r"\b(gyrA|gyrB|parC|parE|acrA|acrB|acrD|acrE|acrF|tolC|marA|marR|soxS|"
    r"emrA|emrB|emrD|emrE|emrK|emrY|mdtA|mdtB|mdtC|mdtD|mdtE|mdtF|mdtG|mdtH|"
    r"mdtK|mdtL|mdtM|mdtN|mdtO|mdtP|mdfA|cusA|cusB|cusC|cusF|copA|"
    r"bla[A-Z]*|beta-lactamase|ampC|ampD|ampE|ampG|"
    r"aac\(|aad[AB]|aph\(|strA|strB|"
    r"sul1|sul2|sul3|dfrA|dfrB|folP|folA|"
    r"tetA|tetB|tetC|tetD|tetM|tetW|"
    r"catA|catB|cmlA|floR|"
    r"mcr-|pmrA|pmrB|phoP|phoQ|arnT|eptA|"
    r"ompC|ompF|ompK|porin)\b",
    re.IGNORECASE,
)


def entropy_stats(w):
    w = np.asarray(w, dtype=float)
    w = w[w > 0]
    w = w / w.sum()
    H = float(-(w * np.log(w)).sum())
    return H, float(np.exp(H)), float(np.sort(w)[::-1][:20].sum())


def prefix_of(gene_id):
    m = re.findall(r"\b([A-Z]{8})_\d{4,5}\b", str(gene_id))
    return m[0] if m else None


def main():
    if SPARSITY.exists():
        sp = pd.read_csv(SPARSITY)
        print("=" * 84)
        print("Q1  ATTENTION CONCENTRATION  (GeneAttentionNet sweep, 5000 genes)")
        print("=" * 84)
        print(f"Maximum possible entropy for 5000 genes: ln(5000) = {np.log(5000):.4f}\n")
        for exp in sorted(sp.exp_id.unique()):
            sub = sp[sp.exp_id == exp]
            n = int(sub.n_genes.iloc[0])
            print(
                f"--- {exp}  pool={sub.pool_type.iloc[0]}  "
                f"multi_task={sub.multi_task.iloc[0]}  n_genes={n} "
                f"(max entropy {np.log(n):.3f}) ---"
            )
            print(sub[["ab", "n_genes_above_1pct", "n_genes_for_90pct", "max_attn", "entropy"]].to_string(index=False))
            print()
    else:
        print(f"(skipped Q1 - {SPARSITY} not found)")

    frames = {}
    for lens in ("A", "B", "C"):
        f = TRILENS / f"attention_{lens}_topgenes.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["lens"] = lens
            frames[lens] = d
        else:
            print(f"WARNING: {f} not found")
    if not frames:
        sys.exit("No attention_*_topgenes.csv files found - cannot continue.")
    att = pd.concat(frames.values(), ignore_index=True)

    gene_col = "gene_id" if "gene_id" in att.columns else att.columns[2]
    w_col = "attn_weight" if "attn_weight" in att.columns else att.columns[3]
    print(
        f"\nLoaded {len(att)} attention rows across lenses {sorted(att.lens.unique())}, {att.ab.nunique()} antibiotics."
    )
    print(f"Rows per lens/drug: {len(att) // (att.lens.nunique() * att.ab.nunique())}")

    print("\n" + "=" * 84)
    print("Q1b  CONCENTRATION OF THE SAVED TriLens TOP-GENE WEIGHTS")
    print("=" * 84)
    rows = []
    for (lens, ab), g in att.groupby(["lens", "ab"]):
        H, eff, top20 = entropy_stats(g[w_col].values)
        rows.append(
            dict(
                lens=lens,
                ab=ab,
                n_rows=len(g),
                max_attn=round(g[w_col].max(), 6),
                entropy_of_saved=round(H, 3),
                effective_genes=round(eff, 1),
                top20_mass=round(top20, 3),
            )
        )
    conc = pd.DataFrame(rows).sort_values(["lens", "ab"])
    print(conc.to_string(index=False))

    print("\n" + "=" * 84)
    print("Q2  DO THE TOP GENES CLUSTER BY SOURCE GENOME PREFIX?")
    print("=" * 84)
    print("If one Prokka prefix dominates a drug's top genes, the attention is")
    print("tracking a lineage signature, not a mechanism.\n")

    all_prefixes = [p for p in (prefix_of(x) for x in att[gene_col]) if p]
    base = Counter(all_prefixes)
    n_pref = len(base)
    print(f"Distinct prefixes seen across all attention rows: {n_pref}")
    print(f"If uniform, each prefix would appear {len(all_prefixes) / max(n_pref, 1):.1f} times overall.\n")

    q2 = []
    for (lens, ab), g in att.groupby(["lens", "ab"]):
        pref = [p for p in (prefix_of(x) for x in g[gene_col]) if p]
        if not pref:
            continue
        c = Counter(pref)
        top_p, top_c = c.most_common(1)[0]
        q2.append(
            dict(
                lens=lens,
                ab=ab,
                n_with_prefix=len(pref),
                distinct_prefixes=len(c),
                top_prefix=top_p,
                top_prefix_count=top_c,
                top_prefix_share=round(top_c / len(pref), 3),
                expected_if_uniform=round(len(pref) / n_pref, 2),
            )
        )
    q2 = pd.DataFrame(q2).sort_values("top_prefix_share", ascending=False)
    print(q2.to_string(index=False))

    print("\n" + "=" * 84)
    print(f"Q3  ANY AMR GENE IN THE TOP {TOP_N} THAT THE TOP-20 CUT MISSED?")
    print("=" * 84)
    if not EGGNOG.exists():
        print(f"(skipped - {EGGNOG} not found)")
    else:
        egg = pd.read_csv(EGGNOG, sep="\t", low_memory=False)
        tag_col = next(
            (c for c in egg.columns if egg[c].astype(str).str.match(r"^[A-Z]{8}_\d{4,5}$").mean() > 0.5), egg.columns[0]
        )
        text_cols = [c for c in egg.columns if egg[c].dtype == object]
        egg["_blob"] = egg[text_cols].astype(str).agg(" | ".join, axis=1)
        lookup = dict(zip(egg[tag_col].astype(str), egg["_blob"]))
        print(f"eggNOG rows: {len(egg)}, key column: '{tag_col}'\n")

        hits, checked, matched = [], 0, 0
        for (lens, ab), g in att.groupby(["lens", "ab"]):
            for _, r in g.head(TOP_N).iterrows():
                checked += 1
                for tag in re.findall(r"\b[A-Z]{8}_\d{4,5}\b", str(r[gene_col])) or [str(r[gene_col])]:
                    blob = lookup.get(tag)
                    if blob is None:
                        continue
                    matched += 1
                    m = AMR_PAT.search(blob)
                    if m:
                        hits.append(
                            dict(
                                lens=lens,
                                ab=ab,
                                rank=r.get("rank"),
                                gene=r[gene_col],
                                locus=tag,
                                match=m.group(0),
                                desc=blob[:110].replace("\n", " "),
                            )
                        )
                    break
        print(
            f"Rows checked: {checked}   annotated: {matched} "
            f"({matched / max(checked, 1) * 100:.1f}%)   AMR hits: {len(hits)}"
        )
        if hits:
            h = pd.DataFrame(hits)
            print("\n*** AMR GENES FOUND IN NEURAL ATTENTION ***")
            print(h.to_string(index=False))
            h.to_csv(DIAG / "attention_amr_hits.csv", index=False)
            print(f"\nSaved -> {DIAG / 'attention_amr_hits.csv'}")
        else:
            print("\nNo known resistance gene found in the top %d attended genes." % TOP_N)

    conc.merge(q2, on=["lens", "ab"], how="outer").to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")

    print("""
How to read this
  Q1  Entropy near ln(n_genes) means the attention is close to uniform.
      Entropy well below it means the model focuses on a few genes.
  Q2  top_prefix_share far above expected_if_uniform means the attended
      genes come mostly from one assembly.
  Q3  attention_amr_hits.csv lists known resistance genes among the
      attended genes, if any.
""")


if __name__ == "__main__":
    main()
