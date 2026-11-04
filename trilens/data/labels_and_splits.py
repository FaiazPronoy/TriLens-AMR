from __future__ import annotations

import argparse
import gc
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ANTIBIOTICS = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]
PHENOTYPE_COLUMNS = {
    "AMP": "ampicillin",
    "CIP": "ciprofloxacin",
    "AMC": "amoxicillin/clavulanic acid",
    "SXT": "trimethoprim/sulfamethoxazole",
    "GEN": "gentamicin",
    "CTX": "cefotaxime",
    "CAZ": "ceftazidime",
}
ST_TO_PHYLOGROUP = {
    "131": "B2",
    "73": "B2",
    "95": "B2",
    "127": "B2",
    "12": "B2",
    "141": "B2",
    "144": "B2",
    "372": "B2",
    "404": "B2",
    "416": "B2",
    "428": "B2",
    "550": "B2",
    "1193": "B2",
    "14": "B2",
    "80": "B2",
    "498": "B2",
    "538": "B2",
    "983": "B2",
    "69": "D",
    "405": "D",
    "354": "D",
    "38": "D",
    "117": "D",
    "115": "D",
    "457": "D",
    "362": "D",
    "10": "A",
    "167": "A",
    "617": "A",
    "44": "A",
    "744": "A",
    "48": "A",
    "93": "A",
    "227": "A",
    "1284": "A",
    "168": "A",
    "752": "A",
    "224": "A",
    "11": "B1",
    "58": "B1",
    "155": "B1",
    "101": "B1",
    "162": "B1",
    "297": "B1",
    "446": "B1",
    "453": "B1",
    "517": "B1",
    "200": "B1",
    "62": "F",
    "59": "F",
    "522": "F",
    "648": "F",
    "88": "C",
    "410": "C",
    "23": "C",
    "350": "E",
    "57": "E",
}
PCA_COMPONENTS = 1024
SEED = 42


def parse_phylogroup(value):
    if pd.isna(value):
        return None
    found = re.findall(r"phylogroup:([A-Z][12]?)", str(value))
    return found[0] if found else None


def parse_sequence_type(value):
    if pd.isna(value):
        return None
    found = re.search(r"MLST\.Escherichia_coli_1\.(\d+)", str(value))
    return found.group(1) if found else None


def to_binary(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"r", "resistant", "i", "intermediate"}:
        return 1.0
    if s in {"s", "susceptible"}:
        return 0.0
    return np.nan


def three_way_split(n, stratify):
    idx = np.arange(n)
    trainval, test = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=stratify)
    train, val = train_test_split(trainval, test_size=0.15 / 0.85, random_state=SEED, stratify=stratify[trainval])
    return {"train_idx": train, "val_idx": val, "test_idx": test}


def build(snp_matrix_path, release_dir, out_path):
    release_dir = Path(release_dir)
    with open(snp_matrix_path, "rb") as f:
        snp = pickle.load(f)
    X_snp = snp["X_snp"]
    genome_names = snp["selected_genome_names"]
    sra_ids = snp["selected_sra_ids"]

    pheno = pd.read_csv(release_dir / "phenotype_matrix" / "phenotype_matrix_08302024.csv")
    metadata = pd.read_csv(release_dir / "strains_metadata.csv")
    samples = pd.DataFrame({"Genome.Name": genome_names, "SRA.Accession": sra_ids})
    samples = samples.merge(pheno, on="Genome.Name", how="left")
    samples = samples.merge(metadata[["Genome.Name", "MLST", "Other.Typing"]], on="Genome.Name", how="left")

    samples["phylogroup_final"] = (
        samples["Other.Typing"]
        .apply(parse_phylogroup)
        .fillna(samples["MLST"].apply(parse_sequence_type).map(ST_TO_PHYLOGROUP))
        .fillna("Unknown")
    )
    groups = sorted(samples["phylogroup_final"].unique())
    group_to_int = {g: i for i, g in enumerate(groups)}
    cluster_labels = np.array([group_to_int[g] for g in samples["phylogroup_final"]], dtype=np.int32)
    onehot = np.zeros((len(samples), len(groups)), dtype=np.float32)
    onehot[np.arange(len(samples)), cluster_labels] = 1.0

    n = len(samples)
    Y = np.full((n, len(ANTIBIOTICS)), np.nan, dtype=np.float32)
    for i, ab in enumerate(ANTIBIOTICS):
        Y[:, i] = samples[PHENOTYPE_COLUMNS[ab]].apply(to_binary).values
    label_mask = ~np.isnan(Y)
    Y_bool = np.nan_to_num(Y, nan=0.0).astype(bool)

    amp_strata = np.where(np.isnan(Y[:, 0]), -1, Y[:, 0].astype(int))
    random_split = three_way_split(n, amp_strata)
    phylogeny_split = three_way_split(n, cluster_labels)
    tr, va, te = random_split["train_idx"], random_split["val_idx"], random_split["test_idx"]
    print(
        f"random split {len(tr)}/{len(va)}/{len(te)}; phylogroup split "
        f"{len(phylogeny_split['train_idx'])}/{len(phylogeny_split['val_idx'])}/{len(phylogeny_split['test_idx'])}"
    )

    X = X_snp.astype(np.float32)
    del X_snp
    gc.collect()
    pca = TruncatedSVD(n_components=PCA_COMPONENTS, random_state=SEED)
    Xtr = pca.fit_transform(X[tr])
    Xva = pca.transform(X[va])
    Xte = pca.transform(X[te])
    variance = pca.explained_variance_ratio_.sum()
    del X
    gc.collect()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
    Xva = scaler.transform(Xva).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    master = {
        "X_pca_train": Xtr,
        "X_pca_val": Xva,
        "X_pca_test": Xte,
        "pca_var_retained": variance,
        "pca_n_components": PCA_COMPONENTS,
        "cluster_labels": cluster_labels,
        "phylogroup_onehot": onehot,
        "phylogroup_names": np.array(groups),
        "phylogroup_to_int": group_to_int,
        "Y": Y_bool,
        "label_mask": label_mask,
        "label_names": np.array(ANTIBIOTICS),
        "antibiotic_to_index": {ab: i for i, ab in enumerate(ANTIBIOTICS)},
        "random_split": random_split,
        "phylogeny_split": phylogeny_split,
        "genome_names": np.array(genome_names),
        "sra_ids": np.array(sra_ids),
        "n_samples": n,
        "n_pca_features": Xtr.shape[1],
        "n_antibiotics": len(ANTIBIOTICS),
        "n_phylogroups": len(groups),
        "preprocess_meta": {
            "source": "Arcadia Science E. coli release, variant file",
            "feature_type": "SNP integer encoding (0-3), then PCA-1024",
            "label_encoding": "R and I = 1, S = 0, untested = masked",
            "phylogroup_source": "Other.Typing field, then a sequence-type lookup",
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(master, f, protocol=4)
    print(f"saved {out_path}; PCA keeps {variance:.3f} of the variance")


def main():
    ap = argparse.ArgumentParser(description="Labels, phylogroups, the random and phylogroup splits, and PCA features.")
    ap.add_argument("--snp-matrix", default="data/processed/arcadia_v2_snp_matrix.pkl")
    ap.add_argument(
        "--release-dir", required=True, help="the Ecoli_AMR_GenotypePhenotype folder of the Arcadia release"
    )
    ap.add_argument("--out", default="data/processed/arcadia_v2_master_features.pkl")
    a = ap.parse_args()
    build(a.snp_matrix, a.release_dir, a.out)


if __name__ == "__main__":
    main()
