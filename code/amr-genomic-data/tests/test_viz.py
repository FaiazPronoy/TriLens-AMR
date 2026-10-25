import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.amr_data import viz
from src.amr_data.gene_filter import build_gene_to_snp_ranges
from src.amr_data.giessen import build_giessen_split, load_giessen_raw

# ---------------------------------------------------------------------------
# Synthetic-data structural tests -- fast, no real artifacts required
# ---------------------------------------------------------------------------

def test_fig_label_overview_synthetic():
    prevalence = {"AMP": 0.5, "CIP": 0.2, "GEN": 0.1}
    coverage = {"AMP": 0.7, "CIP": 0.9, "GEN": 0.85}
    fig = viz.fig_label_overview(prevalence, coverage, n_samples=1000)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_fig_population_structure_folds_to_other():
    counts = {"B2": 3542, "Unknown": 1697, "D": 723, "A": 317, "F": 307, "B1": 258, "C": 114, "E": 4}
    fig = viz.fig_population_structure(counts, donut_top_n=4)
    donut_ax = fig.axes[1]
    wedges = [p for p in donut_ax.patches if isinstance(p, matplotlib.patches.Wedge)]
    assert len(wedges) == 5  # top 4 + "Other"
    plt.close(fig)


def test_fig_population_structure_no_other_when_within_top_n():
    counts = {"A": 10, "B": 5, "C": 2}
    fig = viz.fig_population_structure(counts, donut_top_n=4)
    donut_ax = fig.axes[1]
    wedges = [p for p in donut_ax.patches if isinstance(p, matplotlib.patches.Wedge)]
    assert len(wedges) == 3  # no "Other" slice needed
    plt.close(fig)


def test_fig_split_composition_synthetic():
    split_sizes = {
        "regime_a": {"train_idx": 100, "val_idx": 20, "test_idx": 20},
        "regime_b": {"train_idx": 90, "val_idx": 30, "test_idx": 20},
    }
    fig = viz.fig_split_composition(split_sizes)
    ax = fig.axes[0]
    bars = [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]
    assert len(bars) == 6  # 2 regimes x 3 (train/val/test)
    plt.close(fig)


def test_fig_split_prevalence_shift_diverging_colors():
    diagnostics = pd.DataFrame([
        {"split": "s1", "antibiotic": "AMP", "prev_delta": 0.30},
        {"split": "s1", "antibiotic": "CIP", "prev_delta": -0.20},
        {"split": "s2", "antibiotic": "AMP", "prev_delta": -0.01},
        {"split": "s2", "antibiotic": "CIP", "prev_delta": 0.01},
    ])
    fig = viz.fig_split_prevalence_shift(diagnostics, antibiotic_order=["AMP", "CIP"])
    assert len(fig.axes) == 2
    ax_s1 = fig.axes[0]
    bars = [p for p in ax_s1.patches if isinstance(p, matplotlib.patches.Rectangle)]
    colors = {tuple(np.round(b.get_facecolor(), 3)) for b in bars}
    pos_rgb = tuple(np.round(matplotlib.colors.to_rgba(viz.DIVERGING_POS), 3))
    neg_rgb = tuple(np.round(matplotlib.colors.to_rgba(viz.DIVERGING_NEG), 3))
    assert pos_rgb in colors and neg_rgb in colors
    plt.close(fig)


def test_fig_gene_feature_space_synthetic():
    rng = np.random.RandomState(0)
    sizes_before = rng.randint(1, 500, size=200)
    sizes_after = rng.randint(5, 500, size=50)
    gene_prevalence = rng.beta(0.3, 3, size=500)
    per_strain_gene_count = rng.randint(2000, 8000, size=100)
    gene_prevalence_stats = {"rare_lt05": 300, "shell": 200, "core_gt95": 0}
    per_strain_gene_count_stats = {"mean": 5000.0, "median": 4900}
    fig = viz.fig_gene_feature_space(
        sizes_before, sizes_after, gene_prevalence, per_strain_gene_count, min_snps_per_gene=5,
        gene_prevalence_stats=gene_prevalence_stats, per_strain_gene_count_stats=per_strain_gene_count_stats,
    )
    assert len(fig.axes) == 3
    plt.close(fig)


def test_fig_giessen_overview_synthetic():
    prevalence = {"CIP": 0.45, "CTX": 0.44, "CAZ": 0.34, "GEN": 0.23}
    split_sizes = {"train_idx": 566, "val_idx": 121, "test_idx": 122}
    fig = viz.fig_giessen_overview(prevalence, split_sizes)
    assert len(fig.axes) == 2
    pie_ax = fig.axes[1]
    wedges = [p for p in pie_ax.patches if isinstance(p, matplotlib.patches.Wedge)]
    assert len(wedges) == 3
    plt.close(fig)


# ---------------------------------------------------------------------------
# Real-data smoke tests -- confirm the figures render from the actual
# production artifacts, using the session-scoped fixtures in conftest.py
# ---------------------------------------------------------------------------

def test_fig_label_overview_real_data(master_features):
    mf = master_features
    Y = mf["Y"].astype(float)
    label_mask = mf["label_mask"]
    label_names = list(mf["label_names"])
    prevalence = {ab: float(Y[label_mask[:, j], j].mean()) for j, ab in enumerate(label_names)}
    coverage = {ab: float(label_mask[:, j].mean()) for j, ab in enumerate(label_names)}
    assert len(prevalence) == 7
    fig = viz.fig_label_overview(prevalence, coverage, mf["n_samples"])
    assert len(fig.axes) == 2
    plt.close(fig)


def test_fig_population_structure_real_data(master_features):
    mf = master_features
    cluster_labels = mf["cluster_labels"]
    int_to_phylogroup = {v: k for k, v in mf["phylogroup_to_int"].items()}
    counts = {
        int_to_phylogroup[int(code)]: int((cluster_labels == code).sum())
        for code in np.unique(cluster_labels)
    }
    assert sum(counts.values()) == mf["n_samples"]
    assert counts["B2"] == 3542  # locked reference count, see docs/CHANGELOG_NUMBERS.md
    fig = viz.fig_population_structure(counts)
    plt.close(fig)


def test_gene_size_before_after_matches_filter_metadata(snp_matrix, gene_filtered_legacy):
    gene_to_indices = build_gene_to_snp_ranges(snp_matrix["snp_positions"])
    sizes_before = np.array([len(v) for v in gene_to_indices.values()])
    assert sizes_before.sum() == snp_matrix["X_snp"].shape[1]

    sizes_after = np.array([end - start for start, end in gene_filtered_legacy["gene_to_range"].values()])
    assert sizes_after.sum() == gene_filtered_legacy["n_snps_filtered"]
    assert len(sizes_after) == gene_filtered_legacy["filter_metadata"]["n_genes_selected"]


def test_fig_gene_feature_space_real_data(snp_matrix, gene_filtered_legacy, gene_pa):
    gene_to_indices = build_gene_to_snp_ranges(snp_matrix["snp_positions"])
    sizes_before = np.array([len(v) for v in gene_to_indices.values()])
    sizes_after = np.array([end - start for start, end in gene_filtered_legacy["gene_to_range"].values()])
    gene_prevalence = gene_pa["X_pa"].mean(axis=0).astype(np.float64)
    per_strain_gene_count = gene_pa["X_pa"].sum(axis=1).astype(np.int64)

    fig = viz.fig_gene_feature_space(
        sizes_before, sizes_after, gene_prevalence, per_strain_gene_count,
        min_snps_per_gene=gene_filtered_legacy["filter_metadata"]["min_snps_per_gene"],
        gene_prevalence_stats=gene_pa["gene_prevalence_stats"],
        per_strain_gene_count_stats=gene_pa["per_strain_gene_count_stats"],
    )
    assert len(fig.axes) == 3
    plt.close(fig)


def test_giessen_overview_real_data(paths):
    ds = load_giessen_raw(paths)
    assert ds["n_samples"] == 809
    prevalence = {ab: float(np.nanmean(ds["Y"][:, i])) for i, ab in enumerate(ds["label_names"])}
    split = build_giessen_split(ds)
    split_sizes = {k: len(split[k]) for k in ("train_idx", "val_idx", "test_idx")}
    assert sum(split_sizes.values()) == 809

    fig = viz.fig_giessen_overview(prevalence, split_sizes)
    assert len(fig.axes) == 2
    plt.close(fig)
