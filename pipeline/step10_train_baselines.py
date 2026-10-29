from trilens.cli import configured, module, run_step

if __name__ == "__main__":
    run_step(
        "Baselines: PCA features, phylogroup label alone, and TabNet.",
        {
            "pca": configured("pca_baseline", "trilens.experiments.pca_baseline"),
            "phylogroup_only": module("trilens.experiments.phylogroup_only_baseline"),
            "tabnet_random": module("trilens.experiments.tabnet_baseline"),
            "tabnet_phylogroup": module("trilens.experiments.tabnet_baseline_phylogroup"),
            "tabnet_lineage": module("trilens.experiments.tabnet_baseline_lineage"),
        },
        choose="--model",
    )
