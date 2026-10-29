from trilens.cli import chain, configured, module, run_step

if __name__ == "__main__":
    run_step(
        "Keep the 5,000 most informative genes, fitted on each split's training part.",
        {
            "random": configured("gene_selection", "trilens.experiments.gene_selection"),
            "phylogroup": chain(
                module("trilens.experiments.gene_importance_ranking_phylogroup"),
                module("trilens.experiments.gene_filtered_dataset_phylogroup"),
            ),
            "lineage": chain(
                module("trilens.experiments.gene_importance_ranking_lineage"),
                module("trilens.experiments.gene_filtered_dataset_lineage"),
            ),
        },
        choose="--split",
    )
