from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Labels, phylogroups, the random and phylogroup splits, and 1,024 PCA features.",
        module("trilens.data.labels_and_splits"),
    )
