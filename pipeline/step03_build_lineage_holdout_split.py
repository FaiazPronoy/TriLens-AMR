from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Hold out whole sequence types: the lineage holdout split.", module("trilens.experiments.lineage_holdout_split")
    )
