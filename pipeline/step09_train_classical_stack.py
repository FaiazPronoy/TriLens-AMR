from trilens.cli import configured, module, run_step

if __name__ == "__main__":
    run_step(
        "Five classical learners stacked by logistic regression.",
        {
            "random": configured("classical_stack", "trilens.experiments.classical_stack"),
            "phylogroup": module("trilens.experiments.classical_stack_phylogroup"),
            "lineage": module("trilens.experiments.classical_stack_lineage"),
        },
        choose="--split",
    )
