from trilens.cli import configured, module, run_step

if __name__ == "__main__":
    run_step(
        "The three neural branches with their gate, stacked with the classical learners.",
        {
            "random": configured("trilens_stack", "trilens.experiments.trilens_stack"),
            "phylogroup": module("trilens.experiments.trilens_phylogroup"),
            "lineage": module("trilens.experiments.trilens_lineage"),
        },
        choose="--split",
        env_options={
            "--attention": {
                "conditioned": {"TRILENS_ATTENTION": "conditioned"},
                "standard": {"TRILENS_ATTENTION": "standard"},
            }
        },
    )
