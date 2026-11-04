from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Stack the neural branches with the classical learners on the phylogroup and lineage splits.",
        {
            "phylogroup": module("trilens.experiments.stack_phylogroup"),
            "lineage": module("trilens.experiments.stack_lineage"),
        },
        choose="--split",
        env_options={
            "--attention": {
                "conditioned": {"TRILENS_ATTENTION": "conditioned"},
                "standard": {"TRILENS_ATTENTION": "standard"},
            }
        },
    )
