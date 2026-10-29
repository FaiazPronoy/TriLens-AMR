from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Remove one part of the model at a time.",
        {
            "branches_averaged": module("trilens.experiments.branch_average_ablation"),
            "table": module("trilens.experiments.component_ablation"),
        },
        choose="--part",
    )
