from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Apply frozen models to the external cohort, then drop one BioProject at a time.",
        {
            "transfer": module("trilens.experiments.external_cohort"),
            "robustness": module("trilens.analysis.external_robustness"),
        },
        choose="--part",
    )
