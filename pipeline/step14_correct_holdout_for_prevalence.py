from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Resample the holdout test set to the training prevalence; prevalence-free metrics.",
        {
            "matching": module("trilens.analysis.prevalence_matching"),
            "invariant_metrics": module("trilens.experiments.prevalence_invariant_metrics"),
        },
        choose="--part",
    )
