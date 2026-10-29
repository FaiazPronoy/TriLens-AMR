from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Retrain the neural branch with new seeds and recompute the holdout margin.",
        {
            "seeds": module("trilens.experiments.seed_stability"),
            "margin": module("trilens.analysis.holdout_margin"),
        },
        choose="--part",
    )
