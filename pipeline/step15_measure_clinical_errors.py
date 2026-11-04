from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Very major and major errors, tuned thresholds, and the full metric table.",
        {
            "holdout": module("trilens.analysis.holdout_clinical_errors"),
            "thresholds": module("trilens.analysis.threshold_tuning"),
            "metrics": module("trilens.experiments.full_metric_suite"),
        },
        choose="--part",
    )
