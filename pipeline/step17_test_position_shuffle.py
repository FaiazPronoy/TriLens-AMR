from trilens.cli import configured, module, run_step

if __name__ == "__main__":
    run_step(
        "Does variant order carry signal? Train before and after shuffling the columns.",
        {
            "run": configured("position_shuffle_test", "trilens.experiments.position_shuffle_test"),
            "summary": module("trilens.experiments.position_shuffle_summary"),
        },
        choose="--part",
    )
