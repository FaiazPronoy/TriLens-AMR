from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Country, time and five-fold splits, written once and read back from JSON.",
        {
            "country_time": module("trilens.splits.country_and_time_splits"),
            "folds": module("trilens.splits.fold_splits"),
            "load": module("trilens.splits.split_files"),
        },
        choose="--part",
    )
