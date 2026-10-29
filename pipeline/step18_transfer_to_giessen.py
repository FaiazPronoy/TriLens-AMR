from trilens.cli import configured, run_step

if __name__ == "__main__":
    run_step(
        "Retrain the classical stack on the independent Giessen collection.",
        configured("giessen_transfer", "trilens.experiments.giessen_transfer"),
    )
