from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Add each strain's determinant calls to the pan-genome catalogue on unseen lineages.",
        module("trilens.experiments.determinant_rescue"),
    )
