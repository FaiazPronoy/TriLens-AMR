from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Coverage check and model against determinant rule on unseen lineages.",
        module("trilens.analysis.coverage_and_rule"),
    )
