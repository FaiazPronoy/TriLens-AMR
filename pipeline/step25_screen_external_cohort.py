from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Find the external NCBI isolates with susceptibility results.",
        module("trilens.determinants.external_cohort_screen"),
    )
