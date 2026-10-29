from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Assembly size within lineages, and whether it is batch or biology.",
        {
            "within_lineage": module("trilens.analysis.assembly_size_within_lineage"),
            "mechanism": module("trilens.analysis.assembly_size_mechanism"),
        },
        choose="--part",
    )
