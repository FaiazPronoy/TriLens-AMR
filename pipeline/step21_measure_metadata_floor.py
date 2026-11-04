from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "How much is predictable from metadata and assembly statistics alone.",
        module("trilens.analysis.metadata_floor"),
    )
