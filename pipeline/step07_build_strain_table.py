from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Strain table, labels and metadata inventory used by every later step.",
        module("trilens.analysis.data_inventory"),
    )
