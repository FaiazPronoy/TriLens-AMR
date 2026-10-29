from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Relabel the external cohort from MICs with EUCAST breakpoints.", module("trilens.analysis.mic_harmonisation")
    )
