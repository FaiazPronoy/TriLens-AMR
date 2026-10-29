from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Screen every organism, then compare model and rule in Campylobacter.",
        {
            "screen": module("trilens.determinants.species_screen"),
            "campylobacter": module("trilens.analysis.campylobacter"),
        },
        choose="--part",
    )
