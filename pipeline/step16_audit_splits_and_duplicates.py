from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Split integrity checks and the near-duplicate audit.",
        {
            "integrity": module("trilens.analysis.split_integrity_check"),
            "duplicates_random": module("trilens.experiments.near_duplicate_audit"),
            "duplicates_lineage": module("trilens.experiments.near_duplicate_audit_lineage"),
        },
        choose="--part",
    )
