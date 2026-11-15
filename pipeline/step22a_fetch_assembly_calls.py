from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "AMRFinderPlus calls from the BV-BRC assemblies: repair genome IDs, install the pinned tool, fetch, summarise.",
        {
            "repair_ids": module("trilens.data.genome_id_repair"),
            "install": module("trilens.determinants.amrfinder_install"),
            "fetch": module("trilens.determinants.genome_download_calls", "fetch"),
            "summary": module("trilens.determinants.genome_download_calls", "summary"),
            "build": module("trilens.determinants.genome_download_calls", "build"),
        },
    )
