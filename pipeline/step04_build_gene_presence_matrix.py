from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Download the PLINK presence/absence files and build the matrix with the right orientation.",
        {
            "download": module("trilens.data.gene_presence_source"),
            "build": module("trilens.data.gene_presence_rebuild"),
        },
        choose="--part",
    )
