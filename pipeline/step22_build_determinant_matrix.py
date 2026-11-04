from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Strain by determinant matrix from the AMRFinderPlus calls of NCBI Pathogen Detection.",
        module("trilens.determinants.determinant_matrix"),
    )
