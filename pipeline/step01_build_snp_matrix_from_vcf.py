from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step("Stream the Arcadia variant file into an integer SNP matrix.", module("trilens.data.snp_matrix_from_vcf"))
