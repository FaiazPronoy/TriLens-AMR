from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Re-read SNP columns whose stored values fall outside 0 to 3 from the SNP matrix.",
        module("trilens.data.snp_block_repair"),
    )
