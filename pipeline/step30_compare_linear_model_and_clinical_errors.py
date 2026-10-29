from trilens.cli import module, run_step

if __name__ == "__main__":
    run_step(
        "Sparse logistic regression against gradient boosting and the rule, with clinical error rates.",
        module("trilens.analysis.linear_model_and_clinical_errors"),
    )
