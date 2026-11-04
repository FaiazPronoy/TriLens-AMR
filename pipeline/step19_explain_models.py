from trilens.cli import configured, module, run_step

if __name__ == "__main__":
    run_step(
        "Gene importance, SHAP, marker annotation, attention checks and the strain conditioning table.",
        {
            "gain": configured("gene_importance_gain", "trilens.experiments.gene_importance_gain"),
            "shap": configured("gene_importance_shap", "trilens.experiments.gene_importance_shap"),
            "base_learner_shap": module("trilens.experiments.base_learner_shap"),
            "annotation": module("trilens.experiments.marker_gene_annotation"),
            "lineage_markers": module("trilens.analysis.lineage_marker_test"),
            "provenance": module("trilens.analysis.assembly_provenance_enrichment"),
            "attention": module("trilens.analysis.attention_diagnosis"),
            "strain_conditioning": module("trilens.analysis.strain_conditioning_table"),
        },
        choose="--part",
    )
