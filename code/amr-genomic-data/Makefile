.PHONY: test qa splits gene-pa giessen-verify all-checks

test:
	pytest tests/ -v

qa:
	python -m src.amr_data.qa

splits:
	python scripts/make_splits.py

gene-pa:
	python scripts/build_gene_pa.py

giessen-verify:
	python -c "from src.amr_data.giessen import load_giessen; from src.amr_data import qa; ds = load_giessen(); qa.verify_giessen(ds)"

# Runs everything that CAN run without raw source data re-download, i.e.
# against whatever processed pkls already exist under data/processed/.
all-checks: test qa
