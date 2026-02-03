.PHONY: install download process label analyze figures notebooks test clean help

# Default model for labeling
MODEL ?= llama3.2:latest
BATCH_SIZE ?= 100

# Notebook directory
NOTEBOOKS_DIR := notebooks

help:
	@echo "Quant-Sycophantic Research Pipeline"
	@echo ""
	@echo "Usage:"
	@echo "  make install        Install dependencies"
	@echo "  make download       Download LMSYS dataset"
	@echo "  make process        Process raw data into analysis-ready format"
	@echo "  make label          Run sycophancy labeling (MODEL=llama3.2:8b BATCH_SIZE=100)"
	@echo "  make analyze        Run full analysis pipeline (CLI + all notebooks)"
	@echo "  make notebooks      Run all analysis notebooks in sequence"
	@echo "  make figures        Generate publication figures"
	@echo "  make test           Run tests"
	@echo "  make clean          Remove generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make label MODEL=mistral BATCH_SIZE=50"
	@echo "  make label-resume   Resume interrupted labeling"
	@echo "  make notebook NB=03  Run specific notebook (01-05)"

install:
	pip install -e ".[dev]"

download:
	python -m quant_syco download

process:
	python -m quant_syco process

label:
	python -m quant_syco label --model $(MODEL) --batch-size $(BATCH_SIZE)

label-resume:
	python -m quant_syco label --model $(MODEL) --batch-size $(BATCH_SIZE) --resume

analyze: analyze-cli notebooks
	@echo "Full analysis complete! Check data/outputs/ and paper/figures/"

analyze-cli:
	python -m quant_syco analyze

# Run all notebooks in sequence
notebooks: notebook-01 notebook-02 notebook-03 notebook-04 notebook-05-robustness notebook-05-figures
	@echo "All notebooks executed successfully!"

# Individual notebook targets
notebook-01:
	@echo "Running 01_data_exploration..."
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/01_data_exploration.ipynb

notebook-02:
	@echo "Running 02_labeling_validation..."
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/02_labeling_validation.ipynb

notebook-03:
	@echo "Running 03_descriptive_analysis..."
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/03_descriptive_analysis.ipynb

notebook-04:
	@echo "Running 04_causal_modeling..."
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/04_causal_modeling.ipynb

notebook-05-robustness:
	@echo "Running 05_robustness_checks..."
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/05_robustness_checks.ipynb

notebook-05-figures:
	@echo "Running 05_figures_for_paper..."
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/05_figures_for_paper.ipynb

# Run a specific notebook by number (e.g., make notebook NB=03)
NB ?= 01
notebook:
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/*$(NB)*.ipynb

# Just generate figures (runs figures notebook only)
figures: notebook-05-figures
	@echo "Figures saved to paper/figures/"

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/*.parquet
	rm -rf data/labels/*.parquet
	rm -rf data/outputs/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Development helpers
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/
