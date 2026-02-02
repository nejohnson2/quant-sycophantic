.PHONY: install download process label analyze figures test clean help

# Default model for labeling
MODEL ?= llama3.2:latest
BATCH_SIZE ?= 100

help:
	@echo "Quant-Sycophantic Research Pipeline"
	@echo ""
	@echo "Usage:"
	@echo "  make install        Install dependencies"
	@echo "  make download       Download LMSYS dataset"
	@echo "  make process        Process raw data into analysis-ready format"
	@echo "  make label          Run sycophancy labeling (MODEL=llama3.2:8b BATCH_SIZE=100)"
	@echo "  make analyze        Run descriptive and causal analysis"
	@echo "  make figures        Generate publication figures"
	@echo "  make test           Run tests"
	@echo "  make clean          Remove generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make label MODEL=mistral BATCH_SIZE=50"
	@echo "  make label-resume   Resume interrupted labeling"

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

analyze:
	python -m quant_syco analyze

figures:
	python -m quant_syco figures

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
