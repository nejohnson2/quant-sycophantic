"""Centralized configuration for the quant-syco project."""

from pathlib import Path

# Project root (where pyproject.toml lives)
# config.py is at src/quant_syco/config.py, so 3 levels up to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, LABELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset configuration
LMSYS_DATASET = "lmsys/chatbot_arena_conversations"
LMSYS_SPLIT = "train"

# Labeling configuration
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_BATCH_SIZE = 100
CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N samples

# Return metric windows (days)
RETURN_WINDOWS = (1, 7, 30)

# Topic categories for domain analysis
TOPIC_CATEGORIES = [
    "coding",
    "creative_writing",
    "factual_qa",
    "advice",
    "opinion",
    "math",
    "other",
]

# Processed file names
BATTLES_FILE = "battles.parquet"
LABELS_FILE = "sycophancy_labels.parquet"
ANALYSIS_FILE = "analysis_results.parquet"
