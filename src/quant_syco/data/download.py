"""Download and cache the LMSYS dataset."""

import pandas as pd
from datasets import load_dataset
from rich.console import Console

from ..config import LMSYS_DATASET, LMSYS_SPLIT, RAW_DATA_DIR

console = Console()


def download_lmsys(force: bool = False) -> pd.DataFrame:
    """
    Download the LMSYS chatbot arena conversations dataset.

    Args:
        force: If True, re-download even if cached locally.

    Returns:
        DataFrame with the raw dataset.
    """
    cache_path = RAW_DATA_DIR / "lmsys_raw.parquet"

    if cache_path.exists() and not force:
        console.print(f"[green]Loading cached data from {cache_path}[/green]")
        return pd.read_parquet(cache_path)

    console.print(f"[blue]Downloading {LMSYS_DATASET}...[/blue]")

    ds = load_dataset(LMSYS_DATASET)
    split = LMSYS_SPLIT if LMSYS_SPLIT in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()

    console.print(f"[green]Downloaded {len(df):,} rows[/green]")
    console.print(f"[blue]Columns: {list(df.columns)}[/blue]")

    # Cache locally
    df.to_parquet(cache_path, index=False)
    console.print(f"[green]Cached to {cache_path}[/green]")

    return df


def load_raw_data() -> pd.DataFrame:
    """Load the raw LMSYS data (downloading if needed)."""
    return download_lmsys(force=False)
