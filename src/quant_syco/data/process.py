"""Process raw LMSYS data into analysis-ready format."""

import pandas as pd
from rich.console import Console

from ..config import BATTLES_FILE, PROCESSED_DATA_DIR, RETURN_WINDOWS
from ..features.metrics import (
    add_ts_utc,
    build_battle_level_table,
    compute_return_metrics,
    compute_user_summary,
)
from .download import load_raw_data

console = Console()


def build_battle_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build the main battle-level analysis table.

    Each row represents one A/B battle with:
    - User prompt
    - Assistant A and B responses
    - Winner
    - Model identifiers
    - Timestamps and user ID

    Args:
        df: Raw dataframe. If None, loads from cache.

    Returns:
        Battle-level DataFrame ready for labeling.
    """
    if df is None:
        df = load_raw_data()

    console.print(f"[blue]Processing {len(df):,} battles...[/blue]")

    # Add UTC timestamps
    df = add_ts_utc(df, "tstamp")

    # Build battle table with extracted prompts/responses
    battles = build_battle_level_table(df)

    console.print(f"[green]Built battle table with {len(battles):,} rows[/green]")
    console.print(f"[blue]Columns: {list(battles.columns)}[/blue]")

    return battles


def compute_all_metrics(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """
    Compute all engagement metrics from raw data.

    Returns dict with:
    - battles: Battle-level table
    - returns: Battle table with return metrics
    - user_summary: User-level aggregates
    """
    if df is None:
        df = load_raw_data()

    # Add timestamps
    df = add_ts_utc(df, "tstamp")

    # Build battle table
    battles = build_battle_level_table(df)

    # Compute return metrics
    returns = compute_return_metrics(
        df,
        user_col="judge",
        time_col="ts_utc",
        id_col="question_id",
        windows_days=RETURN_WINDOWS,
    )

    # User-level summary
    user_summary = compute_user_summary(
        returns,
        user_col="judge",
        time_col="ts_utc",
        id_col="question_id",
        windows_days=RETURN_WINDOWS[1:],  # 7d and 30d
    )

    console.print(f"[green]Computed metrics:[/green]")
    console.print(f"  - Battles: {len(battles):,} rows")
    console.print(f"  - Returns: {len(returns):,} rows")
    console.print(f"  - Users: {len(user_summary):,} unique users")

    return {
        "battles": battles,
        "returns": returns,
        "user_summary": user_summary,
    }


def save_processed_data(metrics: dict[str, pd.DataFrame]) -> None:
    """Save processed data to parquet files."""
    for name, df in metrics.items():
        path = PROCESSED_DATA_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        console.print(f"[green]Saved {path}[/green]")


def load_processed_battles() -> pd.DataFrame:
    """Load processed battles table."""
    path = PROCESSED_DATA_DIR / BATTLES_FILE
    if not path.exists():
        raise FileNotFoundError(f"Processed battles not found at {path}. Run 'make process' first.")
    return pd.read_parquet(path)


def load_processed_returns() -> pd.DataFrame:
    """Load processed returns table."""
    path = PROCESSED_DATA_DIR / "returns.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed returns not found at {path}. Run 'make process' first.")
    return pd.read_parquet(path)
