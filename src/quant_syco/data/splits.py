"""Create reproducible train/validation splits for reliability testing."""

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import PROCESSED_DATA_DIR


def create_reliability_split(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a validation split for inter-rater reliability testing.

    Args:
        df: Full dataset.
        val_fraction: Fraction to hold out for validation (default 10%).
        random_state: Random seed for reproducibility.

    Returns:
        (train_df, val_df) tuple.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    prefix: str = "battles",
) -> None:
    """Save train/val splits to parquet."""
    train_path = PROCESSED_DATA_DIR / f"{prefix}_train.parquet"
    val_path = PROCESSED_DATA_DIR / f"{prefix}_val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Saved {len(train_df):,} train rows to {train_path}")
    print(f"Saved {len(val_df):,} val rows to {val_path}")


def load_splits(prefix: str = "battles") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/val splits."""
    train_path = PROCESSED_DATA_DIR / f"{prefix}_train.parquet"
    val_path = PROCESSED_DATA_DIR / f"{prefix}_val.parquet"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    return train_df, val_df
