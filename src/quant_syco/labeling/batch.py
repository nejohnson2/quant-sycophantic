"""Batch processing for sycophancy labeling with checkpoints."""

import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from ..config import CHECKPOINT_INTERVAL, DEFAULT_BATCH_SIZE, DEFAULT_OLLAMA_MODEL, LABELS_DIR
from .ollama_judge import OllamaJudge

console = Console()


def get_checkpoint_path(model: str, side: str = "a") -> Path:
    """Get checkpoint file path for a model/side combination."""
    model_safe = model.replace(":", "_").replace("/", "_")
    return LABELS_DIR / f"checkpoint_{model_safe}_{side}.parquet"


def get_labels_path(model: str, side: str = "a") -> Path:
    """Get final labels file path."""
    model_safe = model.replace(":", "_").replace("/", "_")
    return LABELS_DIR / f"labels_{model_safe}_{side}.parquet"


def load_checkpoint(model: str, side: str = "a") -> pd.DataFrame | None:
    """Load existing checkpoint if available."""
    path = get_checkpoint_path(model, side)
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_checkpoint(df: pd.DataFrame, model: str, side: str = "a") -> None:
    """Save checkpoint to disk."""
    path = get_checkpoint_path(model, side)
    df.to_parquet(path, index=False)


def run_batch_labeling(
    battles: pd.DataFrame,
    model: str = DEFAULT_OLLAMA_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    side: str = "a",
    resume: bool = False,
    rate_limit_delay: float = 0.1,
) -> pd.DataFrame:
    """
    Run sycophancy labeling on a batch of battles.

    Args:
        battles: DataFrame with user_prompt and assistant_{a,b} columns.
        model: Ollama model to use.
        batch_size: Number of samples to process before checkpoint.
        side: Which assistant response to label ("a" or "b").
        resume: If True, resume from checkpoint.
        rate_limit_delay: Delay between API calls (seconds).

    Returns:
        DataFrame with labels added.
    """
    response_col = f"assistant_{side}"
    if response_col not in battles.columns:
        raise ValueError(f"Column {response_col} not found in battles DataFrame")

    # Check for existing checkpoint
    if resume:
        checkpoint = load_checkpoint(model, side)
        if checkpoint is not None:
            console.print(f"[green]Resuming from checkpoint with {len(checkpoint):,} rows[/green]")
            # Find rows not yet processed
            processed_ids = set(checkpoint["question_id"])
            remaining = battles[~battles["question_id"].isin(processed_ids)]
            if len(remaining) == 0:
                console.print("[green]All rows already processed![/green]")
                return checkpoint
            console.print(f"[blue]Processing {len(remaining):,} remaining rows[/blue]")
        else:
            checkpoint = pd.DataFrame()
            remaining = battles
    else:
        checkpoint = pd.DataFrame()
        remaining = battles

    judge = OllamaJudge(model=model)

    results = []
    failed_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Labeling {side.upper()}...", total=len(remaining))

        for i, (idx, row) in enumerate(remaining.iterrows()):
            user_message = str(row.get("user_prompt", ""))
            assistant_response = str(row.get(response_col, ""))

            result = judge.label(user_message, assistant_response)

            results.append(
                {
                    "question_id": row["question_id"],
                    f"sycophancy_{side}": result.sycophancy,
                    f"politeness_{side}": result.politeness,
                    f"reasoning_{side}": result.reasoning,
                    f"label_success_{side}": result.success,
                    f"label_error_{side}": result.error,
                }
            )

            if not result.success:
                failed_count += 1

            progress.update(task, advance=1)

            # Save checkpoint periodically
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                batch_df = pd.DataFrame(results)
                combined = pd.concat([checkpoint, batch_df], ignore_index=True)
                save_checkpoint(combined, model, side)
                console.print(f"[green]Checkpoint saved: {len(combined):,} rows[/green]")

            # Rate limiting
            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)

    # Combine with checkpoint and save final
    batch_df = pd.DataFrame(results)
    final = pd.concat([checkpoint, batch_df], ignore_index=True)

    # Save final labels
    labels_path = get_labels_path(model, side)
    final.to_parquet(labels_path, index=False)

    console.print(f"\n[green]Labeling complete![/green]")
    console.print(f"  Total processed: {len(final):,}")
    console.print(f"  Failed: {failed_count:,}")
    console.print(f"  Saved to: {labels_path}")

    return final


def resume_labeling(
    battles: pd.DataFrame,
    model: str = DEFAULT_OLLAMA_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    side: str = "a",
) -> pd.DataFrame:
    """Resume labeling from checkpoint."""
    return run_batch_labeling(
        battles=battles,
        model=model,
        batch_size=batch_size,
        side=side,
        resume=True,
    )


def label_both_sides(
    battles: pd.DataFrame,
    model: str = DEFAULT_OLLAMA_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Label both assistant_a and assistant_b responses.

    Returns DataFrame with labels for both sides merged.
    """
    console.print("[blue]Labeling side A...[/blue]")
    labels_a = run_batch_labeling(battles, model, batch_size, "a", resume)

    console.print("\n[blue]Labeling side B...[/blue]")
    labels_b = run_batch_labeling(battles, model, batch_size, "b", resume)

    # Merge labels
    merged = labels_a.merge(labels_b, on="question_id", how="outer")

    # Save merged labels
    merged_path = LABELS_DIR / f"labels_{model.replace(':', '_')}_merged.parquet"
    merged.to_parquet(merged_path, index=False)
    console.print(f"\n[green]Merged labels saved to: {merged_path}[/green]")

    return merged


def merge_labels_with_battles(
    battles: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Merge labels back into battles DataFrame."""
    return battles.merge(labels, on="question_id", how="left")
