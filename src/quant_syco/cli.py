"""Command-line interface for quant-syco research pipeline."""

import typer
from rich.console import Console

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_OLLAMA_MODEL,
    LABELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer(
    name="quant-syco",
    help="Quantifying sycophancy impact on user engagement in LMSYS dataset",
    no_args_is_help=True,
)

console = Console()


@app.command()
def download(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
):
    """Download the LMSYS dataset."""
    from .data.download import download_lmsys

    console.print("[blue]Downloading LMSYS dataset...[/blue]")
    df = download_lmsys(force=force)
    console.print(f"[green]Downloaded {len(df):,} battles[/green]")


@app.command()
def process():
    """Process raw data into analysis-ready format."""
    from .data.process import compute_all_metrics, save_processed_data

    console.print("[blue]Processing raw data...[/blue]")
    metrics = compute_all_metrics()
    save_processed_data(metrics)
    console.print("[green]Processing complete![/green]")


@app.command()
def label(
    model: str = typer.Option(DEFAULT_OLLAMA_MODEL, "--model", "-m", help="Ollama model"),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, "--batch-size", "-b", help="Batch size"),
    side: str = typer.Option("both", "--side", "-s", help="Side to label: a, b, or both"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from checkpoint"),
    sample: int = typer.Option(None, "--sample", "-n", help="Label only N samples (for testing)"),
):
    """Run sycophancy labeling using LLM-as-judge."""
    from .data.process import build_battle_table
    from .labeling.batch import label_both_sides, run_batch_labeling

    console.print(f"[blue]Loading battles for labeling...[/blue]")
    battles = build_battle_table()

    if sample:
        console.print(f"[yellow]Sampling {sample} battles for testing[/yellow]")
        battles = battles.sample(n=min(sample, len(battles)), random_state=42)

    console.print(f"[blue]Labeling with model: {model}[/blue]")

    if side == "both":
        label_both_sides(battles, model=model, batch_size=batch_size, resume=resume)
    else:
        run_batch_labeling(battles, model=model, batch_size=batch_size, side=side, resume=resume)

    console.print("[green]Labeling complete![/green]")


@app.command()
def analyze():
    """Run descriptive and causal analysis."""
    import pandas as pd

    from .analysis.descriptive import (
        compute_correlations,
        compute_summary_statistics,
        compute_sycophancy_by_model,
        compute_win_rate_by_sycophancy,
    )
    from .data.process import build_battle_table
    from .features.topics import compute_topic_features

    console.print("[blue]Running analysis...[/blue]")

    # Load battles
    battles = build_battle_table()

    # Check for labels
    label_files = list(LABELS_DIR.glob("labels_*_merged.parquet"))
    if not label_files:
        console.print("[red]No labels found. Run 'quant-syco label' first.[/red]")
        raise typer.Exit(1)

    # Load and merge labels
    labels = pd.read_parquet(label_files[0])
    df = battles.merge(labels, on="question_id", how="left")

    # Add topics
    df = compute_topic_features(df, "user_prompt")

    console.print("\n[bold]Summary Statistics:[/bold]")
    summary = compute_summary_statistics(df)
    console.print(summary.to_string())

    console.print("\n[bold]Sycophancy by Model (Top 10):[/bold]")
    by_model = compute_sycophancy_by_model(df, "model_a", "sycophancy_a")
    console.print(by_model.head(10).to_string())

    console.print("\n[bold]Win Rate by Sycophancy Level:[/bold]")
    win_rate = compute_win_rate_by_sycophancy(df)
    console.print(win_rate.to_string())

    console.print("\n[bold]Correlations:[/bold]")
    x_cols = ["sycophancy_a", "politeness_a"]
    y_cols = ["returned_7d"] if "returned_7d" in df.columns else []
    if y_cols:
        corr = compute_correlations(df, x_cols, y_cols)
        console.print(corr.to_string())

    # Save results
    output_path = OUTPUTS_DIR / "analysis_summary.csv"
    summary.to_csv(output_path, index=False)
    console.print(f"\n[green]Results saved to {output_path}[/green]")


@app.command()
def figures():
    """Generate publication figures."""
    console.print("[blue]Generating figures...[/blue]")
    console.print("[yellow]Run notebooks/05_figures_for_paper.ipynb for full figures[/yellow]")


@app.command()
def validate():
    """Validate labeling with inter-rater reliability."""
    import pandas as pd

    from .analysis.reliability import compute_irr

    console.print("[blue]Computing inter-rater reliability...[/blue]")

    label_files = list(LABELS_DIR.glob("labels_*.parquet"))
    if len(label_files) < 2:
        console.print("[yellow]Need labels from 2+ models for IRR. Run labeling with different models.[/yellow]")
        raise typer.Exit(1)

    # Compare first two models
    labels1 = pd.read_parquet(label_files[0])
    labels2 = pd.read_parquet(label_files[1])

    console.print(f"Comparing: {label_files[0].name} vs {label_files[1].name}")

    for col in ["sycophancy_a", "sycophancy_b"]:
        if col in labels1.columns and col in labels2.columns:
            irr = compute_irr(labels1, labels2, score_col=col)
            console.print(f"\n[bold]{col}:[/bold]")
            console.print(f"  Weighted Kappa: {irr['kappa']:.3f} ({irr['interpretation']})")
            console.print(f"  95% CI: [{irr['ci_lower']:.3f}, {irr['ci_upper']:.3f}]")
            console.print(f"  Exact Agreement: {irr['exact_agreement']:.1%}")
            console.print(f"  Within-1 Agreement: {irr['within_1_agreement']:.1%}")


@app.command()
def status():
    """Show project status and data availability."""
    from .config import LABELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

    console.print("[bold]Project Status[/bold]\n")

    # Raw data
    raw_files = list(RAW_DATA_DIR.glob("*.parquet"))
    if raw_files:
        console.print(f"[green]Raw data: {len(raw_files)} file(s)[/green]")
        for f in raw_files:
            console.print(f"  - {f.name}")
    else:
        console.print("[yellow]Raw data: Not downloaded. Run 'quant-syco download'[/yellow]")

    # Processed data
    proc_files = list(PROCESSED_DATA_DIR.glob("*.parquet"))
    if proc_files:
        console.print(f"\n[green]Processed data: {len(proc_files)} file(s)[/green]")
        for f in proc_files:
            console.print(f"  - {f.name}")
    else:
        console.print("\n[yellow]Processed data: Not processed. Run 'quant-syco process'[/yellow]")

    # Labels
    label_files = list(LABELS_DIR.glob("*.parquet"))
    checkpoint_files = list(LABELS_DIR.glob("checkpoint_*.parquet"))
    if label_files:
        console.print(f"\n[green]Labels: {len(label_files)} file(s)[/green]")
        for f in label_files:
            if "checkpoint" not in f.name:
                console.print(f"  - {f.name}")
        if checkpoint_files:
            console.print(f"  ({len(checkpoint_files)} checkpoint(s) in progress)")
    else:
        console.print("\n[yellow]Labels: None. Run 'quant-syco label'[/yellow]")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
