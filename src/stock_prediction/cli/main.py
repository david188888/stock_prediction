from __future__ import annotations

from pathlib import Path

import typer

from stock_prediction.config import load_config
from stock_prediction.data.downloader import download_dataset
from stock_prediction.data.preprocessing import prepare_dataset
from stock_prediction.pipelines.experiment import run_full_experiment, train_and_evaluate_model

app = typer.Typer(add_completion=False, help="Stock prediction experiment CLI.")


@app.command("download-data")
def download_data(config: str = typer.Option("configs/default.yaml", help="Path to config YAML.")) -> None:
    settings = load_config(config)
    extracted_dir = download_dataset(settings.dataset)
    typer.echo(f"Dataset extracted to {extracted_dir}")


@app.command("prepare-data")
def prepare_data(config: str = typer.Option("configs/default.yaml", help="Path to config YAML.")) -> None:
    settings = load_config(config)
    prepared = prepare_dataset(settings)
    typer.echo(f"Prepared dataset written to {prepared.prepared_path}")


@app.command("train")
def train_model(
    model: str = typer.Option(
        ...,
        help="Model name: linear_regression, linear_regression_scaled, arima, garch_return, lstm, gru, transformer, arima_residual_lstm",
    ),
    config: str = typer.Option("configs/default.yaml", help="Path to config YAML."),
) -> None:
    settings = load_config(config)
    artifacts = train_and_evaluate_model(settings, model)
    typer.echo(f"Metrics written to {artifacts.metrics_path}")
    typer.echo(f"Predictions written to {artifacts.predictions_path}")


@app.command("evaluate")
def evaluate_model(
    model: str = typer.Option(
        ...,
        help="Model name: linear_regression, linear_regression_scaled, arima, garch_return, lstm, gru, transformer, arima_residual_lstm",
    ),
    config: str = typer.Option("configs/default.yaml", help="Path to config YAML."),
) -> None:
    settings = load_config(config)
    artifacts = train_and_evaluate_model(settings, model)
    typer.echo(f"Summary written to {artifacts.summary_path}")


@app.command("run-experiment")
def run_experiment(config: str = typer.Option("configs/default.yaml", help="Path to config YAML.")) -> None:
    settings = load_config(config)
    artifacts = run_full_experiment(settings)
    typer.echo(f"Completed {len(artifacts)} experiment runs.")


if __name__ == "__main__":
    app()
