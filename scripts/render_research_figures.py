from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_prediction.evaluation.reporting import save_conclusion_markdown, save_summary_markdown
from stock_prediction.utils.io import ensure_dir


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _load_csvs(paths: list[Path], *, parse_dates: list[str] | None = None):
    import pandas as pd

    frames = []
    for path in paths:
        if parse_dates:
            available_columns = pd.read_csv(path, nrows=0).columns.tolist()
            selected_parse_dates = [column for column in parse_dates if column in available_columns]
        else:
            selected_parse_dates = None
        frames.append(pd.read_csv(path, parse_dates=selected_parse_dates))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _metrics_files(output_root: Path) -> list[Path]:
    metrics_dir = output_root / "metrics"
    return sorted(
        path
        for path in metrics_dir.glob("*_metrics.csv")
        if path.name != "leaderboard.csv"
    )


def _prediction_files(output_root: Path) -> list[Path]:
    prediction_dir = output_root / "predictions"
    return sorted(
        path
        for path in prediction_dir.glob("*_predictions.csv")
        if path.name != "combined_predictions.csv"
    )


def _training_log_files(output_root: Path) -> list[Path]:
    metrics_dir = output_root / "metrics"
    return sorted(path for path in metrics_dir.glob("*_training_log.csv"))


def aggregate_outputs(output_root: Path):
    import pandas as pd

    metric_files = _metrics_files(output_root)
    if not metric_files:
        raise FileNotFoundError(f"No per-run metrics files found under {output_root / 'metrics'}")

    leaderboard = _load_csvs(metric_files).sort_values(["experiment_key", "evaluation", "rmse", "mae"]).reset_index(drop=True)
    ensure_dir(output_root / "metrics")
    leaderboard_path = output_root / "metrics" / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    save_summary_markdown(output_root / "metrics" / "leaderboard.md", leaderboard)
    save_conclusion_markdown(output_root / "metrics" / "conclusions.md", leaderboard)

    prediction_files = _prediction_files(output_root)
    parse_dates = ["date", "feature_date", "target_date", "split_start_date"]
    combined_predictions = _load_csvs(prediction_files, parse_dates=parse_dates)
    if not combined_predictions.empty:
        combined_predictions = combined_predictions.sort_values(
            ["experiment_key", "evaluation", "symbol", "target_date", "feature_date"]
        ).reset_index(drop=True)
        combined_predictions.to_csv(output_root / "predictions" / "combined_predictions.csv", index=False)

    training_logs = _load_csvs(_training_log_files(output_root))
    if not training_logs.empty:
        training_logs.to_csv(output_root / "metrics" / "combined_training_log.csv", index=False)

    return leaderboard, combined_predictions, training_logs


def _aggregate_metric_grid(frame, value_column: str):
    import pandas as pd

    if frame.empty or value_column not in frame.columns:
        return pd.DataFrame()
    aggregated = (
        frame.groupby(["evaluation", "experiment_key", "model"], dropna=False)[value_column]
        .mean()
        .reset_index()
    )
    return aggregated


def plot_rmse_heatmap(leaderboard, figures_dir: Path) -> None:
    import numpy as np

    plt = _plt()
    aggregated = _aggregate_metric_grid(leaderboard, "rmse")
    if aggregated.empty:
        return

    evaluations = sorted(aggregated["evaluation"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(evaluations), figsize=(7 * len(evaluations), 5), squeeze=False)
    for axis, evaluation in zip(axes[0], evaluations, strict=False):
        subset = aggregated[aggregated["evaluation"] == evaluation]
        pivot = subset.pivot(index="model", columns="experiment_key", values="rmse").sort_index()
        values = pivot.to_numpy(dtype=float)
        image = axis.imshow(values, aspect="auto", cmap="YlGnBu")
        axis.set_title(f"{evaluation} mean RMSE")
        axis.set_xlabel("experiment_key")
        axis.set_ylabel("model")
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels(pivot.index)
        for row_index in range(values.shape[0]):
            for col_index in range(values.shape[1]):
                value = values[row_index, col_index]
                if math.isnan(value):
                    continue
                axis.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Model Comparison Across Experiment Settings", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure01_rmse_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_directional_metrics(leaderboard, figures_dir: Path) -> None:
    import numpy as np

    plt = _plt()
    direction_columns = [column for column in ["da", "up_f1", "down_f1"] if column in leaderboard.columns]
    if not direction_columns:
        return

    aggregated = (
        leaderboard.groupby(["evaluation", "model"], dropna=False)[direction_columns]
        .mean()
        .reset_index()
    )
    evaluations = sorted(aggregated["evaluation"].dropna().unique().tolist())
    fig, axes = plt.subplots(len(evaluations), 1, figsize=(12, 4.5 * max(len(evaluations), 1)), squeeze=False)
    metric_labels = {"da": "DA", "up_f1": "Up F1", "down_f1": "Down F1"}
    colors = {"da": "#264653", "up_f1": "#2a9d8f", "down_f1": "#e76f51"}

    for axis, evaluation in zip(axes[:, 0], evaluations, strict=False):
        subset = aggregated[aggregated["evaluation"] == evaluation].sort_values("model")
        x = np.arange(len(subset))
        width = 0.25
        for index, column in enumerate(direction_columns):
            offset = (index - (len(direction_columns) - 1) / 2) * width
            axis.bar(x + offset, subset[column].to_numpy(dtype=float), width=width, label=metric_labels[column], color=colors[column])
        axis.set_title(f"{evaluation} directional metrics")
        axis.set_ylabel("score")
        axis.set_xlabel("model")
        axis.set_xticks(x)
        axis.set_xticklabels(subset["model"], rotation=35, ha="right")
        axis.set_ylim(0, 1.05)
        axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "figure02_directional_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(training_logs, figures_dir: Path) -> None:
    import pandas as pd

    plt = _plt()
    if training_logs.empty or "epoch" not in training_logs.columns:
        return

    training_logs = training_logs.copy()
    training_logs["epoch"] = pd.to_numeric(training_logs["epoch"], errors="coerce")
    training_logs["train_loss"] = pd.to_numeric(training_logs.get("train_loss"), errors="coerce")
    training_logs["val_loss"] = pd.to_numeric(training_logs.get("val_loss"), errors="coerce")
    training_logs = training_logs.dropna(subset=["epoch", "train_loss"])
    training_logs = training_logs[training_logs["phase"] == "selection"]
    training_logs = training_logs[training_logs["model"].isin(["lstm", "gru", "transformer", "arima_residual_lstm"])]
    if training_logs.empty:
        return

    models = sorted(training_logs["model"].dropna().unique().tolist())
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3.8 * len(models)), squeeze=False)
    for axis, model_name in zip(axes[:, 0], models, strict=False):
        subset = training_logs[training_logs["model"] == model_name]
        train_curve = subset.groupby("epoch", dropna=False)["train_loss"].agg(["mean", "std"]).reset_index()
        axis.plot(train_curve["epoch"], train_curve["mean"], color="#1d3557", linewidth=2, label="train_loss")
        train_std = train_curve["std"].fillna(0.0)
        axis.fill_between(
            train_curve["epoch"],
            train_curve["mean"] - train_std,
            train_curve["mean"] + train_std,
            color="#1d3557",
            alpha=0.15,
        )

        val_subset = subset.dropna(subset=["val_loss"])
        if not val_subset.empty:
            val_curve = val_subset.groupby("epoch", dropna=False)["val_loss"].agg(["mean", "std"]).reset_index()
            axis.plot(val_curve["epoch"], val_curve["mean"], color="#d62828", linewidth=2, label="val_loss")
            val_std = val_curve["std"].fillna(0.0)
            axis.fill_between(
                val_curve["epoch"],
                val_curve["mean"] - val_std,
                val_curve["mean"] + val_std,
                color="#d62828",
                alpha=0.15,
            )
        axis.set_title(f"{model_name} training curve")
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "figure03_training_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_prediction_examples(leaderboard, predictions, figures_dir: Path) -> None:
    import pandas as pd

    plt = _plt()
    if leaderboard.empty or predictions.empty:
        return

    holdout = leaderboard[leaderboard["evaluation"] == "holdout"].copy()
    if holdout.empty:
        holdout = leaderboard.copy()

    best_models = (
        holdout.groupby(["experiment_key", "model"], dropna=False)["rmse"]
        .mean()
        .reset_index()
        .sort_values(["experiment_key", "rmse"])
        .groupby("experiment_key", dropna=False)
        .head(1)
    )
    if best_models.empty:
        return

    selections = []
    for _, row in best_models.head(4).iterrows():
        experiment_key = row["experiment_key"]
        model_name = row["model"]
        best_symbol_row = holdout[
            (holdout["experiment_key"] == experiment_key) & (holdout["model"] == model_name)
        ].sort_values("rmse").head(1)
        if best_symbol_row.empty:
            continue
        selections.append(
            {
                "experiment_key": experiment_key,
                "model": model_name,
                "symbol": best_symbol_row.iloc[0]["symbol"],
            }
        )
    if not selections:
        return

    fig, axes = plt.subplots(len(selections), 1, figsize=(12, 3.8 * len(selections)), squeeze=False)
    for axis, selection in zip(axes[:, 0], selections, strict=False):
        subset = predictions[
            (predictions["experiment_key"] == selection["experiment_key"])
            & (predictions["model"] == selection["model"])
            & (predictions["symbol"] == selection["symbol"])
            & (predictions["evaluation"] == "holdout")
        ].copy()
        if subset.empty:
            subset = predictions[
                (predictions["experiment_key"] == selection["experiment_key"])
                & (predictions["model"] == selection["model"])
                & (predictions["symbol"] == selection["symbol"])
            ].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("target_date")
        axis.plot(subset["target_date"], subset["actual_close"], color="#1d3557", linewidth=2, label="actual")
        axis.plot(subset["target_date"], subset["predicted"], color="#e76f51", linewidth=2, linestyle="--", label="predicted")
        axis.set_title(f"{selection['experiment_key']} | {selection['model']} | {selection['symbol']}")
        axis.set_xlabel("target date")
        axis.set_ylabel("price")
        axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "figure04_prediction_examples.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate overnight training artifacts and render research-style figures.")
    parser.add_argument("--output-root", required=True, help="Output directory used by the overnight training script.")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    figures_dir = ensure_dir(output_root / "figures" / "research")

    leaderboard, predictions, training_logs = aggregate_outputs(output_root)
    plot_rmse_heatmap(leaderboard, figures_dir)
    plot_directional_metrics(leaderboard, figures_dir)
    plot_training_curves(training_logs, figures_dir)
    plot_prediction_examples(leaderboard, predictions, figures_dir)


if __name__ == "__main__":
    main()
