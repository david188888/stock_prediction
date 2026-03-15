from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stock_prediction.config import AppConfig
from stock_prediction.evaluation.metrics import calculate_metrics
from stock_prediction.evaluation.reporting import (
    save_metrics,
    save_prediction_plot,
    save_predictions,
    save_summary_markdown,
)
from stock_prediction.features.windowing import scale_and_window, time_ordered_split
from stock_prediction.models.factory import create_model
from stock_prediction.utils.dependencies import require_dependency
from stock_prediction.utils.io import ensure_dir


@dataclass(slots=True)
class ExperimentArtifacts:
    metrics_path: Path
    predictions_path: Path
    summary_path: Path


def _load_prepared_frame(config: AppConfig):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    prepared_path = config.dataset.processed_dir / config.dataset.prepared_filename
    if not prepared_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at {prepared_path}. Run `prepare-data` first."
        )
    frame = pd.read_csv(prepared_path, parse_dates=[config.experiment.date_column])
    return frame


def _selected_symbols(config: AppConfig, frame) -> list[str]:
    available = sorted(frame[config.experiment.symbol_column].dropna().unique().tolist())
    if config.experiment.stock_symbols:
        return [symbol for symbol in available if symbol in config.experiment.stock_symbols]
    return available


def _tabular_xy(frame, feature_columns: list[str], target_column: str):
    X = frame[feature_columns].to_numpy()
    y = frame[target_column].to_numpy()
    return X, y


def _walk_forward_baseline(train_frame, test_frame, model_name: str, config: AppConfig):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    history = train_frame.copy()
    results = []
    for _, row in test_frame.head(config.experiment.walk_forward_steps).iterrows():
        model = create_model(
            model_name=model_name,
            model_config=config.models["linear_regression" if model_name == "linear_regression" else model_name],
            input_size=len(config.experiment.feature_columns),
            window_size=config.experiment.window_size,
        )
        if model_name == "arima":
            model.fit(None, history["target_next_close"].to_numpy())
            predicted = model.predict([0])[0]
        else:
            X_train, y_train = _tabular_xy(history, config.experiment.feature_columns, "target_next_close")
            model.fit(X_train, y_train)
            predicted = model.predict(row[config.experiment.feature_columns].to_numpy().reshape(1, -1))[0]
        enriched = row.to_dict()
        enriched["predicted"] = float(predicted)
        results.append(enriched)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    return pd.DataFrame(results)


def train_and_evaluate_model(config: AppConfig, model_name: str) -> ExperimentArtifacts:
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")

    frame = _load_prepared_frame(config)
    symbols = _selected_symbols(config, frame)
    if not symbols:
        raise ValueError("No stock symbols available after applying configuration filters.")

    metrics_rows = []
    prediction_frames = []

    output_root = ensure_dir(config.experiment.output_dir)
    model_dir = ensure_dir(output_root / "models" / model_name)

    for symbol in symbols:
        symbol_frame = frame[frame[config.experiment.symbol_column] == symbol].copy()
        if len(symbol_frame) < max(30, config.experiment.window_size + 5):
            continue
        split = time_ordered_split(
            symbol_frame,
            train_ratio=config.experiment.train_ratio,
            val_ratio=config.experiment.val_ratio,
        )

        if model_name in {"linear_regression"}:
            model = create_model(
                model_name=model_name,
                model_config=config.models[model_name],
                input_size=len(config.experiment.feature_columns),
                window_size=config.experiment.window_size,
            )
            train_xy = _tabular_xy(split.train, config.experiment.feature_columns, "target_next_close")
            test_xy = _tabular_xy(split.test, config.experiment.feature_columns, "target_next_close")
            model.fit(*train_xy)
            predictions = model.predict(test_xy[0])
            actual = test_xy[1]
        elif model_name == "arima":
            model = create_model(
                model_name=model_name,
                model_config=config.models[model_name],
                input_size=1,
                window_size=config.experiment.window_size,
            )
            series = split.train["target_next_close"].to_numpy()
            model.fit(None, series)
            predictions = np.asarray(model.predict(split.test))
            actual = split.test["target_next_close"].to_numpy()[: len(predictions)]
        elif model_name in {"lstm", "gru"}:
            windowed = scale_and_window(
                split=split,
                feature_columns=config.experiment.feature_columns,
                target_column="target_next_close",
                window_size=config.experiment.window_size,
            )
            model = create_model(
                model_name=model_name,
                model_config=config.models[model_name],
                input_size=len(config.experiment.feature_columns),
                window_size=config.experiment.window_size,
            )
            model.fit(windowed.train_x, windowed.train_y)
            predictions = model.predict(windowed.test_x)
            actual = windowed.test_y
        elif model_name == "arima_residual_lstm":
            model = create_model(
                model_name=model_name,
                model_config=config.models["hybrid"],
                input_size=1,
                window_size=config.experiment.window_size,
            )
            series = split.train["target_next_close"].to_numpy()
            model.fit(None, series)
            predictions = np.asarray(model.predict(split.test))
            actual = split.test["target_next_close"].to_numpy()[: len(predictions)]
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        metrics = calculate_metrics(actual, predictions)
        metrics_rows.append({"symbol": symbol, "model": model_name, **metrics})

        prediction_frame = split.test.head(len(predictions))[
            [config.experiment.date_column, config.experiment.symbol_column]
        ].copy()
        prediction_frame["actual"] = actual
        prediction_frame["predicted"] = predictions
        prediction_frame["model"] = model_name
        prediction_frames.append(prediction_frame)

        model_path = model_dir / f"{symbol}.bin"
        model.save(model_path)

        if config.experiment.generate_figures:
            save_prediction_plot(
                output_root / "figures" / f"{model_name}_{symbol}.png",
                prediction_frame.rename(columns={config.experiment.date_column: "date"}),
                title=f"{model_name} - {symbol}",
            )

    if not metrics_rows:
        raise RuntimeError("No models were trained. Check dataset size and configuration.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    walk_forward_records = []
    if model_name in {"linear_regression", "arima"}:
        for symbol in symbols:
            symbol_frame = frame[frame[config.experiment.symbol_column] == symbol].copy()
            split = time_ordered_split(
                symbol_frame,
                train_ratio=config.experiment.train_ratio,
                val_ratio=config.experiment.val_ratio,
            )
            walk_forward_df = _walk_forward_baseline(split.train, split.test, model_name, config)
            if not walk_forward_df.empty:
                walk_forward_metrics = calculate_metrics(
                    walk_forward_df["target_next_close"].to_numpy(),
                    walk_forward_df["predicted"].to_numpy(),
                )
                walk_forward_records.append({"symbol": symbol, "model": model_name, **walk_forward_metrics})

    metrics_path = output_root / "metrics" / f"{model_name}_metrics.csv"
    predictions_path = output_root / "predictions" / f"{model_name}_predictions.csv"
    summary_path = output_root / "metrics" / f"{model_name}_summary.md"

    save_metrics(metrics_path, metrics_df)
    save_predictions(predictions_path, predictions_df)

    summary_df = metrics_df.assign(evaluation="holdout").copy()
    if walk_forward_records:
        walk_forward_df = pd.DataFrame(walk_forward_records)
        walk_forward_df.to_csv(output_root / "metrics" / f"{model_name}_walk_forward.csv", index=False)
        summary_df = pd.concat([summary_df, walk_forward_df.assign(evaluation="walk_forward")], ignore_index=True)
    save_summary_markdown(summary_path, summary_df)
    return ExperimentArtifacts(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        summary_path=summary_path,
    )


def run_full_experiment(config: AppConfig, models: list[str] | None = None) -> list[ExperimentArtifacts]:
    selected_models = models or [
        "linear_regression",
        "arima",
        "lstm",
        "gru",
        "arima_residual_lstm",
    ]
    artifacts = [train_and_evaluate_model(config, model_name) for model_name in selected_models]

    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    all_metrics = []
    for artifact in artifacts:
        all_metrics.append(pd.read_csv(artifact.metrics_path))
    leaderboard = pd.concat(all_metrics, ignore_index=True).sort_values(["rmse", "mae"])
    leaderboard_path = config.experiment.output_dir / "metrics" / "leaderboard.csv"
    save_metrics(leaderboard_path, leaderboard)
    save_summary_markdown(config.experiment.output_dir / "metrics" / "leaderboard.md", leaderboard)
    return artifacts
