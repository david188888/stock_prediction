from __future__ import annotations

import copy
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

from stock_prediction.config import AppConfig
from stock_prediction.evaluation.metrics import calculate_metrics
from stock_prediction.evaluation.reporting import (
    save_conclusion_markdown,
    save_metrics,
    save_model_metric_bar_chart,
    save_predictions,
    save_summary_markdown,
    save_top_bottom_plot,
    save_training_log,
    save_walk_forward_error_plot,
)
from stock_prediction.features.selection import resolve_feature_columns
from stock_prediction.features.windowing import (
    TimeSeriesSplit,
    build_single_sequence_input,
    create_sliding_windows,
    scale_and_window,
    time_ordered_split,
)
from stock_prediction.models.factory import create_model
from stock_prediction.utils.dependencies import require_dependency
from stock_prediction.utils.io import ensure_dir


@dataclass(slots=True)
class ExperimentArtifacts:
    metrics_path: Path
    predictions_path: Path
    summary_path: Path
    walk_forward_path: Path | None = None
    training_log_path: Path | None = None
    conclusion_path: Path | None = None


def _load_prepared_frame(config: AppConfig):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    prepared_path = config.dataset.processed_dir / config.dataset.prepared_filename
    if not prepared_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at {prepared_path}. Run `prepare-data` first."
        )
    return pd.read_csv(prepared_path, parse_dates=[config.experiment.date_column])


def _selected_symbols(config: AppConfig, frame) -> list[str]:
    available = sorted(frame[config.experiment.symbol_column].dropna().unique().tolist())
    if config.experiment.stock_symbols:
        return [symbol for symbol in available if symbol in config.experiment.stock_symbols]
    return available


def _seed_for_symbol(config: AppConfig, model_name: str, symbol: str) -> int:
    return int(config.experiment.seed + zlib.adler32(f"{model_name}:{symbol}".encode("utf-8")))


def _set_random_seed(seed: int) -> None:
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch = require_dependency("torch", "Run `uv sync` to install runtime dependencies.")
    except RuntimeError:
        return
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tabular_xy(frame, feature_columns: list[str], target_column: str):
    X = frame[feature_columns].to_numpy(dtype=float)
    y = frame[target_column].to_numpy(dtype=float)
    return X, y


def _merge_frames(frames: list):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()
    return pd.concat(valid_frames, ignore_index=True)


def _empty_like(frame):
    return frame.iloc[0:0].copy()


def _model_config(config: AppConfig, model_name: str, *, epochs_override: int | None = None) -> dict:
    config_key = "hybrid" if model_name == "arima_residual_lstm" else model_name
    payload = copy.deepcopy(config.models[config_key])
    if model_name in {"lstm", "gru", "arima_residual_lstm"}:
        payload["early_stopping_patience"] = config.experiment.early_stopping_patience
        payload["min_delta"] = config.experiment.min_delta
        if epochs_override is not None:
            payload["epochs"] = max(int(epochs_override), 1)
    return payload


def _fit_linear_holdout(split: TimeSeriesSplit, model_name: str, feature_columns: list[str], config: AppConfig):
    train_xy = _tabular_xy(split.train, feature_columns, config.experiment.target_column)
    val_xy = _tabular_xy(split.validation, feature_columns, config.experiment.target_column)
    final_train = _merge_frames([split.train, split.validation])
    final_xy = _tabular_xy(final_train, feature_columns, config.experiment.target_column)
    test_xy = _tabular_xy(split.test, feature_columns, config.experiment.target_column)

    model = create_model(
        model_name=model_name,
        model_config=_model_config(config, model_name),
        input_size=len(feature_columns),
        window_size=config.experiment.window_size,
    )
    validation_data = val_xy if len(split.validation) else None
    model.fit(*train_xy, validation_data=validation_data)
    training_history = [
        {"phase": "selection", **row}
        for row in model.get_training_history()
    ]

    final_model = create_model(
        model_name=model_name,
        model_config=_model_config(config, model_name),
        input_size=len(feature_columns),
        window_size=config.experiment.window_size,
    )
    final_model.fit(*final_xy)
    predictions = final_model.predict(test_xy[0])
    return final_model, predictions, test_xy[1], training_history


def _candidate_arima_orders(config: AppConfig) -> list[tuple[int, int, int]]:
    base_config = config.models["arima"]
    raw_orders = base_config.get("order_grid", [base_config["order"]])
    orders = [tuple(order) for order in raw_orders]
    default_order = tuple(base_config["order"])
    if default_order not in orders:
        orders.append(default_order)
    return orders


def _select_arima_order(train_series, val_series, config: AppConfig):
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    if not config.experiment.tune_arima_order or len(val_series) == 0:
        default_order = tuple(config.models["arima"]["order"])
        return default_order, [{"phase": "selection", "order": str(default_order), "selected": True}]

    best_order = None
    best_score = float("inf")
    records = []
    for order in _candidate_arima_orders(config):
        model = create_model(
            model_name="arima",
            model_config={"order": list(order)},
            input_size=1,
            window_size=config.experiment.window_size,
        )
        model.fit(None, train_series)
        forecast = np.asarray(model.predict(len(val_series)), dtype=float)
        metrics = calculate_metrics(val_series, forecast)
        selected = metrics["rmse"] < best_score
        if selected:
            best_score = metrics["rmse"]
            best_order = order
        records.append({"phase": "selection", "order": str(order), **metrics, "selected": selected})

    return best_order or tuple(config.models["arima"]["order"]), records


def _fit_arima_holdout(split: TimeSeriesSplit, config: AppConfig):
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    train_series = split.train[config.experiment.target_column].to_numpy(dtype=float)
    val_series = split.validation[config.experiment.target_column].to_numpy(dtype=float)
    order, selection_records = _select_arima_order(train_series, val_series, config)
    final_series = _merge_frames([split.train, split.validation])[config.experiment.target_column].to_numpy(dtype=float)
    model = create_model(
        model_name="arima",
        model_config={"order": list(order)},
        input_size=1,
        window_size=config.experiment.window_size,
    )
    model.fit(None, final_series)
    predictions = np.asarray(model.predict(len(split.test)), dtype=float)
    actual = split.test[config.experiment.target_column].to_numpy(dtype=float)
    return model, predictions, actual, order, selection_records


def _sequence_refit_split(split: TimeSeriesSplit):
    combined_train = _merge_frames([split.train, split.validation])
    return TimeSeriesSplit(
        train=combined_train,
        validation=_empty_like(split.validation),
        test=split.test.copy(),
    )


def _fit_sequence_holdout(split: TimeSeriesSplit, model_name: str, feature_columns: list[str], config: AppConfig):
    initial_windowed = scale_and_window(
        split=split,
        feature_columns=feature_columns,
        target_column=config.experiment.target_column,
        window_size=config.experiment.window_size,
    )
    model = create_model(
        model_name=model_name,
        model_config=_model_config(config, model_name),
        input_size=len(feature_columns),
        window_size=config.experiment.window_size,
    )
    validation_data = None
    if len(initial_windowed.val_x):
        validation_data = (initial_windowed.val_x, initial_windowed.val_y)
    model.fit(initial_windowed.train_x, initial_windowed.train_y, validation_data=validation_data)
    training_history = [{"phase": "selection", **row} for row in model.get_training_history()]
    best_epoch = getattr(model, "best_epoch", _model_config(config, model_name)["epochs"])

    refit_split = _sequence_refit_split(split)
    final_windowed = scale_and_window(
        split=refit_split,
        feature_columns=feature_columns,
        target_column=config.experiment.target_column,
        window_size=config.experiment.window_size,
    )
    final_model = create_model(
        model_name=model_name,
        model_config=_model_config(config, model_name, epochs_override=best_epoch),
        input_size=len(feature_columns),
        window_size=config.experiment.window_size,
    )
    final_model.fit(final_windowed.train_x, final_windowed.train_y)
    predictions = final_model.predict(final_windowed.test_x)
    return final_model, predictions, final_windowed.test_y, training_history, int(best_epoch)


def _fit_hybrid_holdout(split: TimeSeriesSplit, config: AppConfig):
    train_series = split.train[config.experiment.target_column].to_numpy(dtype=float)
    val_series = split.validation[config.experiment.target_column].to_numpy(dtype=float)
    order, selection_records = _select_arima_order(train_series, val_series, config)
    model = create_model(
        model_name="arima_residual_lstm",
        model_config={**_model_config(config, "arima_residual_lstm"), "arima_order": list(order)},
        input_size=1,
        window_size=config.experiment.window_size,
    )
    validation_data = val_series if len(val_series) else None
    model.fit(None, train_series, validation_data=validation_data)
    training_history = selection_records + [{"phase": "selection", **row} for row in model.get_training_history()]
    best_epoch = getattr(model.residual_model, "best_epoch", _model_config(config, "arima_residual_lstm")["epochs"])

    final_series = _merge_frames([split.train, split.validation])[config.experiment.target_column].to_numpy(dtype=float)
    final_model = create_model(
        model_name="arima_residual_lstm",
        model_config={**_model_config(config, "arima_residual_lstm", epochs_override=best_epoch), "arima_order": list(order)},
        input_size=1,
        window_size=config.experiment.window_size,
    )
    final_model.fit(None, final_series)
    predictions = final_model.predict(len(split.test))
    actual = split.test[config.experiment.target_column].to_numpy(dtype=float)
    return final_model, predictions, actual, order, training_history, int(best_epoch)


def _build_prediction_frame(split_frame, config: AppConfig, symbol: str, model_name: str, evaluation: str, actual, predictions, feature_columns: list[str]):
    prediction_frame = split_frame[[config.experiment.date_column, config.experiment.symbol_column]].copy()
    prediction_frame["actual"] = actual
    prediction_frame["predicted"] = predictions
    prediction_frame["model"] = model_name
    prediction_frame["evaluation"] = evaluation
    prediction_frame["step"] = list(range(1, len(prediction_frame) + 1))
    prediction_frame["split_start_date"] = split_frame[config.experiment.date_column].iloc[0]
    prediction_frame["feature_count"] = len(feature_columns)
    prediction_frame["feature_set"] = config.experiment.feature_set
    prediction_frame["symbol"] = symbol
    return prediction_frame


def _walk_forward_sequence_prediction(history_frame, row, model_name: str, feature_columns: list[str], config: AppConfig, epochs_override: int | None):
    sklearn_preprocessing = require_dependency(
        "sklearn.preprocessing",
        "Run `uv sync` to install runtime dependencies.",
    )
    scaler = sklearn_preprocessing.StandardScaler()
    history_values = scaler.fit_transform(history_frame[feature_columns].to_numpy(dtype=float))
    current_value = scaler.transform(row[feature_columns].to_numpy(dtype=float).reshape(1, -1))[0]
    train_targets = history_frame[config.experiment.target_column].to_numpy(dtype=float)
    train_x, train_y = create_sliding_windows(history_values, train_targets, config.experiment.window_size)
    if len(train_x) == 0:
        raise ValueError("Insufficient history for sequence walk-forward evaluation.")
    model = create_model(
        model_name=model_name,
        model_config=_model_config(config, model_name, epochs_override=epochs_override),
        input_size=len(feature_columns),
        window_size=config.experiment.window_size,
    )
    model.fit(train_x, train_y)
    inference_x = build_single_sequence_input(history_values, current_value, config.experiment.window_size)
    return float(model.predict(inference_x)[0])


def _walk_forward_predictions(
    split: TimeSeriesSplit,
    model_name: str,
    feature_columns: list[str],
    config: AppConfig,
    *,
    arima_order: tuple[int, int, int] | None = None,
    epochs_override: int | None = None,
):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    history = _merge_frames([split.train, split.validation])
    results = []
    test_head = split.test.head(config.experiment.walk_forward_steps)
    for step_index, (_, row) in enumerate(test_head.iterrows(), start=1):
        row_frame = pd.DataFrame([row])
        if model_name in {"linear_regression", "linear_regression_scaled"}:
            model = create_model(
                model_name=model_name,
                model_config=_model_config(config, model_name),
                input_size=len(feature_columns),
                window_size=config.experiment.window_size,
            )
            train_x, train_y = _tabular_xy(history, feature_columns, config.experiment.target_column)
            model.fit(train_x, train_y)
            predicted = float(model.predict(row[feature_columns].to_numpy(dtype=float).reshape(1, -1))[0])
        elif model_name == "arima":
            model = create_model(
                model_name="arima",
                model_config={"order": list(arima_order or tuple(config.models["arima"]["order"]))},
                input_size=1,
                window_size=config.experiment.window_size,
            )
            model.fit(None, history[config.experiment.target_column].to_numpy(dtype=float))
            predicted = float(model.predict(1)[0])
        elif model_name in {"lstm", "gru"}:
            predicted = _walk_forward_sequence_prediction(history, row, model_name, feature_columns, config, epochs_override)
        elif model_name == "arima_residual_lstm":
            model = create_model(
                model_name=model_name,
                model_config={**_model_config(config, model_name, epochs_override=epochs_override), "arima_order": list(arima_order or tuple(config.models["arima"]["order"]))},
                input_size=1,
                window_size=config.experiment.window_size,
            )
            model.fit(None, history[config.experiment.target_column].to_numpy(dtype=float))
            predicted = float(model.predict(1)[0])
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        enriched = row_frame[[config.experiment.date_column, config.experiment.symbol_column]].copy()
        enriched["actual"] = float(row[config.experiment.target_column])
        enriched["predicted"] = predicted
        enriched["model"] = model_name
        enriched["evaluation"] = "walk_forward"
        enriched["step"] = step_index
        enriched["split_start_date"] = split.test[config.experiment.date_column].iloc[0]
        enriched["feature_count"] = len(feature_columns)
        enriched["feature_set"] = config.experiment.feature_set
        results.append(enriched)
        history = _merge_frames([history, row_frame])

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def train_and_evaluate_model(config: AppConfig, model_name: str) -> ExperimentArtifacts:
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")

    frame = _load_prepared_frame(config)
    symbols = _selected_symbols(config, frame)
    if not symbols:
        raise ValueError("No stock symbols available after applying configuration filters.")

    metrics_rows = []
    prediction_frames = []
    training_logs = []
    walk_forward_frames = []

    output_root = ensure_dir(config.experiment.output_dir)
    model_dir = ensure_dir(output_root / "models" / model_name)

    for symbol in symbols:
        symbol_frame = frame[frame[config.experiment.symbol_column] == symbol].copy()
        if len(symbol_frame) < max(30, config.experiment.window_size + 5):
            continue
        feature_columns = resolve_feature_columns(
            symbol_frame,
            model_name,
            explicit_columns=config.experiment.feature_columns,
            feature_set=config.experiment.feature_set,
            target_column=config.experiment.target_column,
            date_column=config.experiment.date_column,
            symbol_column=config.experiment.symbol_column,
        )
        split = time_ordered_split(
            symbol_frame,
            train_ratio=config.experiment.train_ratio,
            val_ratio=config.experiment.val_ratio,
        )
        _set_random_seed(_seed_for_symbol(config, model_name, symbol))

        selected_order = None
        best_epoch = None
        if model_name in {"linear_regression", "linear_regression_scaled"}:
            model, predictions, actual, log_rows = _fit_linear_holdout(split, model_name, feature_columns, config)
        elif model_name == "arima":
            model, predictions, actual, selected_order, log_rows = _fit_arima_holdout(split, config)
        elif model_name in {"lstm", "gru"}:
            model, predictions, actual, log_rows, best_epoch = _fit_sequence_holdout(split, model_name, feature_columns, config)
        elif model_name == "arima_residual_lstm":
            model, predictions, actual, selected_order, log_rows, best_epoch = _fit_hybrid_holdout(split, config)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        metrics = calculate_metrics(actual, predictions)
        metrics_rows.append(
            {
                "symbol": symbol,
                "model": model_name,
                "evaluation": "holdout",
                "feature_count": len(feature_columns),
                "feature_set": config.experiment.feature_set,
                **metrics,
            }
        )
        prediction_frames.append(
            _build_prediction_frame(split.test, config, symbol, model_name, "holdout", actual, predictions, feature_columns)
        )
        for row in log_rows:
            training_logs.append({"symbol": symbol, "model": model_name, **row})

        model_path = model_dir / f"{symbol}.bin"
        model.save(model_path)

        if "walk_forward" in config.experiment.evaluation_modes and model_name in config.experiment.walk_forward_models:
            walk_forward_frame = _walk_forward_predictions(
                split,
                model_name,
                feature_columns,
                config,
                arima_order=selected_order,
                epochs_override=best_epoch,
            )
            if not walk_forward_frame.empty:
                walk_forward_metrics = calculate_metrics(
                    walk_forward_frame["actual"].to_numpy(dtype=float),
                    walk_forward_frame["predicted"].to_numpy(dtype=float),
                )
                metrics_rows.append(
                    {
                        "symbol": symbol,
                        "model": model_name,
                        "evaluation": "walk_forward",
                        "feature_count": len(feature_columns),
                        "feature_set": config.experiment.feature_set,
                        **walk_forward_metrics,
                    }
                )
                walk_forward_frames.append(walk_forward_frame)
                prediction_frames.append(walk_forward_frame)

    if not metrics_rows:
        raise RuntimeError("No models were trained. Check dataset size and configuration.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["evaluation", "rmse", "mae"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
        [config.experiment.symbol_column, "evaluation", config.experiment.date_column]
    )
    training_df = pd.DataFrame(training_logs)
    walk_forward_df = pd.concat(walk_forward_frames, ignore_index=True) if walk_forward_frames else pd.DataFrame()

    metrics_path = output_root / "metrics" / f"{model_name}_metrics.csv"
    predictions_path = output_root / "predictions" / f"{model_name}_predictions.csv"
    summary_path = output_root / "metrics" / f"{model_name}_summary.md"
    training_log_path = output_root / "metrics" / f"{model_name}_training_log.csv"
    walk_forward_path = output_root / "metrics" / f"{model_name}_walk_forward.csv"
    conclusion_path = output_root / "metrics" / f"{model_name}_conclusion.md"

    save_metrics(metrics_path, metrics_df)
    save_predictions(predictions_path, predictions_df)
    save_summary_markdown(summary_path, metrics_df)
    if not training_df.empty:
        save_training_log(training_log_path, training_df)
    else:
        training_log_path = None
    if not walk_forward_df.empty:
        save_predictions(walk_forward_path, walk_forward_df)
    else:
        walk_forward_path = None
    save_conclusion_markdown(conclusion_path, metrics_df)

    if config.experiment.generate_figures:
        save_model_metric_bar_chart(output_root / "figures" / f"{model_name}_overview.png", metrics_df, title=f"{model_name} metrics")
        save_top_bottom_plot(output_root / "figures" / f"{model_name}_top_bottom.png", metrics_df, title=f"{model_name} best/worst symbols")
        if not walk_forward_df.empty:
            save_walk_forward_error_plot(
                output_root / "figures" / f"{model_name}_walk_forward_errors.png",
                walk_forward_df,
                title=f"{model_name} walk-forward absolute error",
            )

    return ExperimentArtifacts(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        summary_path=summary_path,
        walk_forward_path=walk_forward_path,
        training_log_path=training_log_path,
        conclusion_path=conclusion_path,
    )


def run_full_experiment(config: AppConfig, models: list[str] | None = None) -> list[ExperimentArtifacts]:
    selected_models = models or [
        "linear_regression",
        "linear_regression_scaled",
        "arima",
        "lstm",
        "gru",
        "arima_residual_lstm",
    ]
    artifacts = [train_and_evaluate_model(config, model_name) for model_name in selected_models]

    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    all_metrics = [pd.read_csv(artifact.metrics_path) for artifact in artifacts]
    leaderboard = pd.concat(all_metrics, ignore_index=True).sort_values(["evaluation", "rmse", "mae"])
    leaderboard_path = config.experiment.output_dir / "metrics" / "leaderboard.csv"
    leaderboard_summary_path = config.experiment.output_dir / "metrics" / "leaderboard.md"
    conclusions_path = config.experiment.output_dir / "metrics" / "conclusions.md"
    save_metrics(leaderboard_path, leaderboard)
    save_summary_markdown(leaderboard_summary_path, leaderboard)
    save_conclusion_markdown(conclusions_path, leaderboard)
    return artifacts
