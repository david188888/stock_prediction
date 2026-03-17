from __future__ import annotations

import copy
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

from stock_prediction.config import AppConfig
from stock_prediction.evaluation.metrics import calculate_metrics
from stock_prediction.evaluation.reporting import (
    build_model_comparison_frame,
    save_conclusion_markdown,
    save_metrics,
    save_model_metric_bar_chart,
    save_predictions,
    save_summary_markdown,
    save_top_bottom_plot,
    save_training_log,
    save_walk_forward_error_plot,
)
from stock_prediction.features.selection import (
    resolve_feature_columns,
    resolve_feature_group_columns,
)
from stock_prediction.features.windowing import (
    TimeSeriesSplit,
    build_single_sequence_input,
    build_supervised_split,
    create_sliding_windows,
    scale_and_window,
    scale_and_window_branches,
    time_ordered_split,
)
from stock_prediction.models.factory import create_model
from stock_prediction.utils.dependencies import require_dependency
from stock_prediction.utils.io import ensure_dir


HYBRID_TARGET_RESIDUAL_COLUMN = "target_residual"
ARIMA_LINEAR_PRED_CURRENT_COLUMN = "arima_linear_pred_current"
ARIMA_LINEAR_PRED_NEXT_COLUMN = "arima_linear_pred_next"
ARIMA_RESIDUAL_CURRENT_COLUMN = "arima_residual_current"
PRIMARY_COMPARISON_MODELS = ["arima", "lstm", "arima_residual_lstm"]


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
    frame = pd.read_csv(prepared_path, parse_dates=[config.experiment.date_column])
    if "feature_date" in frame.columns:
        frame["feature_date"] = pd.to_datetime(frame["feature_date"], errors="coerce")
    additions = {}
    if "feature_date" not in frame.columns:
        additions["feature_date"] = frame[config.experiment.date_column]
    if "target_date" in frame.columns:
        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    else:
        additions["target_date"] = frame.groupby(config.experiment.symbol_column)[config.experiment.date_column].shift(
            -config.experiment.prediction_horizon
        )
    if additions:
        frame = frame.copy()
        frame = frame.assign(**additions)
    return frame


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


def _split_symbol_frame(symbol_frame, config: AppConfig) -> tuple[TimeSeriesSplit, TimeSeriesSplit]:
    raw_split = time_ordered_split(
        symbol_frame,
        train_ratio=config.experiment.train_ratio,
        val_ratio=config.experiment.val_ratio,
    )
    sample_split = build_supervised_split(
        raw_split,
        target_column=config.experiment.target_column,
        date_column=config.experiment.date_column,
        symbol_column=config.experiment.symbol_column,
        horizon=config.experiment.prediction_horizon,
    )
    return raw_split, sample_split


def _tabular_xy(frame, feature_columns: list[str], target_column: str):
    X = frame[feature_columns].to_numpy(dtype=float)
    y = frame[target_column].to_numpy(dtype=float)
    return X, y


def _resolve_feature_columns(frame, model_name: str, config: AppConfig) -> list[str]:
    return resolve_feature_columns(
        frame,
        model_name,
        explicit_columns=config.experiment.feature_columns,
        feature_set=config.experiment.feature_set,
        target_column=config.experiment.target_column,
        date_column=config.experiment.date_column,
        symbol_column=config.experiment.symbol_column,
        configured_groups=config.experiment.feature_groups,
    )


def _resolve_hybrid_branch_columns(frame, config: AppConfig) -> dict[str, list[str]]:
    price_basic = resolve_feature_group_columns(
        frame,
        group_name="price_basic",
        target_column=config.experiment.target_column,
        date_column=config.experiment.date_column,
        symbol_column=config.experiment.symbol_column,
        configured_groups=config.experiment.feature_groups,
    )
    returns_volatility = resolve_feature_group_columns(
        frame,
        group_name="returns_volatility",
        target_column=config.experiment.target_column,
        date_column=config.experiment.date_column,
        symbol_column=config.experiment.symbol_column,
        configured_groups=config.experiment.feature_groups,
    )
    sentiment_primary = resolve_feature_group_columns(
        frame,
        group_name="sentiment_primary",
        target_column=config.experiment.target_column,
        date_column=config.experiment.date_column,
        symbol_column=config.experiment.symbol_column,
        configured_groups=config.experiment.feature_groups,
    )
    text_auxiliary = resolve_feature_group_columns(
        frame,
        group_name="text_auxiliary",
        target_column=config.experiment.target_column,
        date_column=config.experiment.date_column,
        symbol_column=config.experiment.symbol_column,
        configured_groups=config.experiment.feature_groups,
    )
    if not sentiment_primary:
        raise ValueError("Hybrid model requires at least one sentiment feature.")

    price_columns = list(
        dict.fromkeys(
            [ARIMA_RESIDUAL_CURRENT_COLUMN]
            + [column for column in price_basic if column != ARIMA_RESIDUAL_CURRENT_COLUMN]
            + [column for column in returns_volatility if column != ARIMA_RESIDUAL_CURRENT_COLUMN]
        )
    )
    return {
        "price": price_columns,
        "sentiment": sentiment_primary,
        "text": text_auxiliary,
    }


def _feature_count(branch_columns: dict[str, list[str]]) -> int:
    return sum(len(columns) for columns in branch_columns.values())


def _hybrid_feature_set_label() -> str:
    return "weighted_sentiment_hybrid"


def _create_hybrid_model(
    config: AppConfig,
    order: tuple[int, int, int],
    branch_columns: dict[str, list[str]],
    *,
    epochs_override: int | None = None,
):
    hybrid_config = _model_config(config, "arima_residual_lstm", epochs_override=epochs_override)
    hybrid_config["arima_order"] = list(order)
    hybrid_config["fusion_weights"] = config.experiment.fusion_weights
    hybrid_config["price_input_size"] = len(branch_columns["price"])
    hybrid_config["sentiment_input_size"] = len(branch_columns["sentiment"])
    hybrid_config["text_input_size"] = len(branch_columns["text"])
    return create_model(
        model_name="arima_residual_lstm",
        model_config=hybrid_config,
        input_size=len(branch_columns["price"]),
        window_size=config.experiment.window_size,
    )


def _candidate_arima_orders(config: AppConfig) -> list[tuple[int, int, int]]:
    base_config = config.models["arima"]
    raw_orders = base_config.get("order_grid", [base_config["order"]])
    orders = [tuple(order) for order in raw_orders]
    default_order = tuple(base_config["order"])
    if default_order not in orders:
        orders.append(default_order)
    return orders


def _attach_arima_signals(frame, current_predictions):
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    enriched = frame.copy()
    enriched[ARIMA_LINEAR_PRED_CURRENT_COLUMN] = np.asarray(current_predictions, dtype=float)
    enriched[ARIMA_RESIDUAL_CURRENT_COLUMN] = (
        enriched["close"].to_numpy(dtype=float) - enriched[ARIMA_LINEAR_PRED_CURRENT_COLUMN].to_numpy(dtype=float)
    )
    return enriched


def _fit_arima_train_frame(frame, order: tuple[int, int, int], config: AppConfig):
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    model = create_model(
        model_name="arima",
        model_config={"order": list(order)},
        input_size=1,
        window_size=config.experiment.window_size,
    )
    series = frame["close"].to_numpy(dtype=float)
    model.fit(None, series)
    current_predictions = np.asarray(model.model_fit.predict(start=0, end=len(series) - 1), dtype=float)
    return model, _attach_arima_signals(frame, current_predictions)


def _forecast_arima_future_frame(history_frame, future_frame, order: tuple[int, int, int], config: AppConfig):
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    if future_frame.empty:
        return future_frame.copy()

    model = create_model(
        model_name="arima",
        model_config={"order": list(order)},
        input_size=1,
        window_size=config.experiment.window_size,
    )
    history_series = history_frame["close"].to_numpy(dtype=float)
    model.fit(None, history_series)
    state = model.model_fit
    running_history = history_series.copy()
    current_predictions = []
    for actual in future_frame["close"].to_numpy(dtype=float):
        forecast = np.asarray(state.forecast(steps=1), dtype=float)
        current_predictions.append(float(forecast[0]))
        if hasattr(state, "append"):
            state = state.append([actual], refit=False)
        else:
            running_history = np.concatenate([running_history, np.asarray([actual], dtype=float)])
            model.fit(None, running_history)
            state = model.model_fit
    return _attach_arima_signals(future_frame, current_predictions)


def _build_hybrid_sample_split(raw_split: TimeSeriesSplit, config: AppConfig) -> TimeSeriesSplit:
    return build_supervised_split(
        raw_split,
        target_column=config.experiment.target_column,
        date_column=config.experiment.date_column,
        symbol_column=config.experiment.symbol_column,
        horizon=config.experiment.prediction_horizon,
        extra_target_sources={
            HYBRID_TARGET_RESIDUAL_COLUMN: ARIMA_RESIDUAL_CURRENT_COLUMN,
            ARIMA_LINEAR_PRED_NEXT_COLUMN: ARIMA_LINEAR_PRED_CURRENT_COLUMN,
        },
    )


def _arima_validation_metrics(raw_split: TimeSeriesSplit, order: tuple[int, int, int], config: AppConfig):
    val_augmented = _forecast_arima_future_frame(raw_split.train, raw_split.validation, order, config)
    validation_split = TimeSeriesSplit(
        train=_empty_like(val_augmented),
        validation=val_augmented,
        test=_empty_like(val_augmented),
    )
    validation_samples = build_supervised_split(
        validation_split,
        target_column=config.experiment.target_column,
        date_column=config.experiment.date_column,
        symbol_column=config.experiment.symbol_column,
        horizon=config.experiment.prediction_horizon,
        extra_target_sources={ARIMA_LINEAR_PRED_NEXT_COLUMN: ARIMA_LINEAR_PRED_CURRENT_COLUMN},
    ).validation
    if validation_samples.empty:
        return calculate_metrics([], [])
    return calculate_metrics(
        validation_samples[config.experiment.target_column].to_numpy(dtype=float),
        validation_samples[ARIMA_LINEAR_PRED_NEXT_COLUMN].to_numpy(dtype=float),
        validation_samples["current_close"].to_numpy(dtype=float),
    )


def _select_arima_order(raw_split: TimeSeriesSplit, config: AppConfig):
    if not config.experiment.tune_arima_order or len(raw_split.validation) <= config.experiment.prediction_horizon:
        default_order = tuple(config.models["arima"]["order"])
        return default_order, [{"phase": "selection", "order": str(default_order), "selected": True}]

    best_order = None
    best_score = float("inf")
    records = []
    for order in _candidate_arima_orders(config):
        metrics = _arima_validation_metrics(raw_split, order, config)
        selected = metrics["rmse"] < best_score
        if selected:
            best_score = metrics["rmse"]
            best_order = order
        records.append({"phase": "selection", "order": str(order), **metrics, "selected": selected})
    return best_order or tuple(config.models["arima"]["order"]), records


def _build_prediction_frame(
    sample_frame,
    config: AppConfig,
    symbol: str,
    model_name: str,
    evaluation: str,
    predictions,
    *,
    feature_count: int,
    feature_set_label: str,
):
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    prediction_frame = sample_frame[["feature_date", "target_date", config.experiment.symbol_column]].copy()
    prediction_frame["date"] = sample_frame["target_date"]
    prediction_frame["current_close"] = sample_frame["current_close"].to_numpy(dtype=float)
    prediction_frame["actual_close"] = sample_frame[config.experiment.target_column].to_numpy(dtype=float)
    prediction_frame["predicted"] = np.asarray(predictions, dtype=float)
    prediction_frame["abs_error"] = (
        prediction_frame["actual_close"].to_numpy(dtype=float) - prediction_frame["predicted"].to_numpy(dtype=float)
    ).__abs__()
    predicted_direction = np.sign(
        prediction_frame["predicted"].to_numpy(dtype=float) - prediction_frame["current_close"].to_numpy(dtype=float)
    )
    actual_direction = np.sign(
        prediction_frame["actual_close"].to_numpy(dtype=float)
        - prediction_frame["current_close"].to_numpy(dtype=float)
    )
    direction_correct = []
    predicted_values = prediction_frame["predicted"].tolist()
    for predicted_raw, predicted_value, actual_value in zip(
        predicted_values,
        predicted_direction.tolist(),
        actual_direction.tolist(),
        strict=False,
    ):
        if pd.isna(predicted_raw) or actual_value == 0:
            direction_correct.append(pd.NA)
        else:
            direction_correct.append(bool(predicted_value == actual_value))
    prediction_frame["direction_correct"] = direction_correct
    prediction_frame["model"] = model_name
    prediction_frame["evaluation"] = evaluation
    prediction_frame["step"] = list(range(1, len(prediction_frame) + 1))
    prediction_frame["split_start_date"] = sample_frame["feature_date"].iloc[0]
    prediction_frame["feature_count"] = feature_count
    prediction_frame["feature_set"] = feature_set_label
    prediction_frame["symbol"] = symbol
    prediction_frame["prediction_horizon"] = config.experiment.prediction_horizon
    return prediction_frame


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
    training_history = [{"phase": "selection", **row} for row in model.get_training_history()]

    final_model = create_model(
        model_name=model_name,
        model_config=_model_config(config, model_name),
        input_size=len(feature_columns),
        window_size=config.experiment.window_size,
    )
    final_model.fit(*final_xy)
    predictions = final_model.predict(test_xy[0])
    return final_model, predictions, training_history


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
    return final_model, predictions, training_history, int(best_epoch)


def _fit_arima_holdout(raw_split: TimeSeriesSplit, config: AppConfig):
    order, selection_records = _select_arima_order(raw_split, config)
    combined_raw = _merge_frames([raw_split.train, raw_split.validation])
    final_model, _ = _fit_arima_train_frame(combined_raw, order, config)
    test_augmented = _forecast_arima_future_frame(combined_raw, raw_split.test, order, config)
    test_sample_split = build_supervised_split(
        TimeSeriesSplit(
            train=_empty_like(test_augmented),
            validation=_empty_like(test_augmented),
            test=test_augmented,
        ),
        target_column=config.experiment.target_column,
        date_column=config.experiment.date_column,
        symbol_column=config.experiment.symbol_column,
        horizon=config.experiment.prediction_horizon,
        extra_target_sources={ARIMA_LINEAR_PRED_NEXT_COLUMN: ARIMA_LINEAR_PRED_CURRENT_COLUMN},
    )
    predictions = test_sample_split.test[ARIMA_LINEAR_PRED_NEXT_COLUMN].to_numpy(dtype=float)
    return final_model, test_sample_split.test, predictions, order, selection_records


def _fit_hybrid_holdout(raw_split: TimeSeriesSplit, config: AppConfig):
    order, selection_records = _select_arima_order(raw_split, config)

    _, train_augmented = _fit_arima_train_frame(raw_split.train, order, config)
    validation_augmented = _forecast_arima_future_frame(raw_split.train, raw_split.validation, order, config)
    selection_split = _build_hybrid_sample_split(
        TimeSeriesSplit(
            train=train_augmented,
            validation=validation_augmented,
            test=_empty_like(validation_augmented),
        ),
        config,
    )
    branch_columns = _resolve_hybrid_branch_columns(selection_split.train, config)
    selection_windowed = scale_and_window_branches(
        selection_split,
        branch_feature_columns=branch_columns,
        target_column=HYBRID_TARGET_RESIDUAL_COLUMN,
        window_size=config.experiment.window_size,
    )

    model = _create_hybrid_model(config, order, branch_columns)
    validation_data = None
    if len(selection_windowed.val_y):
        validation_data = (selection_windowed.val_inputs, selection_windowed.val_y)
    model.fit(
        {"close_series": raw_split.train["close"].to_numpy(dtype=float), **selection_windowed.train_inputs},
        selection_windowed.train_y,
        validation_data=validation_data,
    )
    training_history = selection_records + [{"phase": "selection", **row} for row in model.get_training_history()]
    best_epoch = getattr(model.residual_model, "best_epoch", _model_config(config, "arima_residual_lstm")["epochs"])

    combined_raw = _merge_frames([raw_split.train, raw_split.validation])
    _, combined_augmented = _fit_arima_train_frame(combined_raw, order, config)
    test_augmented = _forecast_arima_future_frame(combined_raw, raw_split.test, order, config)
    final_split = _build_hybrid_sample_split(
        TimeSeriesSplit(
            train=combined_augmented,
            validation=_empty_like(combined_augmented),
            test=test_augmented,
        ),
        config,
    )
    final_branch_columns = _resolve_hybrid_branch_columns(final_split.train, config)
    final_windowed = scale_and_window_branches(
        final_split,
        branch_feature_columns=final_branch_columns,
        target_column=HYBRID_TARGET_RESIDUAL_COLUMN,
        window_size=config.experiment.window_size,
    )
    final_model = _create_hybrid_model(config, order, final_branch_columns, epochs_override=best_epoch)
    final_model.fit(
        {"close_series": combined_raw["close"].to_numpy(dtype=float), **final_windowed.train_inputs},
        final_windowed.train_y,
    )
    predictions = final_model.predict(
        {
            "linear_forecast": final_split.test[ARIMA_LINEAR_PRED_NEXT_COLUMN].to_numpy(dtype=float),
            **final_windowed.test_inputs,
        }
    )
    return final_model, final_split.test, predictions, order, training_history, int(best_epoch), final_branch_columns


def _walk_forward_sequence_prediction(history_frame, row_frame, model_name: str, feature_columns: list[str], config: AppConfig, epochs_override: int | None):
    sklearn_preprocessing = require_dependency(
        "sklearn.preprocessing",
        "Run `uv sync` to install runtime dependencies.",
    )
    scaler = sklearn_preprocessing.StandardScaler()
    fill_values = history_frame[feature_columns].median(numeric_only=True).fillna(0.0)
    history_values = scaler.fit_transform(history_frame[feature_columns].fillna(fill_values).to_numpy(dtype=float))
    current_value = scaler.transform(row_frame[feature_columns].fillna(fill_values).to_numpy(dtype=float))[0]
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


def _build_branch_inference_inputs(history_frame, current_row_frame, branch_columns: dict[str, list[str]], scalers: dict[str, object], window_size: int):
    inputs = {}
    for branch_name, feature_columns in branch_columns.items():
        if not feature_columns:
            continue
        scaler = scalers[branch_name]
        fill_values = history_frame[feature_columns].median(numeric_only=True).fillna(0.0)
        history_values = scaler.transform(history_frame[feature_columns].fillna(fill_values).to_numpy(dtype=float))
        current_value = scaler.transform(current_row_frame[feature_columns].fillna(fill_values).to_numpy(dtype=float))[0]
        inputs[branch_name] = build_single_sequence_input(history_values, current_value, window_size)
    return inputs


def _walk_forward_predictions(
    raw_split: TimeSeriesSplit,
    sample_split: TimeSeriesSplit,
    model_name: str,
    config: AppConfig,
    *,
    feature_columns: list[str] | None = None,
    branch_columns: dict[str, list[str]] | None = None,
    arima_order: tuple[int, int, int] | None = None,
    epochs_override: int | None = None,
):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    results = []
    test_samples = sample_split.test
    if test_samples.empty:
        return pd.DataFrame()

    max_steps = min(config.experiment.walk_forward_steps, len(test_samples))
    history_samples = _merge_frames([sample_split.train, sample_split.validation])

    if model_name == "arima_residual_lstm":
        if branch_columns is None or arima_order is None:
            raise ValueError("Hybrid walk-forward requires branch_columns and arima_order.")
        test_raw = raw_split.test.reset_index(drop=True)
        base_raw = _merge_frames([raw_split.train, raw_split.validation])
        for step_index in range(1, max_steps + 1):
            sample_row = test_samples.iloc[step_index - 1 : step_index].copy()
            observed_raw = _merge_frames([base_raw, test_raw.iloc[:step_index].copy()])
            observed_model, observed_augmented = _fit_arima_train_frame(observed_raw, arima_order, config)
            current_raw = observed_augmented.tail(1).copy()
            observed_samples = _build_hybrid_sample_split(
                TimeSeriesSplit(
                    train=observed_augmented,
                    validation=_empty_like(observed_augmented),
                    test=_empty_like(observed_augmented),
                ),
                config,
            ).train
            if observed_samples.empty:
                continue
            windowed = scale_and_window_branches(
                TimeSeriesSplit(
                    train=observed_samples,
                    validation=_empty_like(observed_samples),
                    test=_empty_like(observed_samples),
                ),
                branch_feature_columns=branch_columns,
                target_column=HYBRID_TARGET_RESIDUAL_COLUMN,
                window_size=config.experiment.window_size,
            )
            model = _create_hybrid_model(config, arima_order, branch_columns, epochs_override=epochs_override)
            model.fit(
                {"close_series": observed_raw["close"].to_numpy(dtype=float), **windowed.train_inputs},
                windowed.train_y,
            )
            linear_forecast = float(observed_model.predict(1)[0])
            branch_inputs = _build_branch_inference_inputs(
                observed_samples,
                current_raw,
                branch_columns,
                windowed.scalers,
                config.experiment.window_size,
            )
            predicted = float(model.predict({"linear_forecast": [linear_forecast], **branch_inputs})[0])
            frame = _build_prediction_frame(
                sample_row,
                config,
                symbol=str(sample_row[config.experiment.symbol_column].iloc[0]),
                model_name=model_name,
                evaluation="walk_forward",
                predictions=[predicted],
                feature_count=_feature_count(branch_columns),
                feature_set_label=_hybrid_feature_set_label(),
            )
            frame["step"] = step_index
            frame["split_start_date"] = test_samples["feature_date"].iloc[0]
            results.append(frame)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    for step_index in range(1, max_steps + 1):
        sample_row = test_samples.iloc[step_index - 1 : step_index].copy()
        if model_name in {"linear_regression", "linear_regression_scaled"}:
            if feature_columns is None:
                raise ValueError("Tabular walk-forward requires feature_columns.")
            model = create_model(
                model_name=model_name,
                model_config=_model_config(config, model_name),
                input_size=len(feature_columns),
                window_size=config.experiment.window_size,
            )
            train_x, train_y = _tabular_xy(history_samples, feature_columns, config.experiment.target_column)
            model.fit(train_x, train_y)
            predicted = float(model.predict(sample_row[feature_columns].to_numpy(dtype=float))[0])
            feature_count = len(feature_columns)
            feature_set_label = config.experiment.feature_set
        elif model_name == "arima":
            if arima_order is None:
                raise ValueError("ARIMA walk-forward requires arima_order.")
            observed_raw = _merge_frames([raw_split.train, raw_split.validation, raw_split.test.iloc[:step_index].copy()])
            model = create_model(
                model_name="arima",
                model_config={"order": list(arima_order)},
                input_size=1,
                window_size=config.experiment.window_size,
            )
            model.fit(None, observed_raw["close"].to_numpy(dtype=float))
            predicted = float(model.predict(1)[0])
            feature_count = 1
            feature_set_label = "close_only"
        elif model_name in {"lstm", "gru"}:
            if feature_columns is None:
                raise ValueError("Sequence walk-forward requires feature_columns.")
            predicted = _walk_forward_sequence_prediction(
                history_samples,
                sample_row,
                model_name,
                feature_columns,
                config,
                epochs_override,
            )
            feature_count = len(feature_columns)
            feature_set_label = config.experiment.feature_set
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        frame = _build_prediction_frame(
            sample_row,
            config,
            symbol=str(sample_row[config.experiment.symbol_column].iloc[0]),
            model_name=model_name,
            evaluation="walk_forward",
            predictions=[predicted],
            feature_count=feature_count,
            feature_set_label=feature_set_label,
        )
        frame["step"] = step_index
        frame["split_start_date"] = test_samples["feature_date"].iloc[0]
        results.append(frame)
        history_samples = _merge_frames([history_samples, sample_row])

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


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
        raw_split, sample_split = _split_symbol_frame(symbol_frame, config)
        if sample_split.train.empty or sample_split.test.empty:
            continue

        _set_random_seed(_seed_for_symbol(config, model_name, symbol))

        feature_columns = None
        branch_columns = None
        selected_order = None
        best_epoch = None
        feature_count = 0
        feature_set_label = config.experiment.feature_set

        if model_name in {"linear_regression", "linear_regression_scaled"}:
            feature_columns = _resolve_feature_columns(sample_split.train, model_name, config)
            feature_count = len(feature_columns)
            model, predictions, log_rows = _fit_linear_holdout(sample_split, model_name, feature_columns, config)
            test_samples = sample_split.test
        elif model_name == "arima":
            model, test_samples, predictions, selected_order, log_rows = _fit_arima_holdout(raw_split, config)
            feature_count = 1
            feature_set_label = "close_only"
        elif model_name in {"lstm", "gru"}:
            feature_columns = _resolve_feature_columns(sample_split.train, model_name, config)
            feature_count = len(feature_columns)
            model, predictions, log_rows, best_epoch = _fit_sequence_holdout(sample_split, model_name, feature_columns, config)
            test_samples = sample_split.test
        elif model_name == "arima_residual_lstm":
            model, test_samples, predictions, selected_order, log_rows, best_epoch, branch_columns = _fit_hybrid_holdout(raw_split, config)
            feature_count = _feature_count(branch_columns)
            feature_set_label = _hybrid_feature_set_label()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        current_close = test_samples["current_close"].to_numpy(dtype=float)
        actual = test_samples[config.experiment.target_column].to_numpy(dtype=float)
        metrics = calculate_metrics(actual, predictions, current_close)
        metrics_rows.append(
            {
                "symbol": symbol,
                "model": model_name,
                "evaluation": "holdout",
                "feature_count": feature_count,
                "feature_set": feature_set_label,
                **metrics,
            }
        )
        prediction_frames.append(
            _build_prediction_frame(
                test_samples,
                config,
                symbol,
                model_name,
                "holdout",
                predictions,
                feature_count=feature_count,
                feature_set_label=feature_set_label,
            )
        )
        for row in log_rows:
            training_logs.append({"symbol": symbol, "model": model_name, **row})

        model_path = model_dir / f"{symbol}.bin"
        model.save(model_path)

        if "walk_forward" in config.experiment.evaluation_modes and model_name in config.experiment.walk_forward_models:
            walk_forward_frame = _walk_forward_predictions(
                raw_split,
                sample_split,
                model_name,
                config,
                feature_columns=feature_columns,
                branch_columns=branch_columns,
                arima_order=selected_order,
                epochs_override=best_epoch,
            )
            if not walk_forward_frame.empty:
                walk_forward_metrics = calculate_metrics(
                    walk_forward_frame["actual_close"].to_numpy(dtype=float),
                    walk_forward_frame["predicted"].to_numpy(dtype=float),
                    walk_forward_frame["current_close"].to_numpy(dtype=float),
                )
                metrics_rows.append(
                    {
                        "symbol": symbol,
                        "model": model_name,
                        "evaluation": "walk_forward",
                        "feature_count": feature_count,
                        "feature_set": feature_set_label,
                        **walk_forward_metrics,
                    }
                )
                walk_forward_frames.append(walk_forward_frame)
                prediction_frames.append(walk_forward_frame)

    if not metrics_rows:
        raise RuntimeError("No models were trained. Check dataset size and configuration.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["evaluation", "rmse", "mae"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
        [config.experiment.symbol_column, "evaluation", "target_date", "feature_date"]
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
                walk_forward_df.rename(columns={"actual_close": "actual"}),
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

    parse_dates = ["date", "feature_date", "target_date", "split_start_date"]
    all_predictions = [pd.read_csv(artifact.predictions_path, parse_dates=parse_dates) for artifact in artifacts]
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    comparison_models = [model_name for model_name in PRIMARY_COMPARISON_MODELS if model_name in selected_models]
    for evaluation in sorted(combined_predictions["evaluation"].dropna().unique().tolist()):
        evaluation_frame = combined_predictions[
            (combined_predictions["evaluation"] == evaluation)
            & (combined_predictions["model"].isin(comparison_models))
        ].copy()
        if evaluation_frame.empty:
            continue
        comparison_frame = build_model_comparison_frame(
            evaluation_frame,
            model_order=comparison_models,
            extreme_volatility_quantile=config.experiment.extreme_volatility_quantile,
        )
        save_predictions(
            config.experiment.output_dir / "predictions" / f"model_comparison_{evaluation}.csv",
            comparison_frame,
        )

    return artifacts
