from __future__ import annotations

from dataclasses import dataclass

from stock_prediction.utils.dependencies import require_dependency


@dataclass(slots=True)
class TimeSeriesSplit:
    train: "pd.DataFrame"
    validation: "pd.DataFrame"
    test: "pd.DataFrame"


@dataclass(slots=True)
class DateSplitBoundaries:
    train_end: "pd.Timestamp"
    validation_end: "pd.Timestamp"


@dataclass(slots=True)
class WindowedData:
    train_x: "np.ndarray"
    train_y: "np.ndarray"
    val_x: "np.ndarray"
    val_y: "np.ndarray"
    test_x: "np.ndarray"
    test_y: "np.ndarray"
    scaler: object


@dataclass(slots=True)
class MultiBranchWindowedData:
    train_inputs: dict[str, "np.ndarray"]
    val_inputs: dict[str, "np.ndarray"]
    test_inputs: dict[str, "np.ndarray"]
    train_y: "np.ndarray"
    val_y: "np.ndarray"
    test_y: "np.ndarray"
    scalers: dict[str, object]


def resolve_date_split_boundaries(frame, date_column: str, train_ratio: float, val_ratio: float) -> DateSplitBoundaries:
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    unique_dates = pd.Series(sorted(frame[date_column].dropna().unique().tolist()))
    if unique_dates.empty:
        raise ValueError("No dates available to build a date-based split.")
    train_index = max(int(len(unique_dates) * train_ratio) - 1, 0)
    validation_index = max(int(len(unique_dates) * (train_ratio + val_ratio)) - 1, train_index)
    return DateSplitBoundaries(
        train_end=pd.Timestamp(unique_dates.iloc[train_index]),
        validation_end=pd.Timestamp(unique_dates.iloc[min(validation_index, len(unique_dates) - 1)]),
    )


def time_ordered_split(frame, train_ratio: float, val_ratio: float) -> TimeSeriesSplit:
    total = len(frame)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    return TimeSeriesSplit(
        train=frame.iloc[:train_end].copy(),
        validation=frame.iloc[train_end:val_end].copy(),
        test=frame.iloc[val_end:].copy(),
    )


def date_ordered_split(frame, date_column: str, boundaries: DateSplitBoundaries) -> TimeSeriesSplit:
    train_mask = frame[date_column] <= boundaries.train_end
    validation_mask = (frame[date_column] > boundaries.train_end) & (frame[date_column] <= boundaries.validation_end)
    test_mask = frame[date_column] > boundaries.validation_end
    return TimeSeriesSplit(
        train=frame.loc[train_mask].copy(),
        validation=frame.loc[validation_mask].copy(),
        test=frame.loc[test_mask].copy(),
    )


def build_supervised_frame(
    frame,
    *,
    target_column: str,
    date_column: str,
    symbol_column: str,
    horizon: int,
    price_column: str = "close",
    extra_target_sources: dict[str, str] | None = None,
):
    extra_target_sources = extra_target_sources or {}
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    if frame.empty:
        empty = frame.copy()
        for column in ["feature_date", "current_close", target_column, "target_date", *extra_target_sources.keys()]:
            if column not in empty.columns:
                empty[column] = pd.Series(dtype="float64" if column != "feature_date" and column != "target_date" else "datetime64[ns]")
        return empty

    supervised = frame.copy()
    supervised["feature_date"] = supervised.get("feature_date", supervised[date_column])
    supervised["current_close"] = supervised[price_column]

    grouped = supervised.groupby(symbol_column, sort=False)
    supervised[target_column] = grouped[price_column].shift(-horizon)
    supervised["target_date"] = grouped[date_column].shift(-horizon)
    for target_name, source_column in extra_target_sources.items():
        supervised[target_name] = grouped[source_column].shift(-horizon)

    required = [target_column, "target_date", "current_close"]
    required.extend(extra_target_sources.keys())
    supervised = supervised.dropna(subset=required).reset_index(drop=True)

    if "feature_date" in supervised.columns:
        supervised["feature_date"] = pd.to_datetime(supervised["feature_date"], errors="coerce")
    supervised["target_date"] = pd.to_datetime(supervised["target_date"], errors="coerce")
    return supervised


def build_supervised_split(
    split: TimeSeriesSplit,
    *,
    target_column: str,
    date_column: str,
    symbol_column: str,
    horizon: int,
    price_column: str = "close",
    extra_target_sources: dict[str, str] | None = None,
) -> TimeSeriesSplit:
    return TimeSeriesSplit(
        train=build_supervised_frame(
            split.train,
            target_column=target_column,
            date_column=date_column,
            symbol_column=symbol_column,
            horizon=horizon,
            price_column=price_column,
            extra_target_sources=extra_target_sources,
        ),
        validation=build_supervised_frame(
            split.validation,
            target_column=target_column,
            date_column=date_column,
            symbol_column=symbol_column,
            horizon=horizon,
            price_column=price_column,
            extra_target_sources=extra_target_sources,
        ),
        test=build_supervised_frame(
            split.test,
            target_column=target_column,
            date_column=date_column,
            symbol_column=symbol_column,
            horizon=horizon,
            price_column=price_column,
            extra_target_sources=extra_target_sources,
        ),
    )


def create_sliding_windows(values, targets, window_size: int):
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    x, y = [], []
    if len(values) < window_size:
        return np.empty((0, window_size, values.shape[-1])), np.empty((0,))
    for start in range(0, len(values) - window_size + 1):
        end = start + window_size
        x.append(values[start:end])
        y.append(targets[end - 1])
    return np.asarray(x), np.asarray(y)


def create_contextual_windows(context_values, context_targets, values, targets, window_size: int):
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    if len(values) == 0:
        feature_width = values.shape[-1] if values.ndim == 2 else context_values.shape[-1]
        return np.empty((0, window_size, feature_width)), np.empty((0,))
    required_context = max(window_size - 1, 0)
    context_tail = context_values[-required_context:] if required_context else context_values[:0]
    target_tail = context_targets[-required_context:] if required_context else context_targets[:0]
    combined_values = np.concatenate([context_tail, values], axis=0)
    combined_targets = np.concatenate([target_tail, targets], axis=0)
    x, y = create_sliding_windows(combined_values, combined_targets, window_size)
    return x[-len(values) :], y[-len(values) :]


def build_single_sequence_input(history_values, current_value, window_size: int):
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    if len(history_values) < window_size - 1:
        raise ValueError("Insufficient history to build a sequence input window.")
    context = history_values[-(window_size - 1) :] if window_size > 1 else history_values[:0]
    current_row = np.asarray(current_value, dtype=float).reshape(1, -1)
    window = np.concatenate([context, current_row], axis=0)
    return window.reshape(1, window_size, -1)


def _winsorize_frame(frame, feature_columns: list[str], quantile_bounds: tuple[float, float]):
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    if frame.empty:
        return frame.copy(), {}
    lower_q, upper_q = quantile_bounds
    clipped = frame.copy()
    bounds = {}
    for column in feature_columns:
        values = clipped[column]
        lower = float(values.quantile(lower_q))
        upper = float(values.quantile(upper_q))
        if np.isnan(lower) or np.isnan(upper):
            continue
        clipped[column] = values.clip(lower=lower, upper=upper)
        bounds[column] = (lower, upper)
    return clipped, bounds


def _apply_winsorize(frame, feature_columns: list[str], bounds: dict[str, tuple[float, float]]):
    clipped = frame.copy()
    for column in feature_columns:
        if column not in clipped.columns:
            continue
        if column not in bounds:
            continue
        lower, upper = bounds[column]
        clipped[column] = clipped[column].clip(lower=lower, upper=upper)
    return clipped


def scale_and_window(
    split: TimeSeriesSplit,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    winsorize_limits: tuple[float, float] = (0.01, 0.99),
) -> WindowedData:
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    sklearn_preprocessing = require_dependency(
        "sklearn.preprocessing",
        "Run `uv sync` to install runtime dependencies.",
    )
    scaler = sklearn_preprocessing.StandardScaler()
    feature_count = len(feature_columns)
    train_frame, winsor_bounds = _winsorize_frame(split.train, feature_columns, winsorize_limits)
    validation_frame = _apply_winsorize(split.validation, feature_columns, winsor_bounds)
    test_frame = _apply_winsorize(split.test, feature_columns, winsor_bounds)
    fill_values = train_frame[feature_columns].median(numeric_only=True).fillna(0.0)

    def _transform(frame):
        if frame.empty:
            return np.empty((0, feature_count))
        return scaler.transform(frame[feature_columns].fillna(fill_values).to_numpy(dtype=float))

    train_values = scaler.fit_transform(train_frame[feature_columns].fillna(fill_values).to_numpy(dtype=float))
    val_values = _transform(validation_frame)
    test_values = _transform(test_frame)

    train_targets = train_frame[target_column].to_numpy(dtype=float)
    val_targets = validation_frame[target_column].to_numpy(dtype=float)
    test_targets = test_frame[target_column].to_numpy(dtype=float)

    train_x, train_y = create_sliding_windows(train_values, train_targets, window_size)
    val_x, val_y = create_contextual_windows(train_values, train_targets, val_values, val_targets, window_size)
    test_context_values = np.concatenate([train_values, val_values], axis=0)
    test_context_targets = np.concatenate([train_targets, val_targets], axis=0)
    test_x, test_y = create_contextual_windows(
        test_context_values,
        test_context_targets,
        test_values,
        test_targets,
        window_size,
    )
    return WindowedData(
        train_x=np.asarray(train_x, dtype=float),
        train_y=np.asarray(train_y, dtype=float),
        val_x=np.asarray(val_x, dtype=float),
        val_y=np.asarray(val_y, dtype=float),
        test_x=np.asarray(test_x, dtype=float),
        test_y=np.asarray(test_y, dtype=float),
        scaler=scaler,
    )


def scale_and_window_branches(
    split: TimeSeriesSplit,
    branch_feature_columns: dict[str, list[str]],
    target_column: str,
    window_size: int,
    winsorize_limits: tuple[float, float] = (0.01, 0.99),
) -> MultiBranchWindowedData:
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    sklearn_preprocessing = require_dependency(
        "sklearn.preprocessing",
        "Run `uv sync` to install runtime dependencies.",
    )

    train_targets = split.train[target_column].to_numpy(dtype=float)
    val_targets = split.validation[target_column].to_numpy(dtype=float)
    test_targets = split.test[target_column].to_numpy(dtype=float)

    dummy_train = np.zeros((len(train_targets), 1), dtype=float)
    dummy_val = np.zeros((len(val_targets), 1), dtype=float)
    dummy_test = np.zeros((len(test_targets), 1), dtype=float)
    _, train_y = create_sliding_windows(dummy_train, train_targets, window_size)
    val_x_dummy, val_y = create_contextual_windows(dummy_train, train_targets, dummy_val, val_targets, window_size)
    test_context_dummy = np.concatenate([dummy_train, dummy_val], axis=0)
    test_context_targets = np.concatenate([train_targets, val_targets], axis=0)
    test_x_dummy, test_y = create_contextual_windows(
        test_context_dummy,
        test_context_targets,
        dummy_test,
        test_targets,
        window_size,
    )

    del val_x_dummy, test_x_dummy

    train_inputs: dict[str, np.ndarray] = {}
    val_inputs: dict[str, np.ndarray] = {}
    test_inputs: dict[str, np.ndarray] = {}
    scalers: dict[str, object] = {}

    for branch_name, feature_columns in branch_feature_columns.items():
        if not feature_columns:
            continue
        scaler = sklearn_preprocessing.StandardScaler()
        train_frame, winsor_bounds = _winsorize_frame(split.train, feature_columns, winsorize_limits)
        validation_frame = _apply_winsorize(split.validation, feature_columns, winsor_bounds)
        test_frame = _apply_winsorize(split.test, feature_columns, winsor_bounds)
        fill_values = train_frame[feature_columns].median(numeric_only=True).fillna(0.0)
        train_values = scaler.fit_transform(train_frame[feature_columns].fillna(fill_values).to_numpy(dtype=float))

        def _transform(frame):
            if frame.empty:
                return np.empty((0, len(feature_columns)))
            return scaler.transform(frame[feature_columns].fillna(fill_values).to_numpy(dtype=float))

        val_values = _transform(validation_frame)
        test_values = _transform(test_frame)

        train_x, _ = create_sliding_windows(train_values, train_targets, window_size)
        val_x, _ = create_contextual_windows(train_values, train_targets, val_values, val_targets, window_size)
        test_context_values = np.concatenate([train_values, val_values], axis=0)
        test_x, _ = create_contextual_windows(
            test_context_values,
            test_context_targets,
            test_values,
            test_targets,
            window_size,
        )

        train_inputs[branch_name] = np.asarray(train_x, dtype=float)
        val_inputs[branch_name] = np.asarray(val_x, dtype=float)
        test_inputs[branch_name] = np.asarray(test_x, dtype=float)
        scalers[branch_name] = scaler

    return MultiBranchWindowedData(
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        test_inputs=test_inputs,
        train_y=np.asarray(train_y, dtype=float),
        val_y=np.asarray(val_y, dtype=float),
        test_y=np.asarray(test_y, dtype=float),
        scalers=scalers,
    )
