from __future__ import annotations

from dataclasses import dataclass

from stock_prediction.utils.dependencies import require_dependency


@dataclass(slots=True)
class TimeSeriesSplit:
    train: "pd.DataFrame"
    validation: "pd.DataFrame"
    test: "pd.DataFrame"


@dataclass(slots=True)
class WindowedData:
    train_x: "np.ndarray"
    train_y: "np.ndarray"
    val_x: "np.ndarray"
    val_y: "np.ndarray"
    test_x: "np.ndarray"
    test_y: "np.ndarray"
    scaler: object


def time_ordered_split(frame, train_ratio: float, val_ratio: float) -> TimeSeriesSplit:
    total = len(frame)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    return TimeSeriesSplit(
        train=frame.iloc[:train_end].copy(),
        validation=frame.iloc[train_end:val_end].copy(),
        test=frame.iloc[val_end:].copy(),
    )


def create_sliding_windows(values, targets, window_size: int):
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    x, y = [], []
    for start in range(0, len(values) - window_size):
        end = start + window_size
        x.append(values[start:end])
        y.append(targets[end])
    if not x:
        return np.empty((0, window_size, values.shape[-1])), np.empty((0,))
    return np.asarray(x), np.asarray(y)


def scale_and_window(
    split: TimeSeriesSplit,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
) -> WindowedData:
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    sklearn_preprocessing = require_dependency(
        "sklearn.preprocessing",
        "Run `uv sync` to install runtime dependencies.",
    )
    scaler = sklearn_preprocessing.StandardScaler()

    train_values = scaler.fit_transform(split.train[feature_columns].to_numpy())
    val_values = scaler.transform(split.validation[feature_columns].to_numpy())
    test_values = scaler.transform(split.test[feature_columns].to_numpy())

    train_targets = split.train[target_column].to_numpy()
    val_targets = split.validation[target_column].to_numpy()
    test_targets = split.test[target_column].to_numpy()

    train_x, train_y = create_sliding_windows(train_values, train_targets, window_size)
    val_x, val_y = create_sliding_windows(val_values, val_targets, window_size)
    test_x, test_y = create_sliding_windows(test_values, test_targets, window_size)
    return WindowedData(
        train_x=np.asarray(train_x, dtype=float),
        train_y=np.asarray(train_y, dtype=float),
        val_x=np.asarray(val_x, dtype=float),
        val_y=np.asarray(val_y, dtype=float),
        test_x=np.asarray(test_x, dtype=float),
        test_y=np.asarray(test_y, dtype=float),
        scaler=scaler,
    )

