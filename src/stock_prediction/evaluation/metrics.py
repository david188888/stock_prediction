from __future__ import annotations

from stock_prediction.utils.dependencies import require_dependency


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _direction_classification_metrics(actual_direction, predicted_direction, positive_label: float) -> dict[str, float]:
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    actual_positive = actual_direction == positive_label
    predicted_positive = predicted_direction == positive_label
    true_positive = float(np.sum(actual_positive & predicted_positive))
    false_positive = float(np.sum(~actual_positive & predicted_positive))
    false_negative = float(np.sum(actual_positive & ~predicted_positive))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    if precision != precision or recall != recall or precision + recall == 0:
        f1 = float("nan")
    else:
        f1 = float(2 * precision * recall / (precision + recall))
    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_metrics(y_true, y_pred, current_close=None) -> dict[str, float]:
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean((2.0 * np.abs(y_true - y_pred)) / np.maximum(denominator, 1e-8)))
    da = float("nan")
    direction_count = 0
    up_precision = float("nan")
    up_recall = float("nan")
    up_f1 = float("nan")
    down_precision = float("nan")
    down_recall = float("nan")
    down_f1 = float("nan")
    if current_close is not None:
        baseline = np.asarray(current_close, dtype=float)
        actual_direction = np.sign(y_true - baseline)
        predicted_direction = np.sign(y_pred - baseline)
        mask = actual_direction != 0
        direction_count = int(mask.sum())
        if direction_count > 0:
            actual_direction = actual_direction[mask]
            predicted_direction = predicted_direction[mask]
            da = float(np.mean((predicted_direction == actual_direction).astype(float)))
            up_metrics = _direction_classification_metrics(actual_direction, predicted_direction, positive_label=1.0)
            down_metrics = _direction_classification_metrics(actual_direction, predicted_direction, positive_label=-1.0)
            up_precision = up_metrics["precision"]
            up_recall = up_metrics["recall"]
            up_f1 = up_metrics["f1"]
            down_precision = down_metrics["precision"]
            down_recall = down_metrics["recall"]
            down_f1 = down_metrics["f1"]
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "smape": smape,
        "da": da,
        "direction_count": direction_count,
        "up_precision": up_precision,
        "up_recall": up_recall,
        "up_f1": up_f1,
        "down_precision": down_precision,
        "down_recall": down_recall,
        "down_f1": down_f1,
    }
