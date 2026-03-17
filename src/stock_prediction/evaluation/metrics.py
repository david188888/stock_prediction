from __future__ import annotations

from stock_prediction.utils.dependencies import require_dependency


def calculate_metrics(y_true, y_pred, current_close=None) -> dict[str, float]:
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    da = float("nan")
    direction_count = 0
    if current_close is not None:
        baseline = np.asarray(current_close, dtype=float)
        actual_direction = np.sign(y_true - baseline)
        predicted_direction = np.sign(y_pred - baseline)
        mask = actual_direction != 0
        direction_count = int(mask.sum())
        if direction_count > 0:
            da = float(np.mean((predicted_direction[mask] == actual_direction[mask]).astype(float)))
    return {"mae": mae, "mse": mse, "rmse": rmse, "da": da, "direction_count": direction_count}
