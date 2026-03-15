from __future__ import annotations

from stock_prediction.utils.dependencies import require_dependency


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    np = require_dependency("numpy", "Run `uv sync` to install runtime dependencies.")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse}

