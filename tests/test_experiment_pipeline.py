import pandas

from stock_prediction.config import load_config
from stock_prediction.pipelines.experiment import _resolve_hybrid_branch_columns


def test_hybrid_branch_columns_include_residual_and_sentiment():
    config = load_config("configs/default.yaml")
    frame = pandas.DataFrame(
        {
            "date": pandas.date_range("2024-01-01", periods=6),
            "feature_date": pandas.date_range("2024-01-01", periods=6),
            "target_date": pandas.date_range("2024-01-02", periods=6),
            "symbol": ["AAA"] * 6,
            "open": [1, 2, 3, 4, 5, 6],
            "high": [2, 3, 4, 5, 6, 7],
            "low": [0, 1, 2, 3, 4, 5],
            "close": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "volume": [10, 11, 12, 13, 14, 15],
            "feature_close_return_1": [0.0, 0.1, 0.1, 0.1, 0.1, 0.1],
            "feargreed": [20, 25, 30, 35, 40, 45],
            "arima_residual_current": [0.1, -0.2, 0.0, 0.3, -0.1, 0.2],
        }
    )

    branch_columns = _resolve_hybrid_branch_columns(frame, config)

    assert "arima_residual_current" in branch_columns["price"]
    assert len(branch_columns["price"]) > 1
    assert branch_columns["sentiment"] == ["feargreed"]
