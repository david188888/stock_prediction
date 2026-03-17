import pandas

from stock_prediction.config import load_config
from stock_prediction.features.windowing import date_ordered_split, resolve_date_split_boundaries
from stock_prediction.pipelines.experiment import _resolve_hybrid_feature_columns


def test_hybrid_feature_columns_include_residual_and_technical_inputs():
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
            "adjclose": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "volume": [10, 11, 12, 13, 14, 15],
            "feature_price_return_1": [0.0, 0.1, 0.1, 0.1, 0.1, 0.1],
            "RSIadjclose15": [20, 25, 30, 35, 40, 45],
            "arima_residual_current": [0.1, -0.2, 0.0, 0.3, -0.1, 0.2],
        }
    )

    feature_columns = _resolve_hybrid_feature_columns(frame, config)

    assert "arima_residual_current" in feature_columns
    assert "RSIadjclose15" in feature_columns
    assert "adjclose" in feature_columns


def test_date_split_uses_shared_calendar_boundaries():
    frame = pandas.DataFrame(
        {
            "date": pandas.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                ]
            )
        }
    )
    boundaries = resolve_date_split_boundaries(frame, "date", 0.5, 0.25)

    symbol_frame = pandas.DataFrame(
        {
            "date": pandas.date_range("2024-01-01", periods=6),
            "symbol": ["AAA"] * 6,
        }
    )
    split = date_ordered_split(symbol_frame, "date", boundaries)
    assert split.train["date"].max() == pandas.Timestamp("2024-01-03")
    assert split.validation["date"].max() == pandas.Timestamp("2024-01-04")
    assert split.test["date"].min() == pandas.Timestamp("2024-01-05")
