import pandas

from stock_prediction.features.selection import build_feature_catalog


def test_build_feature_catalog_separates_sentiment_from_baseline_groups():
    frame = pandas.DataFrame(
        {
            "date": pandas.date_range("2024-01-01", periods=6),
            "symbol": ["AAA"] * 6,
            "open": [1, 2, 3, 4, 5, 6],
            "high": [2, 3, 4, 5, 6, 7],
            "low": [0, 1, 2, 3, 4, 5],
            "close": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "volume": [10, 11, 12, 13, 14, 15],
            "feature_close_return_1": [0.0, 0.1, 0.1, 0.1, 0.1, 0.1],
            "feargreed": [20, 25, 30, 35, 40, 45],
        }
    )
    catalog = build_feature_catalog(
        frame,
        target_column="target_next_close",
        date_column="date",
        symbol_column="symbol",
        configured_groups={"sentiment_primary": ["feargreed"], "text_auxiliary": []},
    )

    assert catalog.sentiment_primary == ["feargreed"]
    assert "feargreed" not in catalog.returns_volatility
