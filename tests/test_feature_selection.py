import pandas

from stock_prediction.features.selection import build_feature_catalog


def test_build_feature_catalog_prioritizes_kaggle_price_and_technical_groups():
    frame = pandas.DataFrame(
        {
            "date": pandas.date_range("2024-01-01", periods=6),
            "symbol": ["AAA"] * 6,
            "age": [10] * 6,
            "open": [1, 2, 3, 4, 5, 6],
            "high": [2, 3, 4, 5, 6, 7],
            "low": [0, 1, 2, 3, 4, 5],
            "close": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "adjclose": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "volume": [10, 11, 12, 13, 14, 15],
            "feature_price_return_1": [0.0, 0.1, 0.1, 0.1, 0.1, 0.1],
            "feature_month": [1] * 6,
            "RSIadjclose15": [10, 20, 30, 40, 50, 60],
            "TARGET": [0, 1, 0, 1, 0, 1],
        }
    )
    catalog = build_feature_catalog(
        frame,
        target_column="target_next_adjclose",
        date_column="date",
        symbol_column="symbol",
        price_column="adjclose",
    )

    assert "adjclose" in catalog.raw_price_volume
    assert "feature_price_return_1" in catalog.price_returns_volatility
    assert "RSIadjclose15" in catalog.provided_technical_indicators
    assert "TARGET" not in catalog.all_numeric_filtered
    assert "age" in catalog.identity_meta
