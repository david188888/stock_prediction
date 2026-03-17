import pytest

numpy = pytest.importorskip("numpy")
pandas = pytest.importorskip("pandas")

from stock_prediction.features.windowing import (
    build_supervised_split,
    create_sliding_windows,
    date_ordered_split,
    resolve_date_split_boundaries,
    scale_and_window,
    time_ordered_split,
)


def test_create_sliding_windows_shapes():
    values = numpy.arange(20, dtype=float).reshape(10, 2)
    targets = numpy.arange(10, dtype=float)
    x, y = create_sliding_windows(values, targets, window_size=3)
    assert x.shape == (8, 3, 2)
    assert y.shape == (8,)
    assert y[0] == 2


def test_scale_and_window_uses_train_only_scaler():
    frame = pandas.DataFrame(
        {
            "open": range(30),
            "high": range(30),
            "low": range(30),
            "close": range(30),
            "adjclose": range(30),
            "volume": range(30),
            "target_next_adjclose": range(30),
        }
    )
    split = time_ordered_split(frame, 0.6, 0.2)
    windowed = scale_and_window(split, ["open", "high", "low", "adjclose", "volume"], "target_next_adjclose", 5)
    assert windowed.train_x.shape[0] > 0
    assert windowed.test_x.shape[1] == 5
    assert windowed.val_x.shape[0] == len(split.validation)
    assert windowed.test_x.shape[0] == len(split.test)


def test_scale_and_window_preserves_split_targets_with_context():
    frame = pandas.DataFrame(
        {
            "open": range(20),
            "high": range(20),
            "low": range(20),
            "close": range(20),
            "adjclose": range(20),
            "volume": range(20),
            "target_next_adjclose": range(100, 120),
        }
    )
    split = time_ordered_split(frame, 0.5, 0.25)
    windowed = scale_and_window(split, ["open", "high", "low", "adjclose", "volume"], "target_next_adjclose", 4)
    numpy.testing.assert_array_equal(windowed.val_y, split.validation["target_next_adjclose"].to_numpy())
    numpy.testing.assert_array_equal(windowed.test_y, split.test["target_next_adjclose"].to_numpy())


def test_build_supervised_split_prevents_cross_boundary_target_leakage():
    frame = pandas.DataFrame(
        {
            "date": pandas.date_range("2024-01-01", periods=12),
            "symbol": ["AAA"] * 12,
            "close": numpy.arange(12, dtype=float),
            "adjclose": numpy.arange(12, dtype=float),
            "open": numpy.arange(12, dtype=float),
            "high": numpy.arange(12, dtype=float),
            "low": numpy.arange(12, dtype=float),
            "volume": numpy.arange(12, dtype=float),
        }
    )
    split = time_ordered_split(frame, 0.5, 0.25)
    supervised = build_supervised_split(
        split,
        target_column="target_next_adjclose",
        date_column="date",
        symbol_column="symbol",
        horizon=1,
        price_column="adjclose",
    )

    assert supervised.train["target_date"].max() < split.validation["date"].min()
    assert supervised.validation["target_date"].max() < split.test["date"].min()
    assert supervised.test["target_date"].max() <= split.test["date"].max()


def test_date_ordered_split_uses_global_boundaries():
    frame = pandas.DataFrame({"date": pandas.date_range("2024-01-01", periods=10)})
    boundaries = resolve_date_split_boundaries(frame, "date", 0.6, 0.2)
    split = date_ordered_split(frame, "date", boundaries)
    assert split.train["date"].max() == pandas.Timestamp("2024-01-06")
    assert split.validation["date"].max() == pandas.Timestamp("2024-01-08")
    assert split.test["date"].min() == pandas.Timestamp("2024-01-09")
