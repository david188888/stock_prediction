import pytest

numpy = pytest.importorskip("numpy")
pandas = pytest.importorskip("pandas")

from stock_prediction.features.windowing import create_sliding_windows, scale_and_window, time_ordered_split


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
            "volume": range(30),
            "target_next_close": range(30),
        }
    )
    split = time_ordered_split(frame, 0.6, 0.2)
    windowed = scale_and_window(split, ["open", "high", "low", "close", "volume"], "target_next_close", 5)
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
            "volume": range(20),
            "target_next_close": range(100, 120),
        }
    )
    split = time_ordered_split(frame, 0.5, 0.25)
    windowed = scale_and_window(split, ["open", "high", "low", "close", "volume"], "target_next_close", 4)
    numpy.testing.assert_array_equal(windowed.val_y, split.validation["target_next_close"].to_numpy())
    numpy.testing.assert_array_equal(windowed.test_y, split.test["target_next_close"].to_numpy())
