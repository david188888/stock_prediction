import pytest

numpy = pytest.importorskip("numpy")
pandas = pytest.importorskip("pandas")

from stock_prediction.features.windowing import create_sliding_windows, scale_and_window, time_ordered_split


def test_create_sliding_windows_shapes():
    values = numpy.arange(20, dtype=float).reshape(10, 2)
    targets = numpy.arange(10, dtype=float)
    x, y = create_sliding_windows(values, targets, window_size=3)
    assert x.shape == (7, 3, 2)
    assert y.shape == (7,)


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
