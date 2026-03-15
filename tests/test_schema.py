import pytest

from stock_prediction.data.schema import infer_column_name


def test_infer_column_name_finds_standard_alias():
    columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
    assert infer_column_name(columns, "symbol") == "Ticker"
    assert infer_column_name(columns, "date") == "Date"


def test_infer_column_name_raises_for_missing_column():
    with pytest.raises(ValueError):
        infer_column_name(["Date", "Close"], "volume")

