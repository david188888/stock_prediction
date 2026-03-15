from __future__ import annotations

from dataclasses import dataclass, asdict


ALIASES = {
    "symbol": ["symbol", "ticker", "stock", "name"],
    "date": ["date", "timestamp", "datetime", "time"],
    "open": ["open", "open_price"],
    "high": ["high", "high_price"],
    "low": ["low", "low_price"],
    "close": ["close", "adj close", "adj_close", "close_price", "price"],
    "volume": ["volume", "vol", "trading_volume"],
}


@dataclass(slots=True)
class DatasetSchema:
    source_file: str
    symbol_column: str | None
    date_column: str
    open_column: str
    high_column: str
    low_column: str
    close_column: str
    volume_column: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def infer_column_name(columns: list[str], logical_name: str) -> str:
    normalized = {_normalize(column): column for column in columns}
    for alias in ALIASES[logical_name]:
        if alias in normalized:
            return normalized[alias]
    raise ValueError(f"Unable to infer `{logical_name}` column from columns: {columns}")


def infer_optional_column_name(columns: list[str], logical_name: str) -> str | None:
    try:
        return infer_column_name(columns, logical_name)
    except ValueError:
        return None
