from __future__ import annotations

from dataclasses import dataclass, asdict


ALIASES = {
    "symbol": ["symbol", "ticker", "company", "stock", "name"],
    "date": ["date", "timestamp", "datetime", "time"],
    "year": ["year"],
    "month": ["month"],
    "day": ["day"],
    "hour": ["hour"],
    "minute": ["minute"],
    "open": ["open", "open_price"],
    "high": ["high", "high_price"],
    "low": ["low", "low_price"],
    "close": ["close", "adj close", "adj_close", "close_price", "price"],
    "adjclose": ["adjclose", "adj_close", "adjusted_close", "adjusted close", "adj close"],
    "volume": ["volume", "vol", "trading_volume"],
    "market": ["market", "exchange"],
    "age": ["age"],
    "target": ["target", "TARGET", "label", "y"],
}


@dataclass(slots=True)
class DatasetSchema:
    source_file: str
    symbol_column: str | None
    date_column: str | None
    year_column: str | None
    month_column: str | None
    day_column: str | None
    hour_column: str | None
    minute_column: str | None
    open_column: str
    high_column: str
    low_column: str
    close_column: str
    adjclose_column: str | None
    volume_column: str
    market_column: str | None
    age_column: str | None
    target_column: str | None

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def infer_column_name(columns: list[str], logical_name: str) -> str:
    normalized = {_normalize(column): column for column in columns}
    for alias in ALIASES[logical_name]:
        normalized_alias = _normalize(alias)
        if normalized_alias in normalized:
            return normalized[normalized_alias]
    raise ValueError(f"Unable to infer `{logical_name}` column from columns: {columns}")


def infer_optional_column_name(columns: list[str], logical_name: str) -> str | None:
    try:
        return infer_column_name(columns, logical_name)
    except ValueError:
        return None
