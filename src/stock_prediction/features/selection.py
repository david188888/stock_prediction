from __future__ import annotations

from dataclasses import dataclass

from stock_prediction.utils.dependencies import require_dependency


TECHNICAL_TOKENS = (
    "rsi",
    "macd",
    "stochastic",
    "ema",
    "sma",
    "adl",
    "mfm",
    "mfv",
    "lag",
    "hammer",
    "supernova",
    "vela",
    "fuerzarelativa",
    "volumenrelativo",
    "incremento",
    "diff",
)

LEAKAGE_TOKENS = (
    "target",
    "future",
    "lead",
    "label",
)


@dataclass(slots=True)
class FeatureCatalog:
    identity_meta: list[str]
    calendar_features: list[str]
    raw_price_volume: list[str]
    price_returns_volatility: list[str]
    provided_technical_indicators: list[str]
    all_numeric_filtered: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        combined_primary = list(
            dict.fromkeys(
                self.raw_price_volume
                + self.price_returns_volatility
                + self.provided_technical_indicators
            )
        )
        return {
            "identity_meta": self.identity_meta,
            "calendar_features": self.calendar_features,
            "raw_price_volume": self.raw_price_volume,
            "price_returns_volatility": self.price_returns_volatility,
            "provided_technical_indicators": self.provided_technical_indicators,
            "price_technical_primary": combined_primary,
            "all_numeric_filtered": self.all_numeric_filtered,
        }


def _filtered_numeric_columns(frame, excluded: set[str]) -> list[str]:
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    numeric = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    kept = []
    seen_signatures: dict[int, str] = {}
    for column in numeric:
        series = frame[column]
        if float(series.isna().mean()) > 0.65:
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        signature = int(pd.util.hash_pandas_object(series.fillna(0), index=False).sum())
        duplicate_of = seen_signatures.get(signature)
        if duplicate_of is not None and frame[duplicate_of].equals(series):
            continue
        seen_signatures[signature] = column
        kept.append(column)
    return kept


def _configured_columns(frame, configured_groups: dict[str, list[str]], group_name: str) -> list[str]:
    configured = configured_groups.get(group_name, [])
    return [column for column in configured if column in frame.columns]


def _is_leakage_column(column: str, target_column: str) -> bool:
    normalized = column.lower()
    if normalized == target_column.lower():
        return True
    if normalized in {"target", "feature_date", "target_date"}:
        return True
    return any(token in normalized for token in LEAKAGE_TOKENS)


def build_feature_catalog(
    frame,
    target_column: str,
    date_column: str,
    symbol_column: str,
    *,
    configured_groups: dict[str, list[str]] | None = None,
    price_column: str = "adjclose",
) -> FeatureCatalog:
    configured_groups = configured_groups or {}
    excluded = {
        target_column,
        date_column,
        symbol_column,
        "feature_date",
        "target_date",
        "target_next_close",
        "target_next_adjclose",
        "TARGET",
    }
    filtered = [
        column
        for column in _filtered_numeric_columns(frame, excluded)
        if not _is_leakage_column(column, target_column)
    ]

    identity_meta = [column for column in ["age"] if column in frame.columns]
    calendar_features = [
        column
        for column in [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "feature_year",
            "feature_month",
            "feature_day",
            "feature_day_of_week",
            "feature_is_month_start",
            "feature_is_month_end",
            "feature_hour",
            "feature_minute",
        ]
        if column in filtered
    ]
    raw_price_volume = [
        column
        for column in ["open", "high", "low", "close", "adjclose", "volume"]
        if column in filtered or column == price_column
        if column in frame.columns
    ]
    price_returns_volatility = [
        column
        for column in filtered
        if column.startswith("feature_")
        and column not in calendar_features
    ]
    provided_technical_indicators = _configured_columns(
        frame,
        configured_groups,
        "provided_technical_indicators",
    )
    if not provided_technical_indicators:
        blocked = set(identity_meta) | set(calendar_features) | set(raw_price_volume) | set(price_returns_volatility)
        provided_technical_indicators = [
            column
            for column in filtered
            if column not in blocked
            if any(token in column.lower() for token in TECHNICAL_TOKENS)
            or not column.startswith("feature_")
        ]
    all_numeric_filtered = [
        column
        for column in filtered
        if column not in identity_meta
    ]

    return FeatureCatalog(
        identity_meta=list(dict.fromkeys(identity_meta)),
        calendar_features=list(dict.fromkeys(calendar_features)),
        raw_price_volume=list(dict.fromkeys(raw_price_volume)),
        price_returns_volatility=list(dict.fromkeys(price_returns_volatility)),
        provided_technical_indicators=list(dict.fromkeys(provided_technical_indicators)),
        all_numeric_filtered=list(dict.fromkeys(all_numeric_filtered)),
    )


def default_feature_set_for_model(model_name: str) -> str:
    if model_name == "transformer":
        return "price_technical_primary"
    if model_name in {"lstm", "gru", "arima_residual_lstm"}:
        return "price_technical_primary"
    return "raw_price_volume"


def resolve_feature_columns(
    frame,
    model_name: str,
    *,
    explicit_columns: list[str],
    feature_set: str,
    target_column: str,
    date_column: str,
    symbol_column: str,
    configured_groups: dict[str, list[str]] | None = None,
    price_column: str = "adjclose",
) -> list[str]:
    if explicit_columns:
        return explicit_columns
    catalog = build_feature_catalog(
        frame,
        target_column,
        date_column,
        symbol_column,
        configured_groups=configured_groups,
        price_column=price_column,
    )
    selected_set = feature_set if feature_set != "auto" else default_feature_set_for_model(model_name)
    feature_map = catalog.as_dict()
    if selected_set not in feature_map:
        raise ValueError(f"Unsupported feature_set: {selected_set}")
    columns = feature_map[selected_set]
    if not columns:
        raise ValueError(f"No feature columns resolved for feature_set={selected_set}")
    return columns


def resolve_feature_group_columns(
    frame,
    *,
    group_name: str,
    target_column: str,
    date_column: str,
    symbol_column: str,
    configured_groups: dict[str, list[str]] | None = None,
    price_column: str = "adjclose",
) -> list[str]:
    catalog = build_feature_catalog(
        frame,
        target_column,
        date_column,
        symbol_column,
        configured_groups=configured_groups,
        price_column=price_column,
    )
    feature_map = catalog.as_dict()
    if group_name not in feature_map:
        raise ValueError(f"Unsupported feature group: {group_name}")
    return feature_map[group_name]
