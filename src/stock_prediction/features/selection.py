from __future__ import annotations

from dataclasses import dataclass

from stock_prediction.utils.dependencies import require_dependency


TECHNICAL_TOKENS = (
    "rsi",
    "macd",
    "sma",
    "ema",
    "stochastic",
    "adl",
    "mfm",
    "mfv",
    "feargreed",
    "hammer",
    "supernova",
    "vela",
    "fuerzarelativa",
    "volumenrelativo",
)

CORE_TECHNICAL_TOKENS = (
    "rsi",
    "macd",
    "stochastic",
    "feargreed",
    "adl",
)


@dataclass(slots=True)
class FeatureCatalog:
    price_basic: list[str]
    technical_indicators: list[str]
    returns_volatility: list[str]
    all_numeric_filtered: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "price_basic": self.price_basic,
            "technical_indicators": self.technical_indicators,
            "returns_volatility": self.returns_volatility,
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
        if float(series.isna().mean()) > 0.35:
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


def build_feature_catalog(frame, target_column: str, date_column: str, symbol_column: str) -> FeatureCatalog:
    excluded = {target_column, date_column, symbol_column}
    filtered = _filtered_numeric_columns(frame, excluded)

    price_basic = [column for column in ["open", "high", "low", "close", "volume"] if column in frame.columns]

    technical = [
        column
        for column in filtered
        if any(token in column.lower() for token in TECHNICAL_TOKENS)
    ]
    returns_volatility = [
        column
        for column in filtered
        if column.startswith("feature_")
        or any(token in column.lower() for token in CORE_TECHNICAL_TOKENS)
    ]

    technical_indicators = list(dict.fromkeys(price_basic + technical))
    returns_volatility = list(dict.fromkeys(returns_volatility))
    if not returns_volatility:
        returns_volatility = price_basic.copy()

    return FeatureCatalog(
        price_basic=price_basic,
        technical_indicators=technical_indicators or price_basic.copy(),
        returns_volatility=returns_volatility,
        all_numeric_filtered=filtered,
    )


def default_feature_set_for_model(model_name: str) -> str:
    if model_name in {"lstm", "gru", "arima_residual_lstm"}:
        return "returns_volatility"
    return "price_basic"


def resolve_feature_columns(
    frame,
    model_name: str,
    *,
    explicit_columns: list[str],
    feature_set: str,
    target_column: str,
    date_column: str,
    symbol_column: str,
) -> list[str]:
    if explicit_columns:
        return explicit_columns
    catalog = build_feature_catalog(frame, target_column, date_column, symbol_column)
    selected_set = feature_set if feature_set != "auto" else default_feature_set_for_model(model_name)
    feature_map = catalog.as_dict()
    if selected_set not in feature_map:
        raise ValueError(f"Unsupported feature_set: {selected_set}")
    columns = feature_map[selected_set]
    if not columns:
        raise ValueError(f"No feature columns resolved for feature_set={selected_set}")
    return columns
