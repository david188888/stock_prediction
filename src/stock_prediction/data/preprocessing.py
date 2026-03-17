from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stock_prediction.config import AppConfig
from stock_prediction.data.schema import DatasetSchema, infer_column_name, infer_optional_column_name
from stock_prediction.features.selection import build_feature_catalog
from stock_prediction.utils.dependencies import require_dependency
from stock_prediction.utils.io import ensure_dir, write_json


@dataclass(slots=True)
class PreparedDataset:
    dataframe: "pd.DataFrame"
    schema: DatasetSchema
    prepared_path: Path


def _load_supported_files(extracted_dir: Path) -> list[tuple[Path, "pd.DataFrame"]]:
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    candidates = sorted(extracted_dir.rglob("*.csv")) + sorted(extracted_dir.rglob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"No supported data files found under {extracted_dir}")

    loaded = []
    for source in candidates:
        if source.suffix.lower() == ".csv":
            frame = pd.read_csv(source)
        else:
            frame = pd.read_excel(source)
        loaded.append((source, frame))
    return loaded


def infer_schema_for_frame(source: Path, frame) -> DatasetSchema:
    columns = list(frame.columns)
    date_column = infer_optional_column_name(columns, "date")
    year_column = infer_optional_column_name(columns, "year")
    month_column = infer_optional_column_name(columns, "month")
    day_column = infer_optional_column_name(columns, "day")
    if date_column is None and not (year_column and month_column and day_column):
        raise ValueError(
            "Unable to infer datetime information. Expected `date` or at least `year/month/day` columns."
        )
    schema = DatasetSchema(
        source_file=str(source),
        symbol_column=infer_optional_column_name(columns, "symbol"),
        date_column=date_column,
        year_column=year_column,
        month_column=month_column,
        day_column=day_column,
        hour_column=infer_optional_column_name(columns, "hour"),
        minute_column=infer_optional_column_name(columns, "minute"),
        open_column=infer_column_name(columns, "open"),
        high_column=infer_column_name(columns, "high"),
        low_column=infer_column_name(columns, "low"),
        close_column=infer_column_name(columns, "close"),
        adjclose_column=infer_optional_column_name(columns, "adjclose"),
        volume_column=infer_column_name(columns, "volume"),
        market_column=infer_optional_column_name(columns, "market"),
        age_column=infer_optional_column_name(columns, "age"),
        target_column=infer_optional_column_name(columns, "target"),
    )
    return schema


def infer_schema(extracted_dir: Path) -> DatasetSchema:
    source, frame = _load_supported_files(extracted_dir)[0]
    return infer_schema_for_frame(source, frame)


def _standardize_optional_columns(frame, schema: DatasetSchema, config: AppConfig):
    rename_map = {
        schema.open_column: "open",
        schema.high_column: "high",
        schema.low_column: "low",
        schema.close_column: "close",
        schema.volume_column: "volume",
    }
    if schema.symbol_column:
        rename_map[schema.symbol_column] = config.experiment.symbol_column
    if schema.date_column:
        rename_map[schema.date_column] = config.experiment.date_column
    if schema.adjclose_column:
        rename_map[schema.adjclose_column] = "adjclose"
    if schema.market_column:
        rename_map[schema.market_column] = "market"
    if schema.age_column:
        rename_map[schema.age_column] = "age"
    if schema.target_column:
        rename_map[schema.target_column] = "TARGET"
    if schema.year_column:
        rename_map[schema.year_column] = "year"
    if schema.month_column:
        rename_map[schema.month_column] = "month"
    if schema.day_column:
        rename_map[schema.day_column] = "day"
    if schema.hour_column:
        rename_map[schema.hour_column] = "hour"
    if schema.minute_column:
        rename_map[schema.minute_column] = "minute"

    standardized = frame.rename(columns=rename_map).copy()
    if config.experiment.symbol_column not in standardized.columns:
        standardized[config.experiment.symbol_column] = Path(schema.source_file).stem.upper()
    return standardized


def _build_timestamp(frame, date_column: str):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    if date_column in frame.columns:
        return pd.to_datetime(frame[date_column], errors="coerce")

    year = frame.get("year")
    month = frame.get("month")
    day = frame.get("day")
    if year is None or month is None or day is None:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")

    hour = frame.get("hour")
    minute = frame.get("minute")
    if hour is None:
        hour = 0
    if minute is None:
        minute = 0
    payload = {
        "year": pd.to_numeric(year, errors="coerce"),
        "month": pd.to_numeric(month, errors="coerce"),
        "day": pd.to_numeric(day, errors="coerce"),
        "hour": pd.to_numeric(hour, errors="coerce"),
        "minute": pd.to_numeric(minute, errors="coerce"),
    }
    return pd.to_datetime(payload, errors="coerce")


def _coerce_numeric_columns(frame, columns: list[str]) -> None:
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _normalize_time_components(frame) -> None:
    timestamp = frame["date"]
    frame["year"] = timestamp.dt.year
    frame["month"] = timestamp.dt.month
    frame["day"] = timestamp.dt.day
    frame["hour"] = timestamp.dt.hour
    frame["minute"] = timestamp.dt.minute


def _drop_invalid_rows(frame, symbol_column: str):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    required_prices = ["open", "high", "low", "close", "adjclose"]
    frame = frame.dropna(subset=["date", symbol_column, *required_prices])
    frame = frame[frame["volume"].fillna(0.0) >= 0.0]
    for column in required_prices:
        frame = frame[frame[column] > 0.0]
    frame = frame[frame["high"] >= frame["low"]]
    frame = frame[frame["high"] >= frame[["open", "close", "adjclose"]].max(axis=1)]
    frame = frame[frame["low"] <= frame[["open", "close", "adjclose"]].min(axis=1)]
    frame[symbol_column] = frame[symbol_column].astype(str).str.strip()
    frame = frame[frame[symbol_column] != ""]
    frame = frame.drop_duplicates(subset=[symbol_column, "date"]).sort_values([symbol_column, "date"])
    return frame.reset_index(drop=True)


def _add_calendar_features(frame) -> None:
    frame["feature_year"] = frame["date"].dt.year.astype("Int64")
    frame["feature_month"] = frame["date"].dt.month.astype("Int64")
    frame["feature_day"] = frame["date"].dt.day.astype("Int64")
    frame["feature_day_of_week"] = frame["date"].dt.dayofweek.astype("Int64")
    frame["feature_is_month_start"] = frame["date"].dt.is_month_start.astype(int)
    frame["feature_is_month_end"] = frame["date"].dt.is_month_end.astype(int)
    frame["feature_hour"] = frame["date"].dt.hour.astype("Int64")
    frame["feature_minute"] = frame["date"].dt.minute.astype("Int64")


def _add_derived_features(frame, symbol_column: str, price_column: str):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    grouped_price = frame.groupby(symbol_column, sort=False)[price_column]
    grouped_volume = frame.groupby(symbol_column, sort=False)["volume"]

    derived_series = {
        "feature_price_return_1": grouped_price.pct_change(),
        "feature_price_return_5": grouped_price.pct_change(5),
        "feature_log_return_1": grouped_price.transform(lambda series: series.pct_change()).add(1.0),
        "feature_volume_return_1": grouped_volume.pct_change(),
        "feature_intraday_return": (frame["close"] - frame["open"]) / frame["open"],
        "feature_high_low_spread": (frame["high"] - frame["low"]) / frame[price_column],
        "feature_rolling_volatility_5": grouped_price.transform(lambda series: series.pct_change().rolling(5).std()),
        "feature_rolling_volatility_10": grouped_price.transform(lambda series: series.pct_change().rolling(10).std()),
        "feature_rolling_mean_return_5": grouped_price.transform(lambda series: series.pct_change().rolling(5).mean()),
    }
    derived_series["feature_log_return_1"] = derived_series["feature_log_return_1"].map(
        lambda value: None if value is None or value <= 0 else pd.NA if pd.isna(value) else value
    )

    for column, values in derived_series.items():
        if column in frame.columns:
            continue
        frame[column] = values

    derived_columns = [column for column in frame.columns if column.startswith("feature_")]
    frame[derived_columns] = frame[derived_columns].replace([float("inf"), float("-inf")], pd.NA)
    if "feature_log_return_1" in frame.columns:
        import math

        frame["feature_log_return_1"] = frame["feature_log_return_1"].map(
            lambda value: 0.0 if pd.isna(value) else math.log(float(value))
        )
    frame[derived_columns] = frame[derived_columns].fillna(0.0)
    return frame


def _sort_optional_columns(frame):
    preferred = [
        "date",
        "feature_date",
        "target_date",
        "symbol",
        "ticker",
        "market",
        "age",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "open",
        "high",
        "low",
        "close",
        "adjclose",
        "volume",
        "TARGET",
        "target_next_close",
        "target_next_adjclose",
    ]
    remaining = [column for column in frame.columns if column not in preferred]
    return frame[[column for column in preferred if column in frame.columns] + remaining]


def prepare_dataset(config: AppConfig) -> PreparedDataset:
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")

    standardized_frames = []
    schemas = []
    loaded_files = _load_supported_files(config.dataset.extracted_dir)
    source = loaded_files[0][0]

    for source_file, frame in loaded_files:
        schema = infer_schema_for_frame(source_file, frame)
        schemas.append(schema.to_dict())
        standardized_frame = _standardize_optional_columns(frame, schema, config)
        standardized_frames.append(standardized_frame)

    standardized = pd.concat(standardized_frames, ignore_index=True)
    standardized["date"] = _build_timestamp(standardized, config.experiment.date_column)
    _coerce_numeric_columns(
        standardized,
        ["open", "high", "low", "close", "adjclose", "volume", "age", "TARGET", "year", "month", "day", "hour", "minute"],
    )
    if "adjclose" not in standardized.columns:
        standardized["adjclose"] = standardized["close"]
    standardized["adjclose"] = standardized["adjclose"].fillna(standardized["close"])
    if "market" not in standardized.columns:
        standardized["market"] = pd.NA
    if "age" not in standardized.columns:
        standardized["age"] = pd.NA
    if "TARGET" not in standardized.columns:
        standardized["TARGET"] = pd.NA

    standardized = _drop_invalid_rows(standardized, config.experiment.symbol_column)
    _normalize_time_components(standardized)
    _add_calendar_features(standardized)
    standardized = _add_derived_features(
        standardized,
        config.experiment.symbol_column,
        config.experiment.price_column,
    )
    standardized["feature_date"] = standardized["date"]
    grouped = standardized.groupby(config.experiment.symbol_column, sort=False)
    standardized["target_date"] = grouped["date"].shift(-config.experiment.prediction_horizon)
    standardized["target_next_close"] = grouped["close"].shift(-config.experiment.prediction_horizon)
    standardized["target_next_adjclose"] = grouped["adjclose"].shift(-config.experiment.prediction_horizon)
    standardized = _sort_optional_columns(standardized)

    feature_catalog = build_feature_catalog(
        standardized,
        target_column=config.experiment.target_column,
        date_column=config.experiment.date_column,
        symbol_column=config.experiment.symbol_column,
        configured_groups=config.experiment.feature_groups,
        price_column=config.experiment.price_column,
    )

    ensure_dir(config.dataset.interim_dir)
    ensure_dir(config.dataset.processed_dir)
    write_json(
        config.dataset.schema_path,
        {
            "source_file": str(source),
            "standardized_columns": {
                "symbol": config.experiment.symbol_column,
                "date": config.experiment.date_column,
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "adjclose": "adjclose",
                "volume": "volume",
                "market": "market",
                "age": "age",
                "target_builtin": "TARGET",
                "feature_date": "feature_date",
                "target_date": "target_date",
                "target": config.experiment.target_column,
                "aux_targets": ["target_next_close", "target_next_adjclose"],
            },
            "price_column": config.experiment.price_column,
            "inferred_schema": schemas,
            "feature_groups": feature_catalog.as_dict(),
        },
    )

    prepared_path = config.dataset.processed_dir / config.dataset.prepared_filename
    standardized.to_csv(prepared_path, index=False)
    return PreparedDataset(
        dataframe=standardized,
        schema=infer_schema_for_frame(source, loaded_files[0][1]),
        prepared_path=prepared_path,
    )
