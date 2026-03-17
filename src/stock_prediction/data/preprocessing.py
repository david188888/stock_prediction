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
    schema = DatasetSchema(
        source_file=str(source),
        symbol_column=infer_optional_column_name(columns, "symbol"),
        date_column=infer_column_name(columns, "date"),
        open_column=infer_column_name(columns, "open"),
        high_column=infer_column_name(columns, "high"),
        low_column=infer_column_name(columns, "low"),
        close_column=infer_column_name(columns, "close"),
        volume_column=infer_column_name(columns, "volume"),
    )
    return schema


def infer_schema(extracted_dir: Path) -> DatasetSchema:
    source, frame = _load_supported_files(extracted_dir)[0]
    return infer_schema_for_frame(source, frame)


def _add_derived_features(frame, symbol_column: str):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    grouped_close = frame.groupby(symbol_column)["close"]
    grouped_volume = frame.groupby(symbol_column)["volume"]

    frame["feature_close_return_1"] = grouped_close.pct_change().replace([float("inf"), float("-inf")], 0.0)
    frame["feature_close_return_5"] = grouped_close.pct_change(5).replace([float("inf"), float("-inf")], 0.0)
    frame["feature_volume_return_1"] = grouped_volume.pct_change().replace([float("inf"), float("-inf")], 0.0)
    frame["feature_intraday_return"] = ((frame["close"] - frame["open"]) / frame["open"]).replace(
        [float("inf"), float("-inf")],
        0.0,
    )
    frame["feature_high_low_spread"] = ((frame["high"] - frame["low"]) / frame["close"]).replace(
        [float("inf"), float("-inf")],
        0.0,
    )
    frame["feature_rolling_volatility_5"] = grouped_close.pct_change().rolling(5).std().reset_index(level=0, drop=True)
    frame["feature_rolling_volatility_10"] = grouped_close.pct_change().rolling(10).std().reset_index(level=0, drop=True)
    frame["feature_rolling_mean_return_5"] = (
        grouped_close.pct_change().rolling(5).mean().reset_index(level=0, drop=True)
    )

    derived_columns = [column for column in frame.columns if column.startswith("feature_")]
    frame[derived_columns] = frame[derived_columns].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    return frame


def prepare_dataset(config: AppConfig) -> PreparedDataset:
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")

    standardized_frames = []
    schemas = []
    loaded_files = _load_supported_files(config.dataset.extracted_dir)
    source = loaded_files[0][0]

    for source_file, frame in loaded_files:
        schema = infer_schema_for_frame(source_file, frame)
        schemas.append(schema.to_dict())
        rename_map = {
            schema.date_column: config.experiment.date_column,
            schema.open_column: "open",
            schema.high_column: "high",
            schema.low_column: "low",
            schema.close_column: "close",
            schema.volume_column: "volume",
        }
        if schema.symbol_column:
            rename_map[schema.symbol_column] = config.experiment.symbol_column

        standardized_frame = frame.rename(columns=rename_map).copy()
        if config.experiment.symbol_column not in standardized_frame.columns:
            standardized_frame[config.experiment.symbol_column] = source_file.stem.upper()
        standardized_frames.append(standardized_frame)

    standardized = pd.concat(standardized_frames, ignore_index=True)

    standardized[config.experiment.date_column] = pd.to_datetime(
        standardized[config.experiment.date_column], errors="coerce"
    )
    standardized = standardized.dropna(subset=[config.experiment.date_column, "close"])
    standardized[config.experiment.symbol_column] = standardized[
        config.experiment.symbol_column
    ].astype(str)
    standardized = standardized.drop_duplicates(
        subset=[config.experiment.symbol_column, config.experiment.date_column]
    )

    for column in ["open", "high", "low", "close", "volume"]:
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")

    standardized = standardized.dropna(subset=["open", "high", "low", "close"])
    standardized = standardized.sort_values(
        [config.experiment.symbol_column, config.experiment.date_column]
    ).reset_index(drop=True)
    standardized = _add_derived_features(standardized, config.experiment.symbol_column)
    standardized["feature_date"] = standardized[config.experiment.date_column]
    standardized["target_date"] = standardized.groupby(config.experiment.symbol_column)[
        config.experiment.date_column
    ].shift(-config.experiment.prediction_horizon)
    feature_catalog = build_feature_catalog(
        standardized,
        target_column=config.experiment.target_column,
        date_column=config.experiment.date_column,
        symbol_column=config.experiment.symbol_column,
        configured_groups=config.experiment.feature_groups,
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
                "volume": "volume",
                "feature_date": "feature_date",
                "target_date": "target_date",
                "target": config.experiment.target_column,
            },
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
