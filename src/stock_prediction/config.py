from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    slug: str
    raw_dir: Path
    extracted_dir: Path
    processed_dir: Path
    interim_dir: Path
    archive_name: str
    schema_path: Path
    prepared_filename: str


@dataclass(slots=True)
class ExperimentConfig:
    stock_symbols: list[str]
    feature_columns: list[str]
    feature_set: str
    target_column: str
    date_column: str
    symbol_column: str
    window_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    output_dir: Path
    generate_figures: bool
    evaluation_modes: list[str]
    walk_forward_steps: int
    walk_forward_models: list[str]
    early_stopping_patience: int
    min_delta: float
    tune_arima_order: bool


@dataclass(slots=True)
class AppConfig:
    dataset: DatasetConfig
    experiment: ExperimentConfig
    models: dict[str, dict[str, Any]]


def _resolve_path(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    root = path.parent.parent
    dataset_payload = payload["dataset"]
    experiment_payload = payload["experiment"]

    dataset = DatasetConfig(
        slug=dataset_payload["slug"],
        raw_dir=_resolve_path(dataset_payload["raw_dir"], root),
        extracted_dir=_resolve_path(dataset_payload["extracted_dir"], root),
        processed_dir=_resolve_path(dataset_payload["processed_dir"], root),
        interim_dir=_resolve_path(dataset_payload["interim_dir"], root),
        archive_name=dataset_payload["archive_name"],
        schema_path=_resolve_path(dataset_payload["schema_path"], root),
        prepared_filename=dataset_payload["prepared_filename"],
    )

    experiment = ExperimentConfig(
        stock_symbols=list(experiment_payload["stock_symbols"]),
        feature_columns=list(experiment_payload.get("feature_columns", [])),
        feature_set=experiment_payload.get("feature_set", "auto"),
        target_column=experiment_payload["target_column"],
        date_column=experiment_payload["date_column"],
        symbol_column=experiment_payload["symbol_column"],
        window_size=int(experiment_payload["window_size"]),
        train_ratio=float(experiment_payload["train_ratio"]),
        val_ratio=float(experiment_payload["val_ratio"]),
        test_ratio=float(experiment_payload["test_ratio"]),
        seed=int(experiment_payload.get("seed", experiment_payload.get("random_seed", 42))),
        output_dir=_resolve_path(experiment_payload["output_dir"], root),
        generate_figures=bool(experiment_payload["generate_figures"]),
        evaluation_modes=list(experiment_payload.get("evaluation_modes", ["holdout"])),
        walk_forward_steps=int(experiment_payload["walk_forward_steps"]),
        walk_forward_models=list(experiment_payload.get("walk_forward_models", [])),
        early_stopping_patience=int(experiment_payload.get("early_stopping_patience", 5)),
        min_delta=float(experiment_payload.get("min_delta", 1e-4)),
        tune_arima_order=bool(experiment_payload.get("tune_arima_order", False)),
    )

    ratio_sum = round(experiment.train_ratio + experiment.val_ratio + experiment.test_ratio, 10)
    if ratio_sum != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    allowed_evaluations = {"holdout", "walk_forward"}
    invalid_evaluations = set(experiment.evaluation_modes) - allowed_evaluations
    if invalid_evaluations:
        raise ValueError(f"Unsupported evaluation modes: {sorted(invalid_evaluations)}")

    return AppConfig(dataset=dataset, experiment=experiment, models=payload["models"])
