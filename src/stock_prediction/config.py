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
    feature_groups: dict[str, list[str]]
    target_column: str
    price_column: str
    date_column: str
    symbol_column: str
    prediction_horizon: int
    prediction_horizons: list[int]
    prediction_cutoff: str
    window_size: int
    window_sizes: list[int]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    split_mode: str
    seed: int
    output_dir: Path
    generate_figures: bool
    evaluation_modes: list[str]
    walk_forward_steps: int
    walk_forward_models: list[str]
    selected_models: list[str]
    backtest_window_type: str
    early_stopping_patience: int
    min_delta: float
    tune_arima_order: bool
    extreme_volatility_quantile: float
    winsorize_limits: tuple[float, float]


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
        feature_groups={
            key: list(value)
            for key, value in dict(experiment_payload.get("feature_groups", {})).items()
        },
        target_column=experiment_payload["target_column"],
        price_column=str(experiment_payload.get("price_column", "adjclose")),
        date_column=experiment_payload["date_column"],
        symbol_column=experiment_payload["symbol_column"],
        prediction_horizon=int(experiment_payload.get("prediction_horizon", 1)),
        prediction_horizons=[
            int(value)
            for value in experiment_payload.get(
                "prediction_horizons",
                [experiment_payload.get("prediction_horizon", 1)],
            )
        ],
        prediction_cutoff=str(experiment_payload.get("prediction_cutoff", "close")),
        window_size=int(experiment_payload["window_size"]),
        window_sizes=[
            int(value)
            for value in experiment_payload.get("window_sizes", [experiment_payload["window_size"]])
        ],
        train_ratio=float(experiment_payload["train_ratio"]),
        val_ratio=float(experiment_payload["val_ratio"]),
        test_ratio=float(experiment_payload["test_ratio"]),
        split_mode=str(experiment_payload.get("split_mode", "date")),
        seed=int(experiment_payload.get("seed", experiment_payload.get("random_seed", 42))),
        output_dir=_resolve_path(experiment_payload["output_dir"], root),
        generate_figures=bool(experiment_payload["generate_figures"]),
        evaluation_modes=list(experiment_payload.get("evaluation_modes", ["holdout"])),
        walk_forward_steps=int(experiment_payload.get("walk_forward_steps", 0)),
        walk_forward_models=list(experiment_payload.get("walk_forward_models", [])),
        selected_models=list(experiment_payload.get("selected_models", [])),
        backtest_window_type=str(experiment_payload.get("backtest_window_type", "expanding")),
        early_stopping_patience=int(experiment_payload.get("early_stopping_patience", 5)),
        min_delta=float(experiment_payload.get("min_delta", 1e-4)),
        tune_arima_order=bool(experiment_payload.get("tune_arima_order", False)),
        extreme_volatility_quantile=float(experiment_payload.get("extreme_volatility_quantile", 0.95)),
        winsorize_limits=tuple(
            float(value) for value in experiment_payload.get("winsorize_limits", [0.01, 0.99])
        ),
    )

    ratio_sum = round(experiment.train_ratio + experiment.val_ratio + experiment.test_ratio, 10)
    if ratio_sum != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    allowed_evaluations = {"holdout", "walk_forward"}
    invalid_evaluations = set(experiment.evaluation_modes) - allowed_evaluations
    if invalid_evaluations:
        raise ValueError(f"Unsupported evaluation modes: {sorted(invalid_evaluations)}")
    allowed_split_modes = {"date", "ratio"}
    if experiment.split_mode not in allowed_split_modes:
        raise ValueError(f"Unsupported split_mode: {experiment.split_mode}")
    if experiment.prediction_horizon < 1:
        raise ValueError("prediction_horizon must be at least 1")
    if any(value < 1 for value in experiment.prediction_horizons):
        raise ValueError("prediction_horizons must all be at least 1")
    if experiment.prediction_cutoff != "close":
        raise ValueError("Only prediction_cutoff='close' is supported in this version")
    if experiment.backtest_window_type != "expanding":
        raise ValueError("Only backtest_window_type='expanding' is supported in this version")
    if not 0.0 < experiment.extreme_volatility_quantile < 1.0:
        raise ValueError("extreme_volatility_quantile must be between 0 and 1")
    if len(experiment.window_sizes) == 0 or any(value < 1 for value in experiment.window_sizes):
        raise ValueError("window_sizes must contain positive integers")
    if len(experiment.winsorize_limits) != 2:
        raise ValueError("winsorize_limits must contain two values")
    lower_limit, upper_limit = experiment.winsorize_limits
    if not 0.0 <= lower_limit < upper_limit <= 1.0:
        raise ValueError("winsorize_limits must satisfy 0 <= lower < upper <= 1")
    if experiment.walk_forward_steps < 0:
        raise ValueError("walk_forward_steps must be non-negative")

    return AppConfig(dataset=dataset, experiment=experiment, models=payload["models"])
