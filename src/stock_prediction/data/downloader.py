from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZipFile

from stock_prediction.config import DatasetConfig
from stock_prediction.utils.io import ensure_dir


def kaggle_credentials_available() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return Path.home().joinpath(".kaggle", "kaggle.json").exists()


def download_dataset(config: DatasetConfig) -> Path:
    ensure_dir(config.raw_dir)
    if not kaggle_credentials_available():
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY or place "
            "~/.kaggle/kaggle.json, or manually extract the dataset into "
            f"{config.extracted_dir}."
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `kaggle`. Run `uv sync` first.") from exc

    archive_path = config.raw_dir / config.archive_name
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        dataset=config.slug,
        path=str(config.raw_dir),
        quiet=False,
        unzip=False,
    )

    if not archive_path.exists():
        downloaded_archives = list(config.raw_dir.glob("*.zip"))
        if not downloaded_archives:
            raise FileNotFoundError("Expected Kaggle archive was not downloaded.")
        archive_path = downloaded_archives[0]

    ensure_dir(config.extracted_dir)
    with ZipFile(archive_path, "r") as archive:
        archive.extractall(config.extracted_dir)
    return config.extracted_dir

