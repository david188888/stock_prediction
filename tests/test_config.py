from pathlib import Path

from stock_prediction.config import load_config


def test_load_config_resolves_paths():
    config = load_config("configs/default.yaml")
    assert config.dataset.raw_dir == Path.cwd() / "data" / "raw"
    assert config.experiment.output_dir == Path.cwd() / "outputs"

