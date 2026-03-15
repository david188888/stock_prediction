from pathlib import Path

from stock_prediction.config import load_config


def test_load_config_resolves_paths():
    config = load_config("configs/default.yaml")
    assert config.dataset.raw_dir == Path.cwd() / "data" / "raw"
    assert config.experiment.output_dir == Path.cwd() / "outputs"
    assert config.experiment.target_column == "target_next_close"
    assert "walk_forward" in config.experiment.evaluation_modes
    assert "linear_regression_scaled" in config.experiment.walk_forward_models
