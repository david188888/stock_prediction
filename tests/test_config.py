from pathlib import Path

from stock_prediction.config import load_config


def test_load_config_resolves_price_workflow_defaults():
    config = load_config("configs/default.yaml")
    assert config.dataset.raw_dir == Path.cwd() / "data" / "raw"
    assert config.experiment.output_dir == Path.cwd() / "outputs"
    assert config.experiment.target_column == "target_next_adjclose"
    assert config.experiment.price_column == "adjclose"
    assert config.experiment.prediction_horizon == 1
    assert config.experiment.prediction_horizons == [1, 5]
    assert config.experiment.window_sizes == [20, 60]
    assert config.experiment.split_mode == "date"
    assert "walk_forward" in config.experiment.evaluation_modes
    assert "garch_return" in config.experiment.walk_forward_models
    assert "transformer" in config.experiment.selected_models
