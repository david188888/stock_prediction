import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("arch")

from stock_prediction.models.factory import create_model


def test_factory_builds_transformer_and_garch_models():
    transformer = create_model(
        "transformer",
        {
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
        },
        input_size=4,
        window_size=5,
    )
    garch = create_model(
        "garch_return",
        {"p": 1, "q": 1, "lags": 1, "mean": "ARX", "dist": "normal"},
        input_size=1,
        window_size=5,
    )

    assert transformer.model_name == "transformer"
    assert garch.model_name == "garch_return"
