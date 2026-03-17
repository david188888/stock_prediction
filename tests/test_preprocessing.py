from pathlib import Path

import pandas

from stock_prediction.config import load_config
from stock_prediction.data.preprocessing import prepare_dataset


def test_prepare_dataset_aligns_to_kaggle_price_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    extracted_dir = raw_dir / "dataset"
    processed_dir = tmp_path / "processed"
    interim_dir = tmp_path / "interim"
    extracted_dir.mkdir(parents=True)

    frame = pandas.DataFrame(
        {
            "date": pandas.date_range("2024-01-01", periods=6),
            "ticker": ["AAA"] * 6,
            "open": [1, 2, 3, 4, 5, 6],
            "high": [2, 3, 4, 5, 6, 7],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            "close": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
            "adjclose": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "volume": [10, 11, 12, 13, 14, 15],
            "RSIadjclose15": [10, 20, 30, 40, 50, 60],
            "TARGET": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    frame.to_csv(extracted_dir / "sample.csv", index=False)

    config_path = tmp_path / "configs.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  slug: test",
                f"  raw_dir: {raw_dir}",
                f"  extracted_dir: {extracted_dir}",
                f"  processed_dir: {processed_dir}",
                f"  interim_dir: {interim_dir}",
                "  archive_name: sample.zip",
                f"  schema_path: {interim_dir / 'schema.json'}",
                "  prepared_filename: prepared.csv",
                "experiment:",
                "  stock_symbols: []",
                "  feature_columns: []",
                "  feature_set: auto",
                "  feature_groups:",
                "    provided_technical_indicators: []",
                "  target_column: target_next_adjclose",
                "  price_column: adjclose",
                "  date_column: date",
                "  symbol_column: symbol",
                "  prediction_horizon: 1",
                "  prediction_horizons: [1]",
                "  prediction_cutoff: close",
                "  window_size: 3",
                "  window_sizes: [3]",
                "  train_ratio: 0.6",
                "  val_ratio: 0.2",
                "  test_ratio: 0.2",
                "  split_mode: date",
                "  seed: 42",
                f"  output_dir: {tmp_path / 'outputs'}",
                "  generate_figures: false",
                "  evaluation_modes: [holdout]",
                "  walk_forward_steps: 0",
                "  walk_forward_models: [linear_regression]",
                "  selected_models: [linear_regression]",
                "  backtest_window_type: expanding",
                "  early_stopping_patience: 2",
                "  min_delta: 0.0001",
                "  tune_arima_order: false",
                "  extreme_volatility_quantile: 0.95",
                "  winsorize_limits: [0.01, 0.99]",
                "models:",
                "  linear_regression: {}",
                "  linear_regression_scaled: {}",
                "  arima:",
                "    order: [1, 1, 0]",
                "  garch_return:",
                "    p: 1",
                "    q: 1",
                "    lags: 1",
                "  lstm:",
                "    hidden_size: 4",
                "    num_layers: 1",
                "    dropout: 0.0",
                "    epochs: 2",
                "    batch_size: 2",
                "    learning_rate: 0.001",
                "  gru:",
                "    hidden_size: 4",
                "    num_layers: 1",
                "    dropout: 0.0",
                "    epochs: 2",
                "    batch_size: 2",
                "    learning_rate: 0.001",
                "  transformer:",
                "    hidden_size: 4",
                "    num_layers: 1",
                "    dropout: 0.0",
                "    epochs: 2",
                "    batch_size: 2",
                "    learning_rate: 0.001",
                "  hybrid:",
                "    arima_order: [1, 1, 0]",
                "    hidden_size: 4",
                "    num_layers: 1",
                "    dropout: 0.0",
                "    epochs: 2",
                "    batch_size: 2",
                "    learning_rate: 0.001",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    prepared = prepare_dataset(config)
    assert "target_next_adjclose" in prepared.dataframe.columns
    assert "target_next_close" in prepared.dataframe.columns
    assert "TARGET" in prepared.dataframe.columns
    assert "feature_date" in prepared.dataframe.columns
    assert "target_date" in prepared.dataframe.columns
    assert "symbol" in prepared.dataframe.columns
    assert "feature_price_return_1" in prepared.dataframe.columns
    assert Path(config.dataset.schema_path).exists()
