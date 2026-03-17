# Stock Prediction

Research-oriented stock price prediction project aligned with the proposal in [report.md](/Users/david/codespace/stock_prediction/report.md) and the Kaggle dataset [luisandresgarcia/stock-market-prediction](https://www.kaggle.com/datasets/luisandresgarcia/stock-market-prediction).

## Documentation Index

- Research proposal / 开题报告: [report.md](/Users/david/codespace/stock_prediction/report.md)
- Detailed code guide / 中文代码说明书: [docs/codebase_guide.md](/Users/david/codespace/stock_prediction/docs/codebase_guide.md)

If you are reviewing this repository for academic discussion, read `report.md` first and then `docs/codebase_guide.md`. The first explains the research intent; the second explains the exact code workflow, module boundaries, experiment protocol, and outputs.

## Scope

- Predict future stock prices on the Kaggle dataset using `adjclose` as the default primary price series.
- Reuse the dataset's built-in technical indicators instead of assuming external news or sentiment inputs.
- Provide a reproducible pipeline for download, preparation, training, holdout evaluation, full walk-forward validation, and experiment reporting.
- Support `linear_regression`, `linear_regression_scaled`, `arima`, `garch_return`, `lstm`, `gru`, `transformer`, and `arima_residual_lstm`.

## Quick Start

```bash
conda env create -f environment.yml
conda activate stock-prediction
stock-prediction download-data --config configs/default.yaml
stock-prediction prepare-data --config configs/default.yaml
stock-prediction run-experiment --config configs/default.yaml
```

If you are running from source without installing the package, use `PYTHONPATH=src`.

## Kaggle-Aligned Workflow

1. `download-data`
   - Download or unpack the Kaggle dataset into `data/raw/stock-market-prediction/`.
2. `prepare-data`
   - Adapt Kaggle columns such as `date`, `ticker`, `open/high/low/close/adjclose`, `volume`, and the built-in technical indicators.
   - Standardize identifiers to `symbol` and rebuild calendar features from the timestamp.
   - Validate prices, remove invalid rows and duplicate timestamps, preserve `TARGET`, and generate `target_next_adjclose` and `target_next_close`.
   - Build feature-group metadata in `data/interim/schema.json`.
3. `train` / `evaluate`
   - Train one model using the active `price_column`, `prediction_horizon`, and `window_size`.
4. `run-experiment`
   - Iterate across the configured `prediction_horizons`, `window_sizes`, and `selected_models`.
   - Save per-run outputs under `outputs/` with matrix-aware filenames.

## Data Processing

The default dataset adapter assumes the Kaggle file contains:

- Raw market data: `open`, `high`, `low`, `close`, `adjclose`, `volume`
- Identifier and time data: `ticker` or `company`, `date`
- Optional metadata: `age`, `market`
- A large set of precomputed technical indicators
- Built-in label: `TARGET` retained for reference only

Prepared outputs include:

- Primary targets: `target_next_adjclose` and `target_next_close`
- Calendar features derived from `date`
- Lightweight derived return and volatility features
- Feature groups:
  - `identity_meta`
  - `calendar_features`
  - `raw_price_volume`
  - `price_returns_volatility`
  - `provided_technical_indicators`
  - `price_technical_primary`
  - `all_numeric_filtered`

Default modeling choices:

- `target_column: target_next_adjclose`
- `price_column: adjclose`
- `split_mode: date`
- `feature_set: auto`

## Models

- `linear_regression`
  - Tabular price and technical-feature baseline
- `linear_regression_scaled`
  - Linear baseline with feature scaling
- `arima`
  - Univariate ARIMA on the primary price series
- `garch_return`
  - GARCH-style return model converted back into next-step price forecasts
- `lstm`
  - Recurrent sequence regressor with validation monitoring and early stopping
- `gru`
  - GRU variant with the same training protocol
- `transformer`
  - Encoder-only sequence regressor over the same windowed inputs
- `arima_residual_lstm`
  - Price-only hybrid that fits ARIMA first and then learns residual dynamics with an LSTM

## Evaluation Protocol

- Holdout evaluation:
  `train -> validation selection -> train+validation refit -> test`
- Walk-forward evaluation:
  expanding window over the full test period by default
- Main metrics:
  `MAE`, `MSE`, `RMSE`
- Supplementary metric:
  directional accuracy `DA`

## Outputs

- `outputs/predictions/`
  - Per-model prediction CSV files keyed by `price_column`, `prediction_horizon`, and `window_size`
- `outputs/metrics/*_metrics.csv`
  - Per-symbol holdout and walk-forward metrics
- `outputs/metrics/*_summary.md`
  - Aggregated summary tables
- `outputs/metrics/*_conclusion.md`
  - Short interpretation pages
- `outputs/metrics/*_training_log.csv`
  - Validation and model-selection logs
- `outputs/metrics/leaderboard.csv`
  - Matrix-wide comparison across models, horizons, and window sizes

## Current Constraints

- The implementation is intentionally aligned to the Kaggle price-and-technical-indicator dataset only.
- External news, sentiment, and macro data are not part of the current experiment path.
- `TARGET` is preserved in preprocessing but is not the default supervised target in this version.
- A fresh full experiment rerun is still required to regenerate committed outputs under the new protocol.

## Kaggle Credentials

The downloader supports the standard Kaggle API setup:

1. Create and activate the environment.
2. Provide either:
   - `KAGGLE_USERNAME` and `KAGGLE_KEY`, or
   - `~/.kaggle/kaggle.json`
3. Run `download-data`.

If credentials are missing, extract the dataset manually under `data/raw/stock-market-prediction/` and continue with `prepare-data`.
