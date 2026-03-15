# Stock Prediction

Research-oriented stock price prediction project aligned with the proposal in [report.md](/Users/david/codespace/stock_prediction/report.md).

## Scope

- Predict next-day close price for each stock independently.
- Provide a reproducible pipeline for download, preparation, training, evaluation, walk-forward validation, and experiment reporting.
- Include `linear_regression`, `linear_regression_scaled`, `arima`, `lstm`, `gru`, and `arima_residual_lstm`.
- Keep the current model family focused on the proposal scope; Transformer/TFT/GARCH are not part of this version.

## Quick Start

```bash
conda env create -f environment.yml
conda activate stock-prediction
stock-prediction download-data --config configs/default.yaml
stock-prediction prepare-data --config configs/default.yaml
stock-prediction run-experiment --config configs/default.yaml
```

If you are running the project from source without installing the package, use `PYTHONPATH=src` with `conda run -n stock-prediction ...`.

## Current Workflow

The current pipeline is:

1. `download-data`
   - Download or unpack the Kaggle dataset into `data/raw/stock-market-prediction/`.
2. `prepare-data`
   - Infer schema aliases for date / OHLCV / symbol columns.
   - Standardize the core columns to `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`.
   - Build the configured target column, which is currently `target_next_close`.
   - Add derived return / spread / volatility features and write feature-group metadata into `data/interim/schema.json`.
3. `train` / `evaluate` / `run-experiment`
   - Split each stock chronologically into train / validation / test.
   - Resolve feature columns from the configured `feature_set`.
   - Run holdout evaluation and, if enabled, expanding walk-forward evaluation.
   - Save metrics, predictions, summaries, conclusions, and training logs under `outputs/`.

## Data Processing

The prepared dataset keeps the source indicators and adds a lightweight set of derived features:

- Core price features: `open`, `high`, `low`, `close`, `volume`
- Derived features: 1-day / 5-day returns, intraday return, high-low spread, rolling volatility, rolling mean return
- Feature groups:
  - `price_basic`
  - `technical_indicators`
  - `returns_volatility`
  - `all_numeric_filtered`

By default, the configuration uses `feature_set: auto`:

- linear models and ARIMA use a price-oriented feature set
- recurrent models default to a return / volatility oriented feature set

## What This Version Adds

- Config-driven target column and feature-set selection.
- Validation-aware training for recurrent models with early stopping and best-epoch refit.
- ARIMA order selection on the validation split.
- Holdout and expanding walk-forward evaluation for every configured model.
- Richer outputs under `outputs/metrics/` including training logs and conclusion summaries.

## Models

The repository currently supports these model entries:

- `linear_regression`
  - Plain linear baseline on tabular features
- `linear_regression_scaled`
  - Linear baseline with feature scaling as a control line
- `arima`
  - Univariate ARIMA with optional validation-based order selection
- `lstm`
  - Recurrent model with validation monitoring, early stopping, and best-epoch refit
- `gru`
  - Recurrent model with the same training protocol as LSTM
- `arima_residual_lstm`
  - Hybrid model that fits ARIMA first and then models residual dynamics with an LSTM

## Outputs

Main output locations:

- `outputs/predictions/`
  - Per-model prediction CSV files with `evaluation`, `step`, and `split_start_date`
- `outputs/metrics/*_metrics.csv`
  - Per-symbol metrics for holdout and, when enabled, walk-forward
- `outputs/metrics/*_summary.md`
  - Aggregated summary tables
- `outputs/metrics/*_conclusion.md`
  - Short conclusion pages for direct interpretation
- `outputs/metrics/*_training_log.csv`
  - Training or selection logs for models that use validation

## Current Results Snapshot

The committed result files under `outputs/metrics/` are from the earlier full run that existed before the vNext refactor. They still show the previous overall pattern:

- `linear_regression`
  - mean RMSE about `1.0977`
- `arima`
  - mean RMSE about `5.58`
- `arima_residual_lstm`
  - mean RMSE about `5.5802`
- `lstm`
  - mean RMSE about `28.3299`
- `gru`
  - mean RMSE about `28.3404`

Important note:

- those committed metrics are useful as a historical baseline
- they are **not** yet a full rerun of the new validation-aware vNext pipeline
- after the refactor, the new code path has only been smoke-tested so far

Smoke tests completed in the `stock-prediction` conda environment:

- holdout path: `linear_regression`, `arima`, `lstm`, `arima_residual_lstm`
- walk-forward path: `linear_regression`, `lstm`

## Current Gaps

- No external macro, news, or sentiment data.
- No Transformer/TFT/GARCH implementation yet.
- The default dataset still mixes many precomputed indicators from the source data, so feature ablation should be interpreted carefully.
- The repository still needs a fresh full experiment rerun to regenerate `outputs/metrics/` under the new protocol.

## Kaggle Credentials

The downloader supports the standard Kaggle API setup:

1. Create and activate the conda environment.
2. Provide either:
   - `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables, or
   - `~/.kaggle/kaggle.json`
3. Run `download-data`.

If credentials are missing, place the extracted dataset manually under `data/raw/stock-market-prediction/` and continue with `prepare-data`.
