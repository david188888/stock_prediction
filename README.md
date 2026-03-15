# Stock Prediction

Research-oriented stock price prediction project aligned with the proposal in [report.md](/Users/david/codespace/stock_prediction/report.md).

## Scope

- Predict next-day close price for each stock independently.
- Provide a reproducible pipeline for download, preparation, training, evaluation, and experiment reporting.
- Include linear regression, ARIMA, LSTM, GRU, and ARIMA + LSTM residual hybrid models.

## Quick Start

```bash
conda env create -f environment.yml
conda activate stock-prediction
pip install -e .
stock-prediction download-data --config configs/default.yaml
stock-prediction prepare-data --config configs/default.yaml
stock-prediction run-experiment --config configs/default.yaml
```

## Kaggle Credentials

The downloader supports the standard Kaggle API setup:

1. Create and activate the conda environment.
2. Provide either:
   - `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables, or
   - `~/.kaggle/kaggle.json`
3. Run `download-data`.

If credentials are missing, place the extracted dataset manually under `data/raw/stock-market-prediction/` and continue with `prepare-data`.
