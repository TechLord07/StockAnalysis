# Stock Price Forecasting with LSTM

[![CI](https://github.com/TechLord07/StockAnalysis/actions/workflows/ci.yml/badge.svg)](https://github.com/TechLord07/StockAnalysis/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular, testable pipeline for forecasting stock closing/opening prices with a
stacked LSTM neural network. Built to demonstrate a clean separation between
data ingestion, preprocessing, modeling, evaluation, and visualization —
rather than one monolithic script.

![Sample prediction](docs/sample_prediction.png)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Testing & CI](#testing--ci)
- [Design Decisions](#design-decisions)
- [Limitations & Disclaimer](#limitations--disclaimer)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

This project trains a multi-layer LSTM (Long Short-Term Memory) recurrent
neural network to forecast the next day's price from a rolling 60-day
window of historical prices. It ships with a working example on NSE Tata
Global stock data, and can be pointed at:

- any local or remote CSV with a `Date`/price column, or
- a live ticker (e.g. `AAPL`, `TSLA`) via the optional `yfinance` integration.

**What this project demonstrates:**

- Turning an exploratory script into a maintainable, package-structured
  codebase (`src/` modules with single responsibilities)
- A configurable, reproducible training pipeline (CLI args, seeding, saved
  configs)
- Proper train/test separation and inverse-transform handling for scaled
  time series
- Meaningful evaluation beyond a single metric (RMSE, MAE, MAPE)
- Unit tests for the pure-logic components (preprocessing, metrics) that run
  without a GPU or network access
- Continuous integration via GitHub Actions

## Architecture

```
                 ┌──────────────┐
                 │  Data Source │  CSV file / URL / yfinance ticker
                 └──────┬───────┘
                        ▼
                ┌───────────────┐
                │ data_loader.py│  load + validate raw OHLCV data
                └──────┬────────┘
                        ▼
                ┌────────────────┐
                │preprocessing.py│  MinMax scaling, windowing (X, y)
                └──────┬─────────┘
                        ▼
                  ┌───────────┐
                  │ model.py  │  stacked LSTM (configurable depth/width)
                  └─────┬─────┘
                        ▼
                 ┌─────────────┐
                 │  main.py    │  training loop, callbacks, orchestration
                 └──────┬──────┘
                        ▼
           ┌────────────┴─────────────┐
           ▼                          ▼
    ┌─────────────┐           ┌───────────────┐
    │ evaluate.py │           │ visualize.py  │
    │ RMSE/MAE/   │           │ loss curve +  │
    │ MAPE        │           │ pred. plot    │
    └─────────────┘           └───────────────┘
```

## Project Structure

```
stock-price-forecasting/
├── main.py                     # CLI entry point: orchestrates the full pipeline
├── src/
│   ├── config.py                # Central Config dataclass (all hyperparameters)
│   ├── data_loader.py           # CSV / URL / yfinance ingestion + validation
│   ├── preprocessing.py         # Scaling and sequence windowing
│   ├── model.py                 # LSTM architecture definition
│   ├── evaluate.py              # RMSE / MAE / MAPE metrics
│   └── visualize.py             # Loss curve and prediction plots (saved, not shown)
├── tests/
│   ├── test_preprocessing.py    # Unit tests: scaling, windowing, edge cases
│   └── test_evaluate.py         # Unit tests: metric correctness
├── .github/workflows/ci.yml     # Lint + test on every push/PR
├── outputs/                     # Generated plots and metrics (git-ignored)
├── saved_models/                # Trained model checkpoints (git-ignored)
├── requirements.txt
├── LICENSE
└── README.md
```

## Quickstart

```bash
# 1. Clone and enter the repo
git clone https://github.com/TechLord07/StockAnalysis.git
cd StockAnalysis

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the default pipeline (sample NSE Tata Global dataset)
python main.py
```

This will train the model, then write:

- `saved_models/lstm_stock_model.keras` — the trained model
- `outputs/training_loss.png` — training loss curve
- `outputs/predictions.png` — actual vs. predicted price plot
- `outputs/metrics.txt` — RMSE / MAE / MAPE on the test set

## Usage

**Train on your own CSV data:**

```bash
python main.py \
  --train-source data/my_stock_train.csv \
  --test-source data/my_stock_test.csv \
  --price-column Close \
  --epochs 50
```

**Train on a live ticker** (requires `pip install yfinance`):

```bash
python main.py --ticker AAPL --period 5y
```

**See all available options:**

```bash
python main.py --help
```

## Configuration

All hyperparameters live in [`src/config.py`](src/config.py) and can be
overridden via CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--sequence-length` | 60 | Look-back window (time steps) |
| `--lstm-units` | 50 | Units per LSTM layer |
| `--num-lstm-layers` | 4 | Number of stacked LSTM layers |
| `--dropout-rate` | 0.2 | Dropout after each LSTM layer |
| `--learning-rate` | 0.001 | Adam optimizer learning rate |
| `--epochs` | 100 | Training epochs (early stopping enabled) |
| `--batch-size` | 32 | Training batch size |
| `--seed` | 42 | Random seed for reproducibility |

## Results

Results below are from a run on the sample NSE Tata Global dataset using the
default configuration (60-day look-back window, 4 stacked LSTM layers, 50
units each).

| Metric | Value |
|---|---|
| MSE | 61.1888 |
| RMSE | 7.8223 |
| MAE | 6.1442 |
| MAPE | 2.72% |

The training loss curve below shows the model converging smoothly over ~80
epochs, with most of the improvement happening in the first 10 epochs before
early stopping kicks in.

![Training loss curve](docs/training_loss.png)

On the held-out test set, the model tracks the overall trend of the price
series closely, with a MAPE of under 3%. As expected for a model that only
sees past price history, it lags slightly at sharp reversals — visible around
the local peaks and troughs in the plot above — since it has no way to
anticipate a turning point before the price data itself shows one.

## Testing & CI

Unit tests cover the pure-logic modules (`preprocessing.py`, `evaluate.py`)
so they run in milliseconds without needing TensorFlow, a GPU, or network
access — the training pipeline itself is exercised separately as an
integration path.

```bash
pip install pytest
pytest tests/ -v
```

Every push and pull request runs these tests plus a `flake8` lint pass via
[GitHub Actions](.github/workflows/ci.yml).

## Design Decisions

- **No `plt.show()` in library code.** Plots are saved to `outputs/` so the
  pipeline runs unattended in CI, on a server, or in a script — not just
  interactively.
- **Config as a dataclass, not scattered globals.** One object to log,
  serialize, or diff between experiments.
- **`yfinance` is optional, not a hard dependency.** The core pipeline
  should not require a live network call to a third-party API to be
  testable or runnable offline against local data.
- **Metrics beyond RMSE.** MAPE expresses error as a percentage of price,
  which is more interpretable across different stocks/price ranges.

## Limitations & Disclaimer

This project is for **educational and portfolio purposes only** and is
**not intended for real trading decisions**. In particular:

- The model uses price history alone — no volume, fundamentals, or news
  sentiment — so it cannot anticipate the kind of external shocks that
  actually move markets.
- Financial time series are notoriously close to a random walk; a model
  that tracks recent trend well on a backtest is not the same as a model
  with real predictive edge.
- Past performance shown in `outputs/` is not indicative of future results.

## Roadmap

- [ ] Add exogenous features (trading volume, technical indicators like RSI/MACD)
- [ ] Walk-forward (rolling-origin) cross-validation instead of a single train/test split
- [ ] Compare against simpler baselines (ARIMA, naive last-value forecast)
- [ ] Package as a small REST API (FastAPI) for on-demand predictions
- [ ] Experiment tracking (e.g. MLflow or Weights & Biases)

## License

Released under the [MIT License](LICENSE).
