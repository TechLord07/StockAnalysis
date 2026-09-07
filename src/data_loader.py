"""
Data acquisition layer.

Supports three interchangeable sources for the same interface:
  1. A local CSV path
  2. A remote CSV URL (e.g. the sample NSE Tata Global dataset)
  3. A live ticker pulled via `yfinance` (optional extra, install with
     `pip install yfinance` -- kept optional so the core pipeline never
     depends on network access to a third-party market-data API).

All three return a pandas DataFrame with at least a `Date` and the
configured price column, sorted ascending by date.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(source: str) -> pd.DataFrame:
    """Load a CSV from a local path or a URL into a sorted DataFrame."""
    logger.info("Loading data from %s", source)
    df = pd.read_csv(source)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    return df


def load_ticker(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a live ticker using yfinance.

    Raises ImportError with a helpful message if yfinance isn't installed,
    rather than making it a hard dependency of the whole project.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "yfinance is required for live ticker downloads. "
            "Install it with: pip install yfinance"
        ) from exc

    logger.info("Downloading %s (%s, interval=%s) via yfinance", symbol, period, interval)
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df = df.reset_index().rename(columns={"index": "Date"})
    return df


def validate_columns(df: pd.DataFrame, price_column: str) -> None:
    """Fail fast with a clear message if the expected column is missing."""
    if price_column not in df.columns:
        raise KeyError(
            f"Expected column '{price_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if df[price_column].isnull().any():
        n_missing = int(df[price_column].isnull().sum())
        logger.warning(
            "%d missing values found in '%s'; forward-filling.", n_missing, price_column
        )
        df[price_column] = df[price_column].ffill().bfill()
