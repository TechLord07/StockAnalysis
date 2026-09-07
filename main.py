#!/usr/bin/env python3
"""
Stock Price Forecasting -- CLI entry point.

Trains a stacked LSTM on a historical price series and evaluates it on a
held-out test period, saving the model, metrics, and plots to disk.

Examples
--------
Run with the bundled sample dataset (NSE Tata Global):
    $ python main.py

Run with your own CSVs:
    $ python main.py --train-source path/to/train.csv --test-source path/to/test.csv \\
                      --price-column Close --epochs 50

Run against a live ticker via yfinance (requires `pip install yfinance`):
    $ python main.py --ticker AAPL --period 5y
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.config import Config
from src.data_loader import load_csv, load_ticker, validate_columns
from src.evaluate import compute_metrics
from src.model import build_lstm_model
from src.preprocessing import create_sequences, prepare_test_inputs, scale_series
from src.visualize import plot_predictions, plot_training_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an LSTM stock price forecaster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default = Config()

    parser.add_argument("--ticker", type=str, default=None,
                         help="Live ticker symbol to fetch via yfinance (overrides --train-source/--test-source).")
    parser.add_argument("--period", type=str, default="5y",
                         help="History period for --ticker (yfinance format, e.g. 1y, 5y, max).")
    parser.add_argument("--train-source", type=str, default=default.train_source,
                         help="Path or URL to the training CSV.")
    parser.add_argument("--test-source", type=str, default=default.test_source,
                         help="Path or URL to the test CSV.")
    parser.add_argument("--price-column", type=str, default=default.price_column,
                         help="Column name to forecast.")
    parser.add_argument("--sequence-length", type=int, default=default.sequence_length,
                         help="Look-back window size, in time steps.")
    parser.add_argument("--lstm-units", type=int, default=default.lstm_units)
    parser.add_argument("--num-lstm-layers", type=int, default=default.num_lstm_layers)
    parser.add_argument("--dropout-rate", type=float, default=default.dropout_rate)
    parser.add_argument("--learning-rate", type=float, default=default.learning_rate)
    parser.add_argument("--epochs", type=int, default=default.epochs)
    parser.add_argument("--batch-size", type=int, default=default.batch_size)
    parser.add_argument("--seed", type=int, default=default.random_seed)
    parser.add_argument("--output-dir", type=str, default=str(default.output_dir))
    parser.add_argument("--model-dir", type=str, default=str(default.model_dir))

    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> int:
    args = parse_args()
    config = Config(
        train_source=args.train_source,
        test_source=args.test_source,
        price_column=args.price_column,
        sequence_length=args.sequence_length,
        lstm_units=args.lstm_units,
        num_lstm_layers=args.num_lstm_layers,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.seed,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )
    set_seed(config.random_seed)

    # 1. Load data -----------------------------------------------------
    if args.ticker:
        full_df = load_ticker(args.ticker, period=args.period)
        validate_columns(full_df, config.price_column)
        split_idx = int(len(full_df) * 0.9)
        train_df, test_df = full_df.iloc[:split_idx], full_df.iloc[split_idx:]
    else:
        train_df = load_csv(config.train_source)
        test_df = load_csv(config.test_source)
        validate_columns(train_df, config.price_column)
        validate_columns(test_df, config.price_column)

    train_series = train_df[config.price_column].values.astype(float)
    test_series = test_df[config.price_column].values.astype(float)
    logger.info("Train samples: %d | Test samples: %d", len(train_series), len(test_series))

    # 2. Preprocess ------------------------------------------------------
    scaled_train, scaler = scale_series(train_series)
    X_train, y_train = create_sequences(scaled_train, config.sequence_length)
    X_test = prepare_test_inputs(train_series, test_series, scaler, config.sequence_length)

    # 3. Build & train model ----------------------------------------------
    model = build_lstm_model(input_shape=(X_train.shape[1], 1), config=config)
    model.summary(print_fn=logger.info)

    callbacks = [
        EarlyStopping(monitor="loss", patience=config.early_stopping_patience, restore_best_weights=True),
        ModelCheckpoint(filepath=str(config.model_path), monitor="loss", save_best_only=True),
    ]
    history = model.fit(
        X_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    plot_training_history(history, config.output_dir / "training_loss.png")

    # 4. Predict & evaluate ------------------------------------------------
    predicted_scaled = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_scaled)

    metrics = compute_metrics(test_series, predicted_prices)
    logger.info("Evaluation on test set:\n%s", metrics)

    plot_predictions(
        actual=test_series,
        predicted=predicted_prices,
        output_path=config.output_dir / "predictions.png",
        title=f"{args.ticker or config.price_column} Price Prediction",
    )

    metrics_path = config.output_dir / "metrics.txt"
    metrics_path.write_text(str(metrics))
    logger.info("Saved model to %s", config.model_path)
    logger.info("Saved plots and metrics to %s", config.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
