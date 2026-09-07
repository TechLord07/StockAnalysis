"""
Model architecture: a stacked LSTM regressor.

Kept configurable (depth, width, dropout, learning rate) via `Config` so the
architecture isn't hardcoded, which makes it easy to run ablations.
"""
from __future__ import annotations

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.config import Config


def build_lstm_model(input_shape: tuple, config: Config) -> Sequential:
    """Build and compile a stacked LSTM model for univariate time-series regression."""
    model = Sequential(name="lstm_stock_forecaster")

    for layer_idx in range(config.num_lstm_layers):
        is_last_lstm = layer_idx == config.num_lstm_layers - 1
        kwargs = {"units": config.lstm_units, "return_sequences": not is_last_lstm}
        if layer_idx == 0:
            kwargs["input_shape"] = input_shape
        model.add(LSTM(**kwargs))
        model.add(Dropout(config.dropout_rate))

    model.add(Dense(units=1))

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="mean_squared_error",
    )
    return model
