"""
Feature engineering and windowing utilities.

The core idea for an LSTM price forecaster: scale the series to [0, 1],
then convert it into overlapping (X, y) windows where X is the previous
`sequence_length` observations and y is the next one.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale_series(values: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """Fit a MinMaxScaler on a 1D array and return the scaled values + scaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = values.reshape(-1, 1)
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def create_sequences(scaled_values: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a scaled 1D series into supervised-learning windows.

    Returns X of shape (n_samples, sequence_length, 1) and y of shape (n_samples,).
    """
    if len(scaled_values) <= sequence_length:
        raise ValueError(
            f"Series length ({len(scaled_values)}) must exceed sequence_length "
            f"({sequence_length}) to create at least one training window."
        )

    X, y = [], []
    for i in range(sequence_length, len(scaled_values)):
        X.append(scaled_values[i - sequence_length:i, 0])
        y.append(scaled_values[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


def prepare_test_inputs(
    train_series: np.ndarray,
    test_series: np.ndarray,
    scaler: MinMaxScaler,
    sequence_length: int,
) -> np.ndarray:
    """
    Build the input windows needed to predict the test period, using the
    tail of the training series to seed the first window (mirrors how
    you'd forecast forward in production, where history precedes the
    prediction horizon).
    """
    combined = np.concatenate([train_series, test_series])
    inputs = combined[len(combined) - len(test_series) - sequence_length:]
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(sequence_length, len(inputs)):
        X_test.append(inputs[i - sequence_length:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test
