"""
Evaluation metrics for the forecasting model.

RMSE alone (the original script's only metric) doesn't tell you much in
isolation -- MAE and MAPE make the error interpretable in the price's own
units and as a percentage, which is what you actually want when judging
"is this forecast good enough to act on".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class Metrics:
    mse: float
    rmse: float
    mae: float
    mape: float

    def __str__(self) -> str:
        return (
            f"MSE:  {self.mse:.4f}\n"
            f"RMSE: {self.rmse:.4f}\n"
            f"MAE:  {self.mae:.4f}\n"
            f"MAPE: {self.mape:.2f}%"
        )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Compute MSE, RMSE, MAE, and MAPE between true and predicted prices."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return Metrics(mse=mse, rmse=rmse, mae=mae, mape=mape)
