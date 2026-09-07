import numpy as np

from src.evaluate import compute_metrics


def test_compute_metrics_perfect_prediction():
    y_true = np.array([100.0, 105.0, 110.0])
    y_pred = np.array([100.0, 105.0, 110.0])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics.mse == 0.0
    assert metrics.rmse == 0.0
    assert metrics.mae == 0.0
    assert metrics.mape == 0.0


def test_compute_metrics_known_error():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])  # off by 10 and -20

    metrics = compute_metrics(y_true, y_pred)

    assert metrics.mae == 15.0  # (10 + 20) / 2
    # |100-110|/100 = 10%, |200-180|/200 = 10% -> average = 10%
    assert round(metrics.mape, 2) == 10.0
