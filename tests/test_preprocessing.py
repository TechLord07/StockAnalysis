import numpy as np
import pytest

from src.preprocessing import create_sequences, prepare_test_inputs, scale_series


def test_scale_series_range():
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    scaled, scaler = scale_series(values)
    assert scaled.min() == pytest.approx(0.0)
    assert scaled.max() == pytest.approx(1.0)
    # inverse transform should recover the original values
    recovered = scaler.inverse_transform(scaled).flatten()
    np.testing.assert_allclose(recovered, values)


def test_create_sequences_shapes():
    scaled = np.linspace(0, 1, 100).reshape(-1, 1)
    sequence_length = 10
    X, y = create_sequences(scaled, sequence_length)

    expected_samples = len(scaled) - sequence_length
    assert X.shape == (expected_samples, sequence_length, 1)
    assert y.shape == (expected_samples,)


def test_create_sequences_raises_on_short_series():
    scaled = np.linspace(0, 1, 5).reshape(-1, 1)
    with pytest.raises(ValueError):
        create_sequences(scaled, sequence_length=10)


def test_prepare_test_inputs_shape():
    train_series = np.linspace(1, 100, 100)
    test_series = np.linspace(101, 120, 20)
    _, scaler = scale_series(train_series)

    X_test = prepare_test_inputs(train_series, test_series, scaler, sequence_length=60)
    assert X_test.shape == (len(test_series), 60, 1)
