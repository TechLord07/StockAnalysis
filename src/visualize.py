"""
Plotting utilities.

Plots are saved to disk instead of calling `plt.show()`, which blocks
execution and breaks any non-interactive run (CI, servers, scripts).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    title: str = "Stock Price Prediction",
) -> Path:
    """Plot actual vs. predicted prices and save to `output_path`."""
    plt.figure(figsize=(10, 5))
    plt.plot(actual, color="black", label="Actual Price")
    plt.plot(predicted, color="tab:green", label="Predicted Price")
    plt.title(title)
    plt.xlabel("Time (trading days)")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_training_history(history, output_path: Path) -> Path:
    """Plot training/validation loss curves and save to `output_path`."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history.get("loss", []), label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
