"""
Centralized configuration for the forecasting pipeline.

Keeping every tunable knob in one dataclass makes experiments reproducible:
you can log a Config instance, diff two runs, or serialize it to JSON/YAML
alongside a saved model so you always know exactly what produced it.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- Data -----------------------------------------------------------
    train_source: str = (
        "https://raw.githubusercontent.com/mwitiderrick/stockprice/"
        "master/NSE-TATAGLOBAL.csv"
    )
    test_source: str = (
        "https://raw.githubusercontent.com/mwitiderrick/stockprice/"
        "master/tatatest.csv"
    )
    price_column: str = "Open"
    sequence_length: int = 60  # look-back window (trading days)

    # --- Model ------------------------------------------------------------
    lstm_units: int = 50
    num_lstm_layers: int = 4
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3

    # --- Training ---------------------------------------------------------
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    random_seed: int = 42

    # --- Paths --------------------------------------------------------
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    model_dir: Path = field(default_factory=lambda: Path("saved_models"))
    model_name: str = "lstm_stock_model.keras"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.model_dir = Path(self.model_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_name
