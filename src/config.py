"""
Configuration module for churn prediction pipeline.

Centralizes all paths, temporal windows, and hyperparameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class PathConfig:
    """File system paths for data and artifacts."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    raw_data: Path = field(init=False)
    processed_data: Path = field(init=False)
    models_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'raw_data', self.project_root / 'events.csv')
        object.__setattr__(self, 'processed_data', self.project_root / 'data' / 'customers.parquet')
        object.__setattr__(self, 'models_dir', self.project_root / 'artifacts')

        self.processed_data.parent.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TemporalConfig:
    """Temporal windowing parameters to prevent data leakage."""

    prediction_window_days: int = 30
    observation_window_days: int = 28
    velocity_short_window_days: int = 7

    @property
    def velocity_multiplier(self) -> float:
        """Normalizes L7 to L28 scale for velocity ratio."""
        return self.observation_window_days / self.velocity_short_window_days


@dataclass(frozen=True)
class RandomForestConfig:
    """Hyperparameters for Random Forest baseline."""

    n_estimators: int = 200
    max_depth: int = 10
    min_samples_split: int = 5
    random_state: int = 42
    class_weight: str = 'balanced'
    n_jobs: int = -1

    def to_sklearn_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state,
            'class_weight': self.class_weight,
            'n_jobs': self.n_jobs,
        }


@dataclass(frozen=True)
class MLPConfig:
    """Hyperparameters for PyTorch MLP."""

    hidden_dims: tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 100
    early_stopping_patience: int = 10
    random_state: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    """Training and evaluation parameters."""

    test_size: float = 0.2
    random_state: int = 42
    smoothing_alpha: float = 1.0


@dataclass(frozen=True)
class Config:
    """Master configuration aggregating all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    rf: RandomForestConfig = field(default_factory=RandomForestConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
