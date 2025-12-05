"""
configuration for time-aware transformer encoder.
this is a discriminative approach that learns dense user representations
rather than attempting to generate future events.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EncoderConfig:
    """
    configuration for the time-aware bert-style encoder.

    this architecture learns a latent representation h_u ∈ R^d for each customer
    by encoding their event sequence through a transformer, then applies a simple
    classifier on top. this is more robust than generative models on small datasets.
    """

    # vocabulary and dimensions
    num_event_types: int = 9  # number of discrete event types
    embedding_dim: int = 128  # d_model for transformer
    max_seq_length: int = 64  # fixed sequence length (pad/truncate)

    # transformer architecture
    num_layers: int = 2  # shallow is better for small data
    num_heads: int = 4  # attention heads
    feedforward_dim: int = 512  # ffn hidden dimension
    dropout: float = 0.1

    # use pre-norm architecture (layer norm before attention)
    # more stable gradients on small datasets
    pre_norm: bool = True

    # time encoding
    time_encoding_dim: int = 32  # sinusoidal features for time

    # value projection
    value_projection_dim: int = 16  # project continuous values

    # pooling strategy
    pooling: str = "mean"  # mean pooling across sequence (robust)

    # focal loss parameters (critical for imbalanced data)
    focal_gamma: float = 2.0  # focusing parameter
    focal_alpha: Optional[float] = None  # class weight (auto-computed)

    # training
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 7  # early stopping patience

    # data
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self):
        assert self.embedding_dim % self.num_heads == 0
        assert self.pooling in ["mean", "max", "cls"]
