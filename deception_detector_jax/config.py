"""Configuration dataclasses for DeceptionDetector-JAX."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the tiny transformer model."""

    # Architecture
    seq_len: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    vocab_size: int = 128
    d_ff: int = 256  # Feed-forward hidden dimension

    # Training
    dropout_rate: float = 0.1
    learning_rate: float = 1e-3

    # Interpretability
    collect_intermediates: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"


@dataclass
class DatasetConfig:
    """Configuration for deception task datasets."""

    # Dataset size
    num_train: int = 10000
    num_val: int = 1000
    num_test: int = 1000

    # Task-specific
    task_name: str = "hidden_check"  # "hidden_check", "secret_goal", "concealed_step"
    seq_len: int = 32
    vocab_size: int = 128

    # Data generation
    seed: Optional[int] = 42

    # Deception parameters
    deception_rate: float = 0.3  # Fraction of examples with deceptive behavior


@dataclass
class TrainingConfig:
    """Configuration for training loops."""

    batch_size: int = 64
    num_epochs: int = 20
    eval_every: int = 500  # Steps
    log_every: int = 100  # Steps

    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 1000  # Steps

    # Misc
    seed: int = 42
