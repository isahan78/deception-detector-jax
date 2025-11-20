#!/usr/bin/env python3
"""Train tiny transformer on deception tasks."""

import argparse
import sys
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deception_detector_jax.config import ModelConfig, TrainingConfig
from deception_detector_jax.models.tiny_transformer import TinyTransformer, init_model
from deception_detector_jax.data.deception_tasks import load_dataset


class TrainState(train_state.TrainState):
    """Extended train state with additional metrics."""
    pass


def create_train_state(rng, config: ModelConfig, learning_rate: float):
    """Create initial training state."""
    model, params = init_model(config, rng)
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx), model


def compute_loss(params, model, batch, deterministic=True, rng=None):
    """
    Compute cross-entropy loss.

    Args:
        params: Model parameters
        model: TinyTransformer model
        batch: Dict with input_ids and target_ids
        deterministic: Whether to use dropout
        rng: Random key for dropout

    Returns:
        loss: Scalar loss
    """
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]

    # Forward pass
    if deterministic:
        logits, _ = model.apply(
            {"params": params},
            input_ids,
            deterministic=deterministic,
            return_cache=False,
        )
    else:
        logits, _ = model.apply(
            {"params": params},
            input_ids,
            deterministic=deterministic,
            return_cache=False,
            rngs={"dropout": rng},
        )

    # Compute cross-entropy loss
    # Mask for valid targets (non-zero)
    mask = target_ids > 0

    # Log probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Gather target log probs
    batch_size, seq_len = target_ids.shape
    batch_idx = jnp.arange(batch_size)[:, None]
    seq_idx = jnp.arange(seq_len)[None, :]
    target_log_probs = log_probs[batch_idx, seq_idx, target_ids]

    # Average over valid positions (add epsilon to avoid division by zero)
    loss = -jnp.sum(target_log_probs * mask) / (jnp.sum(mask) + 1e-8)

    return loss


def compute_accuracy(params, model, batch):
    """Compute prediction accuracy."""
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]

    # Forward pass
    logits, _ = model.apply(
        {"params": params},
        input_ids,
        deterministic=True,
        return_cache=False,
    )

    # Get predictions
    predictions = jnp.argmax(logits, axis=-1)

    # Compute accuracy on valid positions
    mask = target_ids > 0

    correct = (predictions == target_ids) * mask
    accuracy = jnp.sum(correct) / (jnp.sum(mask) + 1e-8)

    return accuracy


@jax.jit
def train_step(state, batch, rng):
    """Single training step."""

    def loss_fn(params):
        return compute_loss(params, state.apply_fn.__self__, batch, deterministic=False, rng=rng)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


@jax.jit
def eval_step(state, batch):
    """Single evaluation step."""
    loss = compute_loss(state.params, state.apply_fn.__self__, batch, deterministic=True)
    accuracy = compute_accuracy(state.params, state.apply_fn.__self__, batch)
    return loss, accuracy


def create_batches(data, batch_size, rng):
    """Create batches from dataset."""
    n_samples = len(data["input_ids"])
    indices = jax.random.permutation(rng, n_samples)

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield {k: v[batch_indices] for k, v in data.items()}


def train_epoch(state, train_data, batch_size, rng):
    """Train for one epoch."""
    epoch_loss = 0.0
    num_batches = 0

    batch_rng, dropout_rng = jax.random.split(rng)

    for batch in create_batches(train_data, batch_size, batch_rng):
        # Convert to JAX arrays
        batch = {k: jnp.array(v) for k, v in batch.items()}

        # Split RNG for this batch's dropout
        dropout_rng, step_rng = jax.random.split(dropout_rng)
        state, loss = train_step(state, batch, step_rng)
        epoch_loss += loss
        num_batches += 1

    return state, epoch_loss / num_batches


def evaluate(state, eval_data, batch_size):
    """Evaluate on validation/test set."""
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    # Use fixed seed for consistent batching
    rng = jax.random.PRNGKey(0)

    for batch in create_batches(eval_data, batch_size, rng):
        batch = {k: jnp.array(v) for k, v in batch.items()}
        loss, accuracy = eval_step(state, batch)
        total_loss += loss
        total_accuracy += accuracy
        num_batches += 1

    return total_loss / num_batches, total_accuracy / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train tiny transformer")

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing train/val/test .npz files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=64,
        help="Model dimension",
    )

    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads",
    )

    parser.add_argument(
        "--n-layers",
        type=int,
        default=2,
        help="Number of transformer layers",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Load datasets
    data_dir = Path(args.data_dir)
    print(f"Loading data from {data_dir}...")

    train_data = load_dataset(str(data_dir / "train.npz"))
    val_data = load_dataset(str(data_dir / "val.npz"))

    print(f"  Train size: {len(train_data['input_ids'])}")
    print(f"  Val size: {len(val_data['input_ids'])}")

    # Create model config
    model_config = ModelConfig(
        seq_len=train_data["input_ids"].shape[1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        vocab_size=128,  # TODO: infer from data
        collect_intermediates=False,  # Disable during training for speed
    )

    print(f"\nModel config:")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_heads: {model_config.n_heads}")
    print(f"  n_layers: {model_config.n_layers}")

    # Initialize model and training state
    rng = jax.random.PRNGKey(args.seed)
    init_rng, train_rng = jax.random.split(rng)

    print("\nInitializing model...")
    state, model = create_train_state(init_rng, model_config, args.learning_rate)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nTraining for {args.num_epochs} epochs...")
    print("=" * 60)

    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(args.num_epochs):
        epoch_start = time.time()

        # Split RNG for this epoch
        train_rng, epoch_rng = jax.random.split(train_rng)

        # Train
        state, train_loss = train_epoch(state, train_data, args.batch_size, epoch_rng)

        # Evaluate
        val_loss, val_acc = evaluate(state, val_data, args.batch_size)

        epoch_time = time.time() - epoch_start

        # Log
        print(
            f"Epoch {epoch + 1:3d}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save history
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # TODO: Save checkpoint
            # For now, just keep params in memory
            print(f"  → New best model (val_loss: {val_loss:.4f})")

    print("=" * 60)
    print("✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")

    # Save final model parameters
    params_path = output_dir / "final_params.npy"
    np.save(params_path, state.params)
    print(f"  Saved parameters to: {params_path}")

    # Save history
    history_path = output_dir / "history.npz"
    np.savez(history_path, **history)
    print(f"  Saved history to: {history_path}")


if __name__ == "__main__":
    main()
