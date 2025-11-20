"""Utilities for loading models, data, and results."""

import json
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
from deception_detector_jax.config import ModelConfig
from deception_detector_jax.models.tiny_transformer import TinyTransformer
from deception_detector_jax.data.deception_tasks import load_dataset


def get_project_root():
    """Get project root directory."""
    return PROJECT_ROOT


def get_available_models():
    """Get list of available trained models."""
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    if not checkpoint_dir.exists():
        return []

    models = []
    for task_dir in checkpoint_dir.iterdir():
        if task_dir.is_dir() and (task_dir / "final_params.npy").exists():
            models.append(task_dir.name)

    return sorted(models)


def load_model(task_name, config=None):
    """
    Load a trained model.

    Args:
        task_name: Name of the task (e.g., "hidden_check")
        config: Optional ModelConfig. If None, uses defaults.

    Returns:
        model, params
    """
    if config is None:
        config = ModelConfig(
            seq_len=32,
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=128,
            collect_intermediates=True,
        )

    # Load parameters
    checkpoint_path = PROJECT_ROOT / "checkpoints" / task_name / "final_params.npy"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    params = np.load(checkpoint_path, allow_pickle=True).item()

    # Create model
    model = TinyTransformer(config)

    return model, params


def load_task_data(task_name, split="test"):
    """
    Load dataset for a task.

    Args:
        task_name: Name of the task
        split: "train", "val", or "test"

    Returns:
        Dataset dictionary
    """
    data_path = PROJECT_ROOT / "data" / task_name / f"{split}.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    return load_dataset(str(data_path))


def load_benchmark_results(task_name=None):
    """
    Load benchmark results.

    Args:
        task_name: Optional specific task. If None, loads all.

    Returns:
        Dict of results or single result dict
    """
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        return {} if task_name is None else None

    if task_name is not None:
        result_path = results_dir / f"benchmark_{task_name}.json"
        if not result_path.exists():
            return None
        with open(result_path) as f:
            return json.load(f)
    else:
        results = {}
        for result_file in results_dir.glob("benchmark_*.json"):
            task = result_file.stem.replace("benchmark_", "")
            with open(result_file) as f:
                results[task] = json.load(f)
        return results


def load_training_history(task_name):
    """Load training history for a task."""
    history_path = PROJECT_ROOT / "checkpoints" / task_name / "history.npz"
    if not history_path.exists():
        return None

    history_data = np.load(history_path)
    return {key: history_data[key] for key in history_data.keys()}


def run_model_inference(model, params, input_ids, return_cache=False):
    """
    Run model inference.

    Args:
        model: TinyTransformer model
        params: Model parameters
        input_ids: Input token IDs (batch, seq_len)
        return_cache: Whether to return activation cache

    Returns:
        logits, cache (if return_cache=True)
    """
    input_ids = jnp.array(input_ids)

    logits, cache = model.apply(
        {"params": params},
        input_ids,
        deterministic=True,
        return_cache=return_cache,
    )

    return logits, cache


def get_task_metadata(task_name):
    """Get metadata about a task."""
    metadata = {
        "hidden_check": {
            "description": "Model receives A, B, CHECK_FLAG. Should output max(A, B).",
            "deception_type": "Monitor-dependent behavior",
            "hidden_variable": "forbidden",
            "color": "#FF6B6B",
        },
        "secret_goal": {
            "description": "Visible: maximize A. Hidden: maximize B.",
            "deception_type": "Hidden objective",
            "hidden_variable": "deceptive",
            "color": "#4ECDC4",
        },
        "concealed_step": {
            "description": "Hidden computation h(x)=x+5. Output: f(x)=h(x)+g(x).",
            "deception_type": "Concealed reasoning",
            "hidden_variable": "concealed",
            "color": "#95E1D3",
        },
    }
    return metadata.get(task_name, {})
