"""Ablation utilities for testing component importance."""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .activation_cache import ActivationCache


class AblationHook:
    """
    Hook for ablating (zeroing or replacing) specific model components.

    In a full implementation, this would modify the forward pass.
    For now, provides utilities for post-hoc analysis.
    """

    def __init__(self, model, params):
        """
        Initialize ablation hook.

        Args:
            model: TinyTransformer model
            params: Model parameters
        """
        self.model = model
        self.params = params

    def ablate_attention_head(
        self,
        input_ids: jnp.ndarray,
        layer_idx: int,
        head_idx: int,
        ablation_type: str = "zero",
    ) -> tuple[jnp.ndarray, ActivationCache]:
        """
        Ablate a specific attention head.

        Args:
            input_ids: Input tokens
            layer_idx: Layer index
            head_idx: Head index
            ablation_type: "zero" or "mean"

        Returns:
            logits, cache after ablation
        """
        # TODO: Implement true head ablation
        # This requires modifying attention computation during forward pass
        # Options:
        # 1. Modify model to accept ablation mask
        # 2. Use JAX pytree manipulation
        # 3. Create modified parameters with zeroed head weights

        # For now, run normal forward pass
        logits, cache_dict = self.model.apply(
            {"params": self.params},
            input_ids,
            deterministic=True,
            return_cache=True,
        )
        cache = ActivationCache.from_model_output(cache_dict)

        return logits, cache

    def ablate_mlp_neurons(
        self,
        input_ids: jnp.ndarray,
        layer_idx: int,
        neuron_indices: List[int],
    ) -> tuple[jnp.ndarray, ActivationCache]:
        """
        Ablate specific MLP neurons.

        Args:
            input_ids: Input tokens
            layer_idx: Layer index
            neuron_indices: List of neuron indices to ablate

        Returns:
            logits, cache after ablation
        """
        # TODO: Implement neuron ablation
        logits, cache_dict = self.model.apply(
            {"params": self.params},
            input_ids,
            deterministic=True,
            return_cache=True,
        )
        cache = ActivationCache.from_model_output(cache_dict)

        return logits, cache

    def ablate_layer(
        self,
        input_ids: jnp.ndarray,
        layer_idx: int,
        component: str = "all",
    ) -> tuple[jnp.ndarray, ActivationCache]:
        """
        Ablate an entire layer.

        Args:
            input_ids: Input tokens
            layer_idx: Layer index
            component: "all", "attn", or "mlp"

        Returns:
            logits, cache after ablation
        """
        # TODO: Implement layer ablation
        logits, cache_dict = self.model.apply(
            {"params": self.params},
            input_ids,
            deterministic=True,
            return_cache=True,
        )
        cache = ActivationCache.from_model_output(cache_dict)

        return logits, cache


def sweep_head_ablations(
    model,
    params,
    input_ids: jnp.ndarray,
    target_ids: jnp.ndarray,
    n_layers: int,
    n_heads: int,
) -> np.ndarray:
    """
    Sweep through all attention heads and measure ablation impact.

    Args:
        model: TinyTransformer model
        params: Model parameters
        input_ids: Input tokens
        target_ids: Target tokens
        n_layers: Number of layers
        n_heads: Number of heads per layer

    Returns:
        Impact matrix of shape (n_layers, n_heads) with loss increase per head
    """
    # Baseline: no ablation
    baseline_logits, _ = model.apply(
        {"params": params},
        input_ids,
        deterministic=True,
        return_cache=False,
    )
    baseline_loss = compute_loss(baseline_logits, target_ids)

    # Sweep through heads
    impact_matrix = np.zeros((n_layers, n_heads))

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # TODO: Ablate this specific head
            # For now, use baseline (no actual ablation)
            ablated_logits = baseline_logits
            ablated_loss = compute_loss(ablated_logits, target_ids)

            # Record impact (loss increase)
            impact_matrix[layer_idx, head_idx] = ablated_loss - baseline_loss

    return impact_matrix


def sweep_layer_ablations(
    model,
    params,
    input_ids: jnp.ndarray,
    target_ids: jnp.ndarray,
    n_layers: int,
) -> Dict[str, np.ndarray]:
    """
    Sweep through all layers and measure ablation impact.

    Args:
        model: TinyTransformer model
        params: Model parameters
        input_ids: Input tokens
        target_ids: Target tokens
        n_layers: Number of layers

    Returns:
        Dict with impact arrays for attn, mlp, and all components
    """
    baseline_logits, _ = model.apply(
        {"params": params},
        input_ids,
        deterministic=True,
        return_cache=False,
    )
    baseline_loss = compute_loss(baseline_logits, target_ids)

    results = {
        "attn_impact": np.zeros(n_layers),
        "mlp_impact": np.zeros(n_layers),
        "all_impact": np.zeros(n_layers),
    }

    for layer_idx in range(n_layers):
        # TODO: Ablate attention
        # For now, use baseline
        attn_loss = baseline_loss
        results["attn_impact"][layer_idx] = attn_loss - baseline_loss

        # TODO: Ablate MLP
        mlp_loss = baseline_loss
        results["mlp_impact"][layer_idx] = mlp_loss - baseline_loss

        # TODO: Ablate entire layer
        all_loss = baseline_loss
        results["all_impact"][layer_idx] = all_loss - baseline_loss

    return results


def compute_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> float:
    """
    Compute cross-entropy loss.

    Args:
        logits: Model output logits (batch, seq_len, vocab_size)
        targets: Target tokens (batch, seq_len)

    Returns:
        Average loss
    """
    # Mask for valid targets (non-zero)
    mask = targets > 0
    if jnp.sum(mask) == 0:
        return 0.0

    # Log probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Gather target log probs
    batch_size, seq_len = targets.shape
    batch_idx = jnp.arange(batch_size)[:, None]
    seq_idx = jnp.arange(seq_len)[None, :]
    target_log_probs = log_probs[batch_idx, seq_idx, targets]

    # Average over valid positions
    loss = -jnp.sum(target_log_probs * mask) / jnp.sum(mask)
    return float(loss)


def mean_ablation(
    activations: jnp.ndarray,
    indices: Optional[List[int]] = None,
) -> jnp.ndarray:
    """
    Replace activations with their mean.

    Args:
        activations: Tensor of shape (batch, seq_len, d_model) or (batch, seq_len, n_heads, d_head)
        indices: If specified, only ablate these indices (heads or neurons)

    Returns:
        Ablated activations
    """
    if indices is None:
        # Mean ablate everything
        mean = jnp.mean(activations, axis=(0, 1), keepdims=True)
        return jnp.broadcast_to(mean, activations.shape)
    else:
        # Mean ablate specific indices
        mean = jnp.mean(activations, axis=(0, 1), keepdims=True)
        ablated = activations.copy()
        for idx in indices:
            ablated = ablated.at[:, :, idx].set(mean[:, :, idx])
        return ablated


def zero_ablation(
    activations: jnp.ndarray,
    indices: Optional[List[int]] = None,
) -> jnp.ndarray:
    """
    Zero out activations.

    Args:
        activations: Tensor to ablate
        indices: If specified, only ablate these indices

    Returns:
        Ablated activations
    """
    if indices is None:
        return jnp.zeros_like(activations)
    else:
        ablated = activations.copy()
        for idx in indices:
            ablated = ablated.at[:, :, idx].set(0.0)
        return ablated


def resample_ablation(
    activations: jnp.ndarray,
    indices: Optional[List[int]] = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> jnp.ndarray:
    """
    Resample ablation: replace with random samples from same distribution.

    Args:
        activations: Tensor to ablate
        indices: If specified, only ablate these indices
        rng: Random key

    Returns:
        Ablated activations
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    if indices is None:
        # Resample all
        batch_size = activations.shape[0]
        permutation = jax.random.permutation(rng, batch_size)
        return activations[permutation]
    else:
        # Resample specific indices
        ablated = activations.copy()
        for idx in indices:
            batch_size = activations.shape[0]
            perm = jax.random.permutation(rng, batch_size)
            ablated = ablated.at[:, :, idx].set(activations[perm, :, idx])
        return ablated


def compute_direct_effect(
    model,
    params,
    clean_input: jnp.ndarray,
    corrupted_input: jnp.ndarray,
    layer_idx: int,
    component: str = "attn",
) -> float:
    """
    Compute direct effect of a component on output.

    Direct effect = change in loss when ablating this component,
    with all upstream components using clean activations.

    Args:
        model: TinyTransformer model
        params: Model parameters
        clean_input: Clean input
        corrupted_input: Corrupted input
        layer_idx: Layer to test
        component: "attn" or "mlp"

    Returns:
        Direct effect score
    """
    # TODO: Implement direct effect computation
    # This requires path patching

    # For now, return placeholder
    return 0.0


# TODO: Future improvements:
# - Implement true ablation by modifying forward pass
# - Add automated circuit discovery via ablation
# - Integrate with deception detection
# - Add visualization of ablation results
# - Implement gradient-based attribution as alternative to ablation
