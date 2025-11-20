"""Activation patching utilities for causal intervention experiments."""

from typing import Dict, Any, Callable, Optional
import jax
import jax.numpy as jnp
from functools import partial

from .activation_cache import ActivationCache


class ActivationPatcher:
    """
    Utilities for patching activations during model forward passes.

    Activation patching is a core technique in mechanistic interpretability:
    - Run model on input A, cache activations
    - Run model on input B, but replace some activations with those from A
    - Observe how output changes to understand causal structure
    """

    def __init__(self, model, params):
        """
        Initialize patcher with model and parameters.

        Args:
            model: TinyTransformer model
            params: Model parameters
        """
        self.model = model
        self.params = params

    def patch_attention_output(
        self,
        input_ids: jnp.ndarray,
        source_cache: ActivationCache,
        layer_idx: int,
        head_idx: Optional[int] = None,
    ) -> tuple[jnp.ndarray, ActivationCache]:
        """
        Patch attention output from source cache into a new forward pass.

        Args:
            input_ids: Input tokens for the corrupted/new run
            source_cache: Cache from the clean run to patch from
            layer_idx: Which layer to patch
            head_idx: If specified, only patch this attention head

        Returns:
            logits: Output after patching
            cache: New cache with patched activations

        Note:
            This is a simplified patching implementation.
            Full implementation would require model modifications or JAX hooks.
            For now, this demonstrates the concept.
        """
        # TODO: Implement full activation patching with JAX pytree manipulation
        # This requires either:
        # 1. Modified model with explicit patching hooks
        # 2. JAX pytree transformation during forward pass
        # 3. Custom VJP/JVP rules

        # Placeholder: Run normal forward pass
        logits, new_cache_dict = self.model.apply(
            {"params": self.params},
            input_ids,
            deterministic=True,
            return_cache=True,
        )
        new_cache = ActivationCache.from_model_output(new_cache_dict)

        # In a full implementation, we would:
        # 1. Get source attention output
        # 2. Replace corresponding activations in forward pass
        # 3. Continue computation from that point

        return logits, new_cache

    def patch_residual_stream(
        self,
        input_ids: jnp.ndarray,
        source_cache: ActivationCache,
        layer_idx: int,
        position: str = "post_attn",
        token_pos: Optional[int] = None,
    ) -> tuple[jnp.ndarray, ActivationCache]:
        """
        Patch residual stream from source cache.

        Args:
            input_ids: Input tokens
            source_cache: Source cache to patch from
            layer_idx: Layer index
            position: "post_attn" or "post_mlp"
            token_pos: If specified, only patch this token position

        Returns:
            logits, cache after patching
        """
        # TODO: Full implementation with JAX pytree manipulation
        logits, new_cache_dict = self.model.apply(
            {"params": self.params},
            input_ids,
            deterministic=True,
            return_cache=True,
        )
        new_cache = ActivationCache.from_model_output(new_cache_dict)
        return logits, new_cache


def patch_and_compare(
    model,
    params,
    clean_input: jnp.ndarray,
    corrupted_input: jnp.ndarray,
    patch_layer: int,
    component: str = "attn",
) -> Dict[str, Any]:
    """
    Simplified activation patching experiment.

    Args:
        model: TinyTransformer model
        params: Model parameters
        clean_input: Clean input tokens
        corrupted_input: Corrupted input tokens
        patch_layer: Which layer to patch
        component: "attn" or "mlp"

    Returns:
        Dict with clean_logits, corrupted_logits, patched_logits, and metrics
    """
    # Run clean forward pass
    clean_logits, clean_cache_dict = model.apply(
        {"params": params},
        clean_input,
        deterministic=True,
        return_cache=True,
    )
    clean_cache = ActivationCache.from_model_output(clean_cache_dict)

    # Run corrupted forward pass
    corrupted_logits, corrupted_cache_dict = model.apply(
        {"params": params},
        corrupted_input,
        deterministic=True,
        return_cache=True,
    )
    corrupted_cache = ActivationCache.from_model_output(corrupted_cache_dict)

    # TODO: Implement actual patching
    # For now, return both caches for manual analysis
    patched_logits = corrupted_logits  # Placeholder

    # Compute metrics
    clean_pred = jnp.argmax(clean_logits, axis=-1)
    corrupted_pred = jnp.argmax(corrupted_logits, axis=-1)
    patched_pred = jnp.argmax(patched_logits, axis=-1)

    return {
        "clean_logits": clean_logits,
        "corrupted_logits": corrupted_logits,
        "patched_logits": patched_logits,
        "clean_pred": clean_pred,
        "corrupted_pred": corrupted_pred,
        "patched_pred": patched_pred,
        "clean_cache": clean_cache,
        "corrupted_cache": corrupted_cache,
    }


def compute_patching_effect(
    clean_logits: jnp.ndarray,
    corrupted_logits: jnp.ndarray,
    patched_logits: jnp.ndarray,
    metric: str = "kl",
) -> float:
    """
    Compute the effect of patching on model output.

    Args:
        clean_logits: Logits from clean run
        corrupted_logits: Logits from corrupted run
        patched_logits: Logits from patched run
        metric: "kl" for KL divergence or "l2" for L2 distance

    Returns:
        Patching effect score (higher = more causal importance)
    """
    if metric == "kl":
        # KL divergence from clean to patched
        clean_probs = jax.nn.softmax(clean_logits, axis=-1)
        patched_probs = jax.nn.softmax(patched_logits, axis=-1)

        kl = jnp.sum(clean_probs * jnp.log(clean_probs / (patched_probs + 1e-10)), axis=-1)
        return float(kl.mean())

    elif metric == "l2":
        # L2 distance
        diff = clean_logits - patched_logits
        return float(jnp.linalg.norm(diff))

    else:
        raise ValueError(f"Unknown metric: {metric}")


def mean_ablation_patching(
    model,
    params,
    inputs: jnp.ndarray,
    layer_idx: int,
    component: str = "attn",
) -> jnp.ndarray:
    """
    Perform mean ablation: replace activations with their mean across dataset.

    This is a simpler form of patching useful for identifying important features.

    Args:
        model: TinyTransformer model
        params: Model parameters
        inputs: Batch of inputs (batch_size, seq_len)
        layer_idx: Layer to ablate
        component: "attn" or "mlp"

    Returns:
        logits: Output after mean ablation
    """
    # Get activations for all inputs
    _, cache_dict = model.apply(
        {"params": params},
        inputs,
        deterministic=True,
        return_cache=True,
    )
    cache = ActivationCache.from_model_output(cache_dict)

    # Compute mean activation
    if component == "attn":
        acts = cache.get_attention_output(layer_idx)
    elif component == "mlp":
        acts = cache.get_mlp_activations(layer_idx).get("mlp_output")
    else:
        raise ValueError(f"Unknown component: {component}")

    if acts is None:
        raise ValueError(f"No activations found for component {component} at layer {layer_idx}")

    mean_act = jnp.mean(acts, axis=0, keepdims=True)

    # TODO: Re-run model with mean activation patched in
    # For now, return original logits
    logits, _ = model.apply(
        {"params": params},
        inputs,
        deterministic=True,
        return_cache=False,
    )

    return logits


# TODO: Future improvements:
# - Implement true activation patching with JAX pytree manipulation
# - Add path patching (patching entire computational paths)
# - Add automated patching sweeps across all layers/heads
# - Integrate with deception detection benchmarks
