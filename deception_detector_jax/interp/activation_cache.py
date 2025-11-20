"""Activation caching utilities for mechanistic interpretability."""

from typing import Dict, Any, Optional, List
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict


class ActivationCache:
    """
    Store and manage intermediate activations from transformer forward passes.

    This class provides utilities to:
    - Store activations from model runs
    - Access activations by layer and component
    - Compute statistics over activations
    - Export activations for visualization
    """

    def __init__(self, cache_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize activation cache.

        Args:
            cache_dict: Optional pre-filled cache dictionary from model forward pass
        """
        self.cache = cache_dict if cache_dict is not None else {}

    @classmethod
    def from_model_output(cls, cache_dict: Dict[str, Any]) -> "ActivationCache":
        """Create ActivationCache from model output."""
        return cls(cache_dict)

    def get_embeddings(self) -> Dict[str, jnp.ndarray]:
        """Get token and position embeddings."""
        return self.cache.get("embeddings", {})

    def get_block_cache(self, layer_idx: int) -> Dict[str, Any]:
        """Get cache for a specific transformer block."""
        if "blocks" not in self.cache:
            return {}
        if layer_idx >= len(self.cache["blocks"]):
            raise IndexError(f"Layer {layer_idx} not found in cache")
        return self.cache["blocks"][layer_idx]

    def get_attention_weights(self, layer_idx: int) -> Optional[jnp.ndarray]:
        """
        Get attention weights for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Attention weights of shape (batch, n_heads, seq_len, seq_len)
        """
        block_cache = self.get_block_cache(layer_idx)
        if "attn" in block_cache and block_cache["attn"] is not None:
            return block_cache["attn"].get("attn_weights")
        return None

    def get_attention_output(self, layer_idx: int) -> Optional[jnp.ndarray]:
        """Get attention output before residual connection."""
        block_cache = self.get_block_cache(layer_idx)
        if "attn" in block_cache and block_cache["attn"] is not None:
            return block_cache["attn"].get("attn_output")
        return None

    def get_mlp_activations(self, layer_idx: int) -> Dict[str, jnp.ndarray]:
        """
        Get MLP activations for a specific layer.

        Returns:
            Dict with keys: mlp_pre_act, mlp_post_act, mlp_output
        """
        block_cache = self.get_block_cache(layer_idx)
        if "mlp" in block_cache and block_cache["mlp"] is not None:
            return block_cache["mlp"]
        return {}

    def get_residual_stream(self, layer_idx: int, position: str = "post_attn") -> Optional[jnp.ndarray]:
        """
        Get residual stream at a specific point.

        Args:
            layer_idx: Layer index
            position: "post_attn" or "post_mlp"

        Returns:
            Residual stream tensor
        """
        block_cache = self.get_block_cache(layer_idx)
        if position == "post_attn":
            return block_cache.get("resid_post_attn")
        elif position == "post_mlp":
            return block_cache.get("resid_post_mlp")
        else:
            raise ValueError(f"Unknown position: {position}")

    def get_final_output(self) -> Optional[jnp.ndarray]:
        """Get final layer norm output before LM head."""
        return self.cache.get("final_norm_output")

    def get_all_attention_weights(self) -> List[jnp.ndarray]:
        """Get attention weights from all layers."""
        weights = []
        for i in range(len(self.cache.get("blocks", []))):
            attn_w = self.get_attention_weights(i)
            if attn_w is not None:
                weights.append(attn_w)
        return weights

    def get_all_mlp_activations(self) -> List[Dict[str, jnp.ndarray]]:
        """Get MLP activations from all layers."""
        activations = []
        for i in range(len(self.cache.get("blocks", []))):
            mlp_acts = self.get_mlp_activations(i)
            if mlp_acts:
                activations.append(mlp_acts)
        return activations

    def compute_activation_stats(self, layer_idx: int) -> Dict[str, float]:
        """
        Compute statistics for activations in a layer.

        Returns:
            Dict with mean, std, max norms of activations
        """
        stats = {}

        # Attention output stats
        attn_output = self.get_attention_output(layer_idx)
        if attn_output is not None:
            stats["attn_mean_norm"] = float(jnp.linalg.norm(attn_output, axis=-1).mean())
            stats["attn_max_norm"] = float(jnp.linalg.norm(attn_output, axis=-1).max())

        # MLP stats
        mlp_acts = self.get_mlp_activations(layer_idx)
        if "mlp_post_act" in mlp_acts:
            mlp_hidden = mlp_acts["mlp_post_act"]
            stats["mlp_mean_norm"] = float(jnp.linalg.norm(mlp_hidden, axis=-1).mean())
            stats["mlp_max_norm"] = float(jnp.linalg.norm(mlp_hidden, axis=-1).max())

        # Residual stream stats
        resid = self.get_residual_stream(layer_idx, "post_mlp")
        if resid is not None:
            stats["resid_mean_norm"] = float(jnp.linalg.norm(resid, axis=-1).mean())
            stats["resid_max_norm"] = float(jnp.linalg.norm(resid, axis=-1).max())

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert cache to plain dictionary (for serialization)."""
        return self.cache

    def __repr__(self) -> str:
        n_layers = len(self.cache.get("blocks", []))
        has_embeddings = "embeddings" in self.cache
        has_final = "final_norm_output" in self.cache
        return (
            f"ActivationCache(layers={n_layers}, "
            f"has_embeddings={has_embeddings}, "
            f"has_final={has_final})"
        )


def run_with_cache(model, params, input_ids, deterministic=True):
    """
    Run model forward pass and return both outputs and cache.

    Args:
        model: TinyTransformer model
        params: Model parameters
        input_ids: Input token IDs
        deterministic: Whether to use dropout

    Returns:
        logits: Model output logits
        cache: ActivationCache object
    """
    logits, cache_dict = model.apply(
        {"params": params},
        input_ids,
        deterministic=deterministic,
        return_cache=True,
    )
    cache = ActivationCache.from_model_output(cache_dict)
    return logits, cache


def compare_caches(cache1: ActivationCache, cache2: ActivationCache, layer_idx: int) -> Dict[str, float]:
    """
    Compare activations between two caches at a specific layer.

    Args:
        cache1: First activation cache
        cache2: Second activation cache
        layer_idx: Layer to compare

    Returns:
        Dict with L2 differences for each component
    """
    diffs = {}

    # Compare attention outputs
    attn1 = cache1.get_attention_output(layer_idx)
    attn2 = cache2.get_attention_output(layer_idx)
    if attn1 is not None and attn2 is not None:
        diffs["attn_l2"] = float(jnp.linalg.norm(attn1 - attn2))

    # Compare MLP activations
    mlp1 = cache1.get_mlp_activations(layer_idx)
    mlp2 = cache2.get_mlp_activations(layer_idx)
    if "mlp_post_act" in mlp1 and "mlp_post_act" in mlp2:
        diffs["mlp_l2"] = float(jnp.linalg.norm(mlp1["mlp_post_act"] - mlp2["mlp_post_act"]))

    # Compare residual streams
    resid1 = cache1.get_residual_stream(layer_idx, "post_mlp")
    resid2 = cache2.get_residual_stream(layer_idx, "post_mlp")
    if resid1 is not None and resid2 is not None:
        diffs["resid_l2"] = float(jnp.linalg.norm(resid1 - resid2))

    return diffs
