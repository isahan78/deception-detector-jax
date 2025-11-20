"""Tiny Transformer implementation in Flax with activation caching for interpretability."""

from typing import Optional, Dict, Any
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict

from ..config import ModelConfig


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with cached attention weights."""

    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_cache: bool = False,
    ):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            deterministic: Whether to use dropout
            return_cache: Whether to return cached attention weights

        Returns:
            output: Attention output
            cache: Dict with attention weights (if return_cache=True)
        """
        batch_size, seq_len, d_model = x.shape
        d_head = d_model // self.config.n_heads

        # Project to Q, K, V
        qkv = nn.Dense(3 * d_model, name="qkv_proj")(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.config.n_heads, d_head)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)

        # Transpose to (batch, n_heads, seq_len, d_head)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(d_head)

        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)

        attn_weights = jax.nn.softmax(scores, axis=-1)

        # Apply dropout
        if not deterministic:
            attn_weights = nn.Dropout(self.config.dropout_rate)(
                attn_weights, deterministic=False
            )

        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, v)

        # Transpose back and reshape
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)

        # Output projection
        output = nn.Dense(d_model, name="out_proj")(attn_output)

        if return_cache:
            cache = {"attn_weights": attn_weights, "attn_output": attn_output}
            return output, cache
        return output, None


class MLP(nn.Module):
    """Feed-forward MLP with cached activations."""

    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
        return_cache: bool = False,
    ):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            deterministic: Whether to use dropout
            return_cache: Whether to return cached activations

        Returns:
            output: MLP output
            cache: Dict with intermediate activations (if return_cache=True)
        """
        # First layer with GELU activation
        hidden = nn.Dense(self.config.d_ff, name="fc1")(x)
        hidden = nn.gelu(hidden)

        # Dropout
        if not deterministic:
            hidden = nn.Dropout(self.config.dropout_rate)(hidden, deterministic=False)

        # Second layer
        output = nn.Dense(self.config.d_model, name="fc2")(hidden)

        if return_cache:
            cache = {"mlp_pre_act": x, "mlp_post_act": hidden, "mlp_output": output}
            return output, cache
        return output, None


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""

    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_cache: bool = False,
    ):
        """
        Args:
            x: Input tensor
            mask: Optional attention mask
            deterministic: Whether to use dropout
            return_cache: Whether to cache intermediate activations

        Returns:
            output: Block output
            cache: Dict with all intermediate activations (if return_cache=True)
        """
        cache = {}

        # Pre-LN: LayerNorm before attention
        normed = nn.LayerNorm(name="ln1")(x)

        # Self-attention
        attn_out, attn_cache = MultiHeadAttention(self.config, name="attn")(
            normed, mask=mask, deterministic=deterministic, return_cache=return_cache
        )

        # Residual connection
        x = x + attn_out

        if return_cache:
            cache.update({"attn": attn_cache, "resid_post_attn": x})

        # Pre-LN: LayerNorm before MLP
        normed = nn.LayerNorm(name="ln2")(x)

        # MLP
        mlp_out, mlp_cache = MLP(self.config, name="mlp")(
            normed, deterministic=deterministic, return_cache=return_cache
        )

        # Residual connection
        x = x + mlp_out

        if return_cache:
            cache.update({"mlp": mlp_cache, "resid_post_mlp": x})

        return x, cache if return_cache else None


class TinyTransformer(nn.Module):
    """
    Tiny Transformer language model with full activation caching.

    Designed for mechanistic interpretability and deception detection.
    """

    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        deterministic: bool = True,
        return_cache: bool = False,
    ):
        """
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            deterministic: Whether to use dropout
            return_cache: Whether to return cached activations

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            cache: Dict with all intermediate activations (if return_cache=True)
        """
        batch_size, seq_len = input_ids.shape
        cache = {} if return_cache else None

        # Token embeddings
        token_embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
            name="token_embed",
        )(input_ids)

        # Position embeddings
        positions = jnp.arange(seq_len)[None, :]  # (1, seq_len)
        pos_embed = nn.Embed(
            num_embeddings=self.config.seq_len,
            features=self.config.d_model,
            name="pos_embed",
        )(positions)

        # Combine embeddings
        x = token_embed + pos_embed

        if return_cache:
            cache["embeddings"] = {"token_embed": token_embed, "pos_embed": pos_embed}

        # Apply dropout to embeddings
        if not deterministic:
            x = nn.Dropout(self.config.dropout_rate)(x, deterministic=False)

        # Transformer blocks
        if return_cache:
            cache["blocks"] = []

        for layer_idx in range(self.config.n_layers):
            x, block_cache = TransformerBlock(
                self.config, name=f"block_{layer_idx}"
            )(x, deterministic=deterministic, return_cache=return_cache)

            if return_cache:
                cache["blocks"].append(block_cache)

        # Final layer norm
        x = nn.LayerNorm(name="ln_final")(x)

        if return_cache:
            cache["final_norm_output"] = x

        # Language modeling head
        logits = nn.Dense(self.config.vocab_size, name="lm_head")(x)

        return logits, cache


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """Create causal attention mask for autoregressive generation."""
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    # Reshape for broadcasting: (1, 1, seq_len, seq_len)
    mask = mask[None, None, :, :]
    return mask.astype(bool)


def init_model(config: ModelConfig, rng: jax.random.PRNGKey):
    """Initialize model parameters."""
    model = TinyTransformer(config)
    dummy_input = jnp.ones((1, config.seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_input, deterministic=True, return_cache=False)
    return model, variables["params"]
