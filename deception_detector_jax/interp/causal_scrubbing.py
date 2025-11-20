"""Causal scrubbing implementation for testing mechanistic hypotheses."""

from typing import Dict, Any, Optional, List, Callable
import jax
import jax.numpy as jnp
import numpy as np

from .activation_cache import ActivationCache


def project_onto_subspace(
    activations: jnp.ndarray,
    basis_vectors: jnp.ndarray,
    keep: bool = True,
) -> jnp.ndarray:
    """
    Project activations onto (or off of) a subspace.

    Args:
        activations: Tensor of shape (..., d_model)
        basis_vectors: Orthonormal basis of shape (k, d_model) for subspace
        keep: If True, keep only subspace component; if False, remove it

    Returns:
        Projected activations
    """
    # Project onto subspace
    # P = B^T B where B is basis_vectors
    # projection = activations @ B^T @ B

    # Compute projection matrix
    proj_matrix = jnp.matmul(basis_vectors.T, basis_vectors)  # (d_model, d_model)

    # Apply projection
    original_shape = activations.shape
    acts_flat = activations.reshape(-1, activations.shape[-1])  # (batch * seq, d_model)

    if keep:
        # Keep only subspace component
        projected = jnp.matmul(acts_flat, proj_matrix)
    else:
        # Remove subspace component (project onto orthogonal complement)
        identity = jnp.eye(activations.shape[-1])
        orthogonal_proj = identity - proj_matrix
        projected = jnp.matmul(acts_flat, orthogonal_proj)

    projected = projected.reshape(original_shape)
    return projected


def compute_pca_basis(
    activations: jnp.ndarray,
    n_components: int = 5,
) -> jnp.ndarray:
    """
    Compute PCA basis vectors for a set of activations.

    Args:
        activations: Tensor of shape (batch, seq_len, d_model)
        n_components: Number of principal components

    Returns:
        Basis vectors of shape (n_components, d_model)
    """
    # Flatten batch and sequence dimensions
    acts_flat = activations.reshape(-1, activations.shape[-1])  # (batch*seq, d_model)

    # Center the data
    mean = jnp.mean(acts_flat, axis=0, keepdims=True)
    centered = acts_flat - mean

    # Compute covariance matrix
    cov = jnp.matmul(centered.T, centered) / (centered.shape[0] - 1)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Return top n_components
    basis = eigenvectors[:, :n_components].T  # (n_components, d_model)
    return basis


class CausalScrubber:
    """
    Causal scrubbing: test mechanistic hypotheses by selectively preserving
    information in the model's computational graph.

    Based on "Causal Scrubbing" (Redwood Research, 2022).
    """

    def __init__(self, model, params):
        """
        Initialize scrubber.

        Args:
            model: TinyTransformer model
            params: Model parameters
        """
        self.model = model
        self.params = params

    def scrub_subspace(
        self,
        input_ids: jnp.ndarray,
        layer_idx: int,
        basis_vectors: jnp.ndarray,
        component: str = "mlp",
        keep: bool = False,
    ) -> tuple[jnp.ndarray, ActivationCache]:
        """
        Run model with a subspace scrubbed (removed or isolated).

        Args:
            input_ids: Input tokens
            layer_idx: Layer to scrub
            basis_vectors: Subspace basis
            component: "mlp" or "attn"
            keep: If True, keep only this subspace; if False, remove it

        Returns:
            logits, cache after scrubbing
        """
        # TODO: Implement full scrubbing with JAX pytree manipulation
        # This requires modifying activations during forward pass

        # For now, run normal forward pass
        logits, cache_dict = self.model.apply(
            {"params": self.params},
            input_ids,
            deterministic=True,
            return_cache=True,
        )
        cache = ActivationCache.from_model_output(cache_dict)

        # In full implementation:
        # 1. Get activations at layer_idx, component
        # 2. Project onto/off of subspace
        # 3. Continue forward pass with modified activations

        return logits, cache

    def scrub_attention_head(
        self,
        input_ids: jnp.ndarray,
        layer_idx: int,
        head_idx: int,
    ) -> tuple[jnp.ndarray, ActivationCache]:
        """
        Scrub (zero out) a specific attention head.

        Args:
            input_ids: Input tokens
            layer_idx: Layer index
            head_idx: Head index to scrub

        Returns:
            logits, cache after scrubbing
        """
        # TODO: Implement head-specific scrubbing
        logits, cache_dict = self.model.apply(
            {"params": self.params},
            input_ids,
            deterministic=True,
            return_cache=True,
        )
        cache = ActivationCache.from_model_output(cache_dict)
        return logits, cache


def scrub_and_evaluate(
    model,
    params,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    layer_idx: int,
    subspace_basis: jnp.ndarray,
    keep: bool = False,
) -> Dict[str, float]:
    """
    Evaluate model performance with and without scrubbing.

    Args:
        model: TinyTransformer model
        params: Model parameters
        inputs: Input tokens
        targets: Target tokens
        layer_idx: Layer to scrub
        subspace_basis: Subspace basis to scrub
        keep: Whether to keep or remove subspace

    Returns:
        Dict with baseline_loss, scrubbed_loss, loss_change
    """
    # Baseline: normal forward pass
    baseline_logits, _ = model.apply(
        {"params": params},
        inputs,
        deterministic=True,
        return_cache=False,
    )

    # Compute baseline loss
    baseline_loss = compute_cross_entropy(baseline_logits, targets)

    # Scrubbed: forward pass with scrubbing
    scrubber = CausalScrubber(model, params)
    scrubbed_logits, _ = scrubber.scrub_subspace(
        inputs, layer_idx, subspace_basis, keep=keep
    )

    # Compute scrubbed loss
    scrubbed_loss = compute_cross_entropy(scrubbed_logits, targets)

    return {
        "baseline_loss": float(baseline_loss),
        "scrubbed_loss": float(scrubbed_loss),
        "loss_change": float(scrubbed_loss - baseline_loss),
        "loss_increase_pct": float(100 * (scrubbed_loss - baseline_loss) / baseline_loss),
    }


def compute_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> float:
    """Compute cross-entropy loss."""
    # Get logits at target positions
    # Assuming targets are at specific positions in sequence
    batch_size, seq_len, vocab_size = logits.shape

    # Simple version: average over all positions where target != 0
    mask = targets > 0
    if jnp.sum(mask) == 0:
        return 0.0

    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Gather log probs for target tokens
    batch_idx = jnp.arange(batch_size)[:, None]
    seq_idx = jnp.arange(seq_len)[None, :]
    target_log_probs = log_probs[batch_idx, seq_idx, targets]

    # Average over valid positions
    loss = -jnp.sum(target_log_probs * mask) / jnp.sum(mask)
    return float(loss)


def identify_deceptive_subspace(
    clean_activations: jnp.ndarray,
    deceptive_activations: jnp.ndarray,
    n_components: int = 3,
) -> jnp.ndarray:
    """
    Identify subspace that distinguishes clean vs deceptive activations.

    Uses difference in means as the primary direction, plus PCA on difference.

    Args:
        clean_activations: Activations from clean/honest examples (batch, seq, d_model)
        deceptive_activations: Activations from deceptive examples
        n_components: Number of basis vectors to return

    Returns:
        Basis vectors of shape (n_components, d_model)
    """
    # Flatten batch/sequence
    clean_flat = clean_activations.reshape(-1, clean_activations.shape[-1])
    deceptive_flat = deceptive_activations.reshape(-1, deceptive_activations.shape[-1])

    # Primary direction: difference in means
    mean_diff = jnp.mean(deceptive_flat, axis=0) - jnp.mean(clean_flat, axis=0)
    mean_diff = mean_diff / (jnp.linalg.norm(mean_diff) + 1e-8)

    if n_components == 1:
        return mean_diff[None, :]  # (1, d_model)

    # Additional directions: PCA on differences
    all_acts = jnp.concatenate([clean_flat, deceptive_flat], axis=0)
    pca_basis = compute_pca_basis(all_acts[None, :, :], n_components=n_components - 1)

    # Combine mean direction with PCA directions
    basis = jnp.concatenate([mean_diff[None, :], pca_basis], axis=0)

    # Orthogonalize using Gram-Schmidt
    basis = gram_schmidt(basis)

    return basis


def gram_schmidt(vectors: jnp.ndarray) -> jnp.ndarray:
    """
    Gram-Schmidt orthogonalization.

    Args:
        vectors: Array of shape (n_vectors, d_model)

    Returns:
        Orthonormal vectors
    """
    n_vectors, d_model = vectors.shape
    orthogonal = []

    for i in range(n_vectors):
        vec = vectors[i]

        # Subtract projections onto previous vectors
        for prev in orthogonal:
            vec = vec - jnp.dot(vec, prev) * prev

        # Normalize
        norm = jnp.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
            orthogonal.append(vec)

    return jnp.stack(orthogonal) if orthogonal else jnp.array([])


# TODO: Future improvements:
# - Implement full computational graph scrubbing
# - Add hypothesis testing framework
# - Integrate with deception detection pipeline
# - Add visualization of scrubbing results
