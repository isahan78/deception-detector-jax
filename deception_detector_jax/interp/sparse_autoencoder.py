"""Sparse Autoencoder (SAE) for learning interpretable features.

This is a placeholder for future SAE implementation.

Sparse autoencoders are useful for:
- Discovering monosemantic features in neural networks
- Decomposing superposition in activations
- Finding interpretable directions in activation space
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for learning interpretable features from activations.

    TODO: Implement full SAE training pipeline
    - L1 sparsity penalty
    - Reconstruction loss
    - Feature visualization
    - Integration with transformer activations
    """

    d_input: int  # Input dimension (typically d_model)
    d_hidden: int  # Hidden dimension (expansion factor, e.g., 4x or 8x)
    sparsity_coef: float = 1e-3  # L1 sparsity coefficient

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass of sparse autoencoder.

        Args:
            x: Input activations of shape (batch, seq_len, d_input)
            training: Whether in training mode

        Returns:
            reconstructed: Reconstructed activations
            hidden: Sparse hidden features
        """
        # Encoder
        hidden = nn.Dense(self.d_hidden, name="encoder")(x)
        hidden = nn.relu(hidden)

        # Decoder
        reconstructed = nn.Dense(self.d_input, name="decoder")(hidden)

        return reconstructed, hidden

    def compute_loss(
        self,
        x: jnp.ndarray,
        reconstructed: jnp.ndarray,
        hidden: jnp.ndarray,
    ) -> Tuple[float, dict]:
        """
        Compute SAE loss.

        Args:
            x: Original input
            reconstructed: Reconstructed output
            hidden: Sparse hidden features

        Returns:
            total_loss: Combined loss
            loss_dict: Dict with individual loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = jnp.mean((x - reconstructed) ** 2)

        # Sparsity loss (L1 on hidden activations)
        sparsity_loss = jnp.mean(jnp.abs(hidden))

        # Total loss
        total_loss = recon_loss + self.sparsity_coef * sparsity_loss

        loss_dict = {
            "recon_loss": float(recon_loss),
            "sparsity_loss": float(sparsity_loss),
            "total_loss": float(total_loss),
            "mean_activation": float(jnp.mean(jnp.abs(hidden))),
        }

        return total_loss, loss_dict


def train_sae_on_activations(
    activations: jnp.ndarray,
    d_hidden: int = 256,
    sparsity_coef: float = 1e-3,
    num_steps: int = 1000,
) -> Tuple[SparseAutoencoder, dict]:
    """
    Train sparse autoencoder on cached activations.

    TODO: Implement full training loop

    Args:
        activations: Cached activations from transformer
        d_hidden: Hidden dimension for SAE
        sparsity_coef: L1 sparsity coefficient
        num_steps: Number of training steps

    Returns:
        trained_sae: Trained SAE model
        params: SAE parameters
    """
    # TODO: Implement SAE training
    # 1. Initialize SAE
    # 2. Create optimizer
    # 3. Training loop with reconstruction + sparsity loss
    # 4. Return trained model

    d_input = activations.shape[-1]
    sae = SparseAutoencoder(
        d_input=d_input,
        d_hidden=d_hidden,
        sparsity_coef=sparsity_coef,
    )

    # Placeholder return
    return sae, {}


def extract_interpretable_features(
    sae: SparseAutoencoder,
    params: dict,
    activations: jnp.ndarray,
) -> jnp.ndarray:
    """
    Extract interpretable sparse features using trained SAE.

    Args:
        sae: Trained SAE model
        params: SAE parameters
        activations: Input activations

    Returns:
        features: Sparse feature activations
    """
    # TODO: Implement feature extraction
    reconstructed, features = sae.apply({"params": params}, activations)
    return features


def find_deception_features(
    sae: SparseAutoencoder,
    params: dict,
    clean_activations: jnp.ndarray,
    deceptive_activations: jnp.ndarray,
    threshold: float = 0.5,
) -> jnp.ndarray:
    """
    Identify SAE features that activate primarily on deceptive examples.

    TODO: Implement deception feature identification

    Args:
        sae: Trained SAE
        params: SAE parameters
        clean_activations: Activations from clean examples
        deceptive_activations: Activations from deceptive examples
        threshold: Threshold for feature selectivity

    Returns:
        deception_feature_indices: Indices of features associated with deception
    """
    # TODO: Implement
    # 1. Extract features for both clean and deceptive
    # 2. Compute selectivity score for each feature
    # 3. Return features with high selectivity for deceptive examples

    return jnp.array([])


# TODO: Future work
# - Implement full SAE training pipeline
# - Add feature visualization tools
# - Integrate with deception detection benchmark
# - Add automated feature interpretation
# - Implement feature steering/intervention
