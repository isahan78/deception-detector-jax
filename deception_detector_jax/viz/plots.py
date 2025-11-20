"""Visualization utilities for mechanistic interpretability."""

from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import jax.numpy as jnp

from ..interp.activation_cache import ActivationCache


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    layer_idx: int,
    head_idx: Optional[int] = None,
    tokens: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
):
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: Attention weights of shape (batch, n_heads, seq_len, seq_len)
                          or (n_heads, seq_len, seq_len)
        layer_idx: Layer index for title
        head_idx: If specified, plot only this head
        tokens: Optional list of token strings for labels
        title: Custom title
        figsize: Figure size
        save_path: If specified, save figure to this path
    """
    # Convert to numpy if needed
    if isinstance(attention_weights, jnp.ndarray):
        attention_weights = np.array(attention_weights)

    # Handle batch dimension
    if len(attention_weights.shape) == 4:
        attention_weights = attention_weights[0]  # Take first batch item

    # Select specific head if requested
    if head_idx is not None:
        attn_to_plot = attention_weights[head_idx]
        n_heads = 1
    else:
        attn_to_plot = attention_weights
        n_heads = attention_weights.shape[0]

    # Create subplots for each head
    if n_heads == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]
    else:
        ncols = min(4, n_heads)
        nrows = (n_heads + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_heads > 1 else [axes]

    for i in range(n_heads):
        if head_idx is not None:
            weights = attn_to_plot
            h_idx = head_idx
        else:
            weights = attn_to_plot[i]
            h_idx = i

        im = axes[i].imshow(weights, cmap="viridis", aspect="auto")
        axes[i].set_title(f"Head {h_idx}")
        axes[i].set_xlabel("Key Position")
        axes[i].set_ylabel("Query Position")

        # Add token labels if provided
        if tokens is not None:
            seq_len = len(tokens)
            axes[i].set_xticks(range(seq_len))
            axes[i].set_yticks(range(seq_len))
            axes[i].set_xticklabels(tokens, rotation=45, ha="right")
            axes[i].set_yticklabels(tokens)

        plt.colorbar(im, ax=axes[i])

    # Hide unused subplots
    for i in range(n_heads, len(axes)):
        axes[i].axis("off")

    if title is None:
        title = f"Attention Weights - Layer {layer_idx}"
    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


def plot_activation_norms(
    cache: ActivationCache,
    layer_idx: int,
    component: str = "mlp",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
):
    """
    Plot activation norms across sequence positions.

    Args:
        cache: ActivationCache object
        layer_idx: Layer index
        component: "mlp", "attn", or "resid"
        figsize: Figure size
        save_path: If specified, save figure to this path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if component == "mlp":
        mlp_acts = cache.get_mlp_activations(layer_idx)
        if "mlp_post_act" in mlp_acts:
            acts = mlp_acts["mlp_post_act"]
            # Compute norms across d_model dimension
            norms = np.linalg.norm(np.array(acts), axis=-1)  # (batch, seq_len)

            # Plot mean norm across batch
            mean_norms = norms.mean(axis=0)
            std_norms = norms.std(axis=0)

            ax1.plot(mean_norms, label="Mean Norm")
            ax1.fill_between(
                range(len(mean_norms)),
                mean_norms - std_norms,
                mean_norms + std_norms,
                alpha=0.3,
            )
            ax1.set_xlabel("Sequence Position")
            ax1.set_ylabel("Activation Norm")
            ax1.set_title(f"MLP Activation Norms - Layer {layer_idx}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Heatmap of norms across batch
            im = ax2.imshow(norms, aspect="auto", cmap="viridis")
            ax2.set_xlabel("Sequence Position")
            ax2.set_ylabel("Batch Index")
            ax2.set_title("Activation Norms (Batch View)")
            plt.colorbar(im, ax=ax2)

    elif component == "attn":
        attn_output = cache.get_attention_output(layer_idx)
        if attn_output is not None:
            acts = np.array(attn_output)
            norms = np.linalg.norm(acts, axis=-1)

            mean_norms = norms.mean(axis=0)
            std_norms = norms.std(axis=0)

            ax1.plot(mean_norms, label="Mean Norm")
            ax1.fill_between(
                range(len(mean_norms)),
                mean_norms - std_norms,
                mean_norms + std_norms,
                alpha=0.3,
            )
            ax1.set_xlabel("Sequence Position")
            ax1.set_ylabel("Activation Norm")
            ax1.set_title(f"Attention Output Norms - Layer {layer_idx}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            im = ax2.imshow(norms, aspect="auto", cmap="viridis")
            ax2.set_xlabel("Sequence Position")
            ax2.set_ylabel("Batch Index")
            ax2.set_title("Activation Norms (Batch View)")
            plt.colorbar(im, ax=ax2)

    elif component == "resid":
        resid = cache.get_residual_stream(layer_idx, "post_mlp")
        if resid is not None:
            acts = np.array(resid)
            norms = np.linalg.norm(acts, axis=-1)

            mean_norms = norms.mean(axis=0)
            std_norms = norms.std(axis=0)

            ax1.plot(mean_norms, label="Mean Norm")
            ax1.fill_between(
                range(len(mean_norms)),
                mean_norms - std_norms,
                mean_norms + std_norms,
                alpha=0.3,
            )
            ax1.set_xlabel("Sequence Position")
            ax1.set_ylabel("Residual Stream Norm")
            ax1.set_title(f"Residual Stream Norms - Layer {layer_idx}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            im = ax2.imshow(norms, aspect="auto", cmap="viridis")
            ax2.set_xlabel("Sequence Position")
            ax2.set_ylabel("Batch Index")
            ax2.set_title("Residual Stream Norms (Batch View)")
            plt.colorbar(im, ax=ax2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


def plot_residual_drift(
    cache: ActivationCache,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
):
    """
    Plot how residual stream evolves across layers.

    Args:
        cache: ActivationCache object
        figsize: Figure size
        save_path: If specified, save figure to this path
    """
    n_layers = len(cache.cache.get("blocks", []))

    # Collect residual norms at each layer
    post_attn_norms = []
    post_mlp_norms = []

    for layer_idx in range(n_layers):
        resid_attn = cache.get_residual_stream(layer_idx, "post_attn")
        resid_mlp = cache.get_residual_stream(layer_idx, "post_mlp")

        if resid_attn is not None:
            norm = np.linalg.norm(np.array(resid_attn), axis=-1).mean()
            post_attn_norms.append(norm)

        if resid_mlp is not None:
            norm = np.linalg.norm(np.array(resid_mlp), axis=-1).mean()
            post_mlp_norms.append(norm)

    fig, ax = plt.subplots(figsize=figsize)

    layers = range(n_layers)
    ax.plot(layers, post_attn_norms, marker="o", label="Post-Attention", linewidth=2)
    ax.plot(layers, post_mlp_norms, marker="s", label="Post-MLP", linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Residual Stream Norm")
    ax.set_title("Residual Stream Evolution Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


def plot_head_ablation_impact(
    impact_matrix: np.ndarray,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot impact matrix from head ablation sweep.

    Args:
        impact_matrix: Matrix of shape (n_layers, n_heads) with ablation impacts
        figsize: Figure size
        save_path: If specified, save figure to this path
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(impact_matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title("Attention Head Ablation Impact\n(Higher = More Important)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Loss Increase")

    # Add grid
    ax.set_xticks(range(impact_matrix.shape[1]))
    ax.set_yticks(range(impact_matrix.shape[0]))
    ax.grid(True, alpha=0.3, color="white", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


def plot_deception_comparison(
    clean_cache: ActivationCache,
    deceptive_cache: ActivationCache,
    layer_idx: int,
    component: str = "mlp",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
):
    """
    Compare activations between clean and deceptive examples.

    Args:
        clean_cache: Cache from clean examples
        deceptive_cache: Cache from deceptive examples
        layer_idx: Layer to compare
        component: "mlp" or "attn"
        figsize: Figure size
        save_path: If specified, save figure to this path
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    if component == "mlp":
        clean_acts = clean_cache.get_mlp_activations(layer_idx).get("mlp_post_act")
        decept_acts = deceptive_cache.get_mlp_activations(layer_idx).get("mlp_post_act")
    else:
        clean_acts = clean_cache.get_attention_output(layer_idx)
        decept_acts = deceptive_cache.get_attention_output(layer_idx)

    if clean_acts is not None and decept_acts is not None:
        clean_acts = np.array(clean_acts)
        decept_acts = np.array(decept_acts)

        # Compute norms
        clean_norms = np.linalg.norm(clean_acts, axis=-1).mean(axis=0)
        decept_norms = np.linalg.norm(decept_acts, axis=-1).mean(axis=0)

        # Plot clean
        ax1.plot(clean_norms, color="green", linewidth=2)
        ax1.set_title("Clean Examples")
        ax1.set_xlabel("Sequence Position")
        ax1.set_ylabel("Activation Norm")
        ax1.grid(True, alpha=0.3)

        # Plot deceptive
        ax2.plot(decept_norms, color="red", linewidth=2)
        ax2.set_title("Deceptive Examples")
        ax2.set_xlabel("Sequence Position")
        ax2.set_ylabel("Activation Norm")
        ax2.grid(True, alpha=0.3)

        # Plot difference
        diff = decept_norms - clean_norms
        ax3.plot(diff, color="purple", linewidth=2)
        ax3.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax3.set_title("Difference (Deceptive - Clean)")
        ax3.set_xlabel("Sequence Position")
        ax3.set_ylabel("Norm Difference")
        ax3.grid(True, alpha=0.3)

    fig.suptitle(f"{component.upper()} Activations - Layer {layer_idx}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
):
    """
    Plot training loss and accuracy curves.

    Args:
        history: Dict with keys like "train_loss", "val_loss", "train_acc", "val_acc"
        figsize: Figure size
        save_path: If specified, save figure to this path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    if "train_loss" in history:
        ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    if "train_acc" in history:
        ax2.plot(history["train_acc"], label="Train Acc", linewidth=2)
    if "val_acc" in history:
        ax2.plot(history["val_acc"], label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig
