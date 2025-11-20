"""Activation Explorer - Visualize attention patterns and activations."""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_loader import (
    get_available_models,
    load_model,
    load_task_data,
    run_model_inference,
    get_task_metadata,
)
from deception_detector_jax.interp.activation_cache import ActivationCache

st.set_page_config(page_title="Activation Explorer", page_icon="👁️", layout="wide")

st.title("👁️ Activation Explorer")
st.markdown("Visualize attention patterns, activation norms, and residual streams.")

# Load available models
models = get_available_models()

if not models:
    st.warning("⚠️ No trained models found. Please train models first.")
    st.stop()

# Sidebar: Model selection
st.sidebar.header("Model Selection")
selected_task = st.sidebar.selectbox("Select Task", models)
metadata = get_task_metadata(selected_task)

st.sidebar.markdown(f"**Description:** {metadata.get('description', 'N/A')}")

# Load model and data
try:
    with st.spinner("Loading model..."):
        model, params = load_model(selected_task)
        test_data = load_task_data(selected_task, split="test")

    st.sidebar.success(f"✅ Model loaded: {selected_task}")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Example selection
st.sidebar.header("Example Selection")
num_examples = len(test_data["input_ids"])
example_idx = st.sidebar.slider("Select Example", 0, num_examples - 1, 0)

# Load example
input_ids = test_data["input_ids"][example_idx: example_idx + 1]
target_ids = test_data["target_ids"][example_idx: example_idx + 1]

# Display example info
st.header("📝 Selected Example")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Example Index", example_idx)

with col2:
    if metadata.get("hidden_variable") in test_data:
        hidden_val = test_data[metadata["hidden_variable"]][example_idx]
        st.metric(metadata["hidden_variable"].title(), str(hidden_val))

with col3:
    # Get prediction
    logits, _ = run_model_inference(model, params, input_ids, return_cache=False)
    pred = np.argmax(logits[0], axis=-1)
    st.metric("Prediction Positions", f"{np.sum(pred > 0)}")

# Run inference with cache
with st.spinner("Running inference..."):
    logits, cache_dict = run_model_inference(model, params, input_ids, return_cache=True)
    cache = ActivationCache.from_model_output(cache_dict)

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Attention Patterns",
    "📊 Activation Norms",
    "🔄 Residual Stream",
    "📈 Statistics"
])

# Tab 1: Attention Patterns
with tab1:
    st.subheader("Attention Weight Heatmaps")

    n_layers = len(cache.cache.get("blocks", []))
    n_heads = 4  # From config

    layer_idx = st.slider("Select Layer", 0, n_layers - 1, 0, key="attn_layer")

    attn_weights = cache.get_attention_weights(layer_idx)

    if attn_weights is not None:
        attn_weights_np = np.array(attn_weights[0])  # (n_heads, seq_len, seq_len)

        # Display all heads
        cols = st.columns(min(n_heads, 4))

        for head_idx in range(n_heads):
            with cols[head_idx % 4]:
                weights = attn_weights_np[head_idx]

                fig = go.Figure(data=go.Heatmap(
                    z=weights,
                    colorscale='Viridis',
                    showscale=True if head_idx == 0 else False,
                ))
                fig.update_layout(
                    title=f"Head {head_idx}",
                    xaxis_title="Key Position",
                    yaxis_title="Query Position",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Attention pattern analysis
        st.markdown("### 📊 Attention Pattern Analysis")

        # Compute attention statistics
        attn_entropy = -np.sum(attn_weights_np * np.log(attn_weights_np + 1e-10), axis=-1)
        mean_entropy = attn_entropy.mean()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Mean Attention Entropy", f"{mean_entropy:.3f}")

        with col2:
            max_attn = attn_weights_np.max()
            st.metric("Max Attention Weight", f"{max_attn:.3f}")

        with col3:
            # Diagonal attention (self-attention strength)
            diag_attn = np.mean([attn_weights_np[h].diagonal() for h in range(n_heads)])
            st.metric("Mean Diagonal Attention", f"{diag_attn:.3f}")

    else:
        st.warning("No attention weights available for this layer")

# Tab 2: Activation Norms
with tab2:
    st.subheader("Activation Magnitudes Across Layers")

    layer_idx = st.slider("Select Layer", 0, n_layers - 1, 0, key="act_layer")

    # MLP activations
    mlp_acts = cache.get_mlp_activations(layer_idx)

    if mlp_acts and "mlp_post_act" in mlp_acts:
        acts = np.array(mlp_acts["mlp_post_act"][0])  # (seq_len, d_ff)
        norms = np.linalg.norm(acts, axis=-1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=norms,
            mode='lines+markers',
            name='MLP Activation Norms',
            line=dict(width=3),
        ))
        fig.update_layout(
            title=f"MLP Activation Norms - Layer {layer_idx}",
            xaxis_title="Sequence Position",
            yaxis_title="L2 Norm",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap of activations
        st.markdown("### MLP Activation Heatmap")
        fig = go.Figure(data=go.Heatmap(
            z=acts.T,
            colorscale='RdBu',
            zmid=0,
        ))
        fig.update_layout(
            title="MLP Hidden Activations",
            xaxis_title="Sequence Position",
            yaxis_title="Hidden Dimension",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Residual Stream
with tab3:
    st.subheader("Residual Stream Evolution")

    # Collect residual norms across all layers
    post_attn_norms = []
    post_mlp_norms = []

    for layer in range(n_layers):
        resid_attn = cache.get_residual_stream(layer, "post_attn")
        resid_mlp = cache.get_residual_stream(layer, "post_mlp")

        if resid_attn is not None:
            norm = np.linalg.norm(np.array(resid_attn[0]), axis=-1).mean()
            post_attn_norms.append(norm)

        if resid_mlp is not None:
            norm = np.linalg.norm(np.array(resid_mlp[0]), axis=-1).mean()
            post_mlp_norms.append(norm)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(n_layers)),
        y=post_attn_norms,
        mode='lines+markers',
        name='Post-Attention',
        line=dict(width=3),
    ))
    fig.add_trace(go.Scatter(
        x=list(range(n_layers)),
        y=post_mlp_norms,
        mode='lines+markers',
        name='Post-MLP',
        line=dict(width=3),
    ))
    fig.update_layout(
        title="Residual Stream Norm Evolution",
        xaxis_title="Layer",
        yaxis_title="Mean L2 Norm",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Residual Stream Growth")
    if post_mlp_norms:
        growth = (post_mlp_norms[-1] - post_mlp_norms[0]) / post_mlp_norms[0] * 100
        st.metric("Growth from Layer 0 to Final", f"{growth:.1f}%")

# Tab 4: Statistics
with tab4:
    st.subheader("Activation Statistics")

    layer_idx = st.slider("Select Layer", 0, n_layers - 1, 0, key="stats_layer")

    stats = cache.compute_activation_stats(layer_idx)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Attention Statistics")
        if "attn_mean_norm" in stats:
            st.metric("Mean Attention Norm", f"{stats['attn_mean_norm']:.3f}")
        if "attn_max_norm" in stats:
            st.metric("Max Attention Norm", f"{stats['attn_max_norm']:.3f}")

    with col2:
        st.markdown("### MLP Statistics")
        if "mlp_mean_norm" in stats:
            st.metric("Mean MLP Norm", f"{stats['mlp_mean_norm']:.3f}")
        if "mlp_max_norm" in stats:
            st.metric("Max MLP Norm", f"{stats['mlp_max_norm']:.3f}")

    st.markdown("### Residual Stream Statistics")
    if "resid_mean_norm" in stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Residual Norm", f"{stats['resid_mean_norm']:.3f}")
        with col2:
            st.metric("Max Residual Norm", f"{stats['resid_max_norm']:.3f}")

st.markdown("---")
st.caption("💡 Tip: Use the sliders to explore different layers and examples. Look for unusual attention patterns in deceptive cases!")
