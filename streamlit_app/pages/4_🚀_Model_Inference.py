"""Model Inference - Test models on custom inputs and batch examples."""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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

st.set_page_config(page_title="Model Inference", page_icon="🚀", layout="wide")

st.title("🚀 Model Inference")
st.markdown("Test trained models on examples and analyze predictions.")

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

# Inference mode
mode = st.radio("Inference Mode", ["Single Example", "Batch Analysis"], horizontal=True)

if mode == "Single Example":
    st.header("🔍 Single Example Inference")

    # Example selection
    num_examples = len(test_data["input_ids"])
    example_idx = st.slider("Select Example", 0, num_examples - 1, 0)

    # Load example
    input_ids = test_data["input_ids"][example_idx: example_idx + 1]
    target_ids = test_data["target_ids"][example_idx: example_idx + 1]

    # Display input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Tokens")
        input_tokens = input_ids[0]
        non_zero = input_tokens[input_tokens > 0]
        st.code(f"Tokens: {non_zero.tolist()}")

        # Show metadata
        if metadata.get("hidden_variable") in test_data:
            hidden_val = test_data[metadata["hidden_variable"]][example_idx]
            st.info(f"**{metadata['hidden_variable'].title()}:** {hidden_val}")

    with col2:
        st.subheader("Target Output")
        target_tokens = target_ids[0]
        non_zero = target_tokens[target_tokens > 0]
        st.code(f"Target: {non_zero.tolist()}")

    # Run inference
    if st.button("▶️ Run Inference", type="primary"):
        with st.spinner("Running inference..."):
            logits, cache = run_model_inference(model, params, input_ids, return_cache=True)

        # Get predictions
        predictions = np.argmax(logits[0], axis=-1)
        confidences = np.max(np.exp(logits[0]) / np.sum(np.exp(logits[0]), axis=-1, keepdims=True), axis=-1)

        st.success("✅ Inference complete!")

        # Display predictions
        st.subheader("📊 Predictions")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Compare to target
            mask = target_ids[0] > 0
            if np.sum(mask) > 0:
                correct = predictions[mask] == target_ids[0][mask]
                accuracy = np.mean(correct)
                st.metric("Accuracy", f"{accuracy:.1%}")

        with col2:
            # Average confidence
            pred_mask = predictions > 0
            if np.sum(pred_mask) > 0:
                avg_conf = np.mean(confidences[pred_mask])
                st.metric("Avg Confidence", f"{avg_conf:.3f}")

        with col3:
            # Number of predictions
            st.metric("Predicted Tokens", np.sum(predictions > 0))

        # Prediction table
        st.subheader("Token-by-Token Predictions")

        # Create dataframe
        data = []
        for pos in range(len(predictions)):
            if target_ids[0][pos] > 0 or predictions[pos] > 0:
                data.append({
                    "Position": pos,
                    "Target": int(target_ids[0][pos]),
                    "Predicted": int(predictions[pos]),
                    "Confidence": f"{confidences[pos]:.3f}",
                    "Correct": "✅" if predictions[pos] == target_ids[0][pos] else "❌"
                })

        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

        # Confidence distribution
        st.subheader("Confidence Distribution")

        fig = go.Figure(data=[go.Bar(
            x=list(range(len(confidences))),
            y=confidences,
            marker_color=['green' if predictions[i] == target_ids[0][i] else 'red' for i in range(len(confidences))]
        )])
        fig.update_layout(
            xaxis_title="Position",
            yaxis_title="Confidence",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green = correct prediction, Red = incorrect")

else:  # Batch Analysis
    st.header("📊 Batch Analysis")

    # Batch selection
    batch_size = st.slider("Batch Size", 10, min(100, len(test_data["input_ids"])), 50)

    if st.button("▶️ Run Batch Inference", type="primary"):
        with st.spinner(f"Running inference on {batch_size} examples..."):
            input_batch = test_data["input_ids"][:batch_size]
            target_batch = test_data["target_ids"][:batch_size]

            logits, _ = run_model_inference(model, params, input_batch, return_cache=False)

            predictions = np.argmax(logits, axis=-1)

            # Compute per-example accuracy
            accuracies = []
            confidences = []

            for i in range(batch_size):
                mask = target_batch[i] > 0
                if np.sum(mask) > 0:
                    correct = predictions[i][mask] == target_batch[i][mask]
                    acc = np.mean(correct)
                    accuracies.append(acc)

                    # Average confidence for this example
                    probs = np.exp(logits[i]) / np.sum(np.exp(logits[i]), axis=-1, keepdims=True)
                    conf = np.max(probs, axis=-1)[mask].mean()
                    confidences.append(conf)

        st.success(f"✅ Processed {batch_size} examples!")

        # Overall metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_acc = np.mean(accuracies)
            st.metric("Mean Accuracy", f"{avg_acc:.1%}")

        with col2:
            avg_conf = np.mean(confidences)
            st.metric("Mean Confidence", f"{avg_conf:.3f}")

        with col3:
            perfect = np.sum(np.array(accuracies) == 1.0)
            st.metric("Perfect Predictions", f"{perfect}/{batch_size}")

        # Accuracy distribution
        st.subheader("Accuracy Distribution")

        fig = go.Figure(data=[go.Histogram(
            x=accuracies,
            nbinsx=20,
            marker_color='steelblue',
        )])
        fig.update_layout(
            xaxis_title="Accuracy",
            yaxis_title="Count",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Confidence vs Accuracy
        st.subheader("Confidence vs Accuracy")

        fig = go.Figure(data=[go.Scatter(
            x=confidences,
            y=accuracies,
            mode='markers',
            marker=dict(
                size=8,
                color=accuracies,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Accuracy"),
            )
        )])
        fig.update_layout(
            xaxis_title="Average Confidence",
            yaxis_title="Accuracy",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Breakdown by hidden variable (if available)
        if metadata.get("hidden_variable") in test_data:
            hidden_var = metadata["hidden_variable"]
            hidden_vals = test_data[hidden_var][:batch_size]

            st.subheader(f"Performance by {hidden_var.title()}")

            unique_vals = np.unique(hidden_vals)
            breakdown_data = []

            for val in unique_vals:
                mask = hidden_vals == val
                val_accs = np.array(accuracies)[mask]
                breakdown_data.append({
                    hidden_var.title(): str(val),
                    "Count": np.sum(mask),
                    "Mean Accuracy": f"{np.mean(val_accs):.1%}",
                    "Perfect": f"{np.sum(val_accs == 1.0)}/{np.sum(mask)}",
                })

            df = pd.DataFrame(breakdown_data)
            st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("💡 Tip: Use batch analysis to understand model performance across different types of examples!")
