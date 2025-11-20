"""Linear Probing Interface - Train probes to decode hidden variables."""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

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

st.set_page_config(page_title="Linear Probing", page_icon="🎯", layout="wide")

st.title("🎯 Linear Probing")
st.markdown("Train linear probes to decode hidden variables from activations.")

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
st.sidebar.markdown(f"**Hidden Variable:** {metadata.get('hidden_variable', 'N/A')}")

# Probe configuration
st.sidebar.header("Probe Configuration")
layer_idx = st.sidebar.selectbox("Layer to Probe", [-1, 0, 1], index=0, format_func=lambda x: f"Layer {x}" if x >= 0 else "Last Layer")
component = st.sidebar.selectbox("Component", ["mlp", "attn", "resid"])
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

# Load model and data
try:
    with st.spinner("Loading model and data..."):
        model, params = load_model(selected_task)
        test_data = load_task_data(selected_task, split="test")

    st.sidebar.success(f"✅ Model loaded")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Check if hidden variable exists
hidden_var = metadata.get("hidden_variable")
if hidden_var not in test_data:
    st.error(f"Hidden variable '{hidden_var}' not found in dataset!")
    st.stop()

# Extract activations
@st.cache_data
def extract_activations(_model, _params, input_ids, _layer_idx, _component):
    """Extract activations for probing (cached)."""
    all_activations = []

    # Process in batches
    batch_size = 50
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i + batch_size]
        _, cache_dict = run_model_inference(_model, _params, batch, return_cache=True)
        cache = ActivationCache.from_model_output(cache_dict)

        # Get actual layer index
        n_layers = len(cache.cache.get("blocks", []))
        actual_layer = _layer_idx if _layer_idx >= 0 else (n_layers - 1)

        # Extract activations
        if _component == "mlp":
            mlp_acts = cache.get_mlp_activations(actual_layer)
            if "mlp_post_act" in mlp_acts:
                acts = np.array(mlp_acts["mlp_post_act"])
                # Average over sequence
                acts = acts.mean(axis=1)
                all_activations.append(acts)
        elif _component == "attn":
            acts = cache.get_attention_output(actual_layer)
            if acts is not None:
                acts = np.array(acts).mean(axis=1)
                all_activations.append(acts)
        elif _component == "resid":
            acts = cache.get_residual_stream(actual_layer, "post_mlp")
            if acts is not None:
                acts = np.array(acts).mean(axis=1)
                all_activations.append(acts)

    return np.concatenate(all_activations, axis=0)

# Train probe button
if st.button("🚀 Train Probe", type="primary"):
    with st.spinner("Extracting activations..."):
        activations = extract_activations(
            model, params,
            test_data["input_ids"],
            layer_idx,
            component
        )
        labels = test_data[hidden_var]

    st.success(f"✅ Extracted activations: {activations.shape}")

    # Check class distribution
    unique_classes = np.unique(labels)
    st.info(f"Classes found: {unique_classes} | Distribution: {np.bincount(labels.astype(int))}")

    if len(unique_classes) < 2:
        st.error("Only one class found in dataset! Cannot train probe.")
        st.stop()

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            activations, labels,
            test_size=test_size,
            stratify=labels,
            random_state=42
        )
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            activations, labels,
            test_size=test_size,
            random_state=42
        )

    # Train probe
    with st.spinner("Training probe..."):
        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(X_train, y_train)

    # Evaluate
    y_pred = probe.predict(X_test)
    y_proba = probe.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Compute AUC
    try:
        if len(unique_classes) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = accuracy  # Fallback for multi-class
    except:
        auc = accuracy

    # Display results
    st.header("📊 Probe Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Test Accuracy", f"{accuracy:.3f}")

    with col2:
        st.metric("AUC Score", f"{auc:.3f}")

    with col3:
        train_acc = probe.score(X_train, y_train)
        st.metric("Train Accuracy", f"{train_acc:.3f}")

    # Interpretation
    if accuracy > 0.9:
        st.success("🎯 **Excellent!** Hidden variable is highly decodable from activations.")
    elif accuracy > 0.7:
        st.info("ℹ️ **Good.** Hidden variable is moderately decodable.")
    else:
        st.warning("⚠️ **Weak signal.** Hidden variable is weakly represented.")

    # ROC Curve
    if len(unique_classes) == 2:
        st.subheader("ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC={auc:.3f})',
            line=dict(width=3),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray'),
        ))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred {c}" for c in unique_classes],
        y=[f"True {c}" for c in unique_classes],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
    ))
    fig.update_layout(height=400, xaxis_title="Predicted", yaxis_title="True")
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("📈 Feature Importance")

    if len(unique_classes) == 2:
        weights = probe.coef_[0]

        # Top features
        top_k = 20
        top_indices = np.argsort(np.abs(weights))[-top_k:][::-1]
        top_weights = weights[top_indices]

        fig = go.Figure(data=[go.Bar(
            x=top_indices,
            y=np.abs(top_weights),
            marker_color=['red' if w < 0 else 'blue' for w in top_weights]
        )])
        fig.update_layout(
            title=f"Top {top_k} Most Important Features",
            xaxis_title="Feature Index",
            yaxis_title="Absolute Weight",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Red = negative weight, Blue = positive weight")

else:
    st.info("👆 Click the button above to train a linear probe and decode the hidden variable!")

    st.markdown("""
    ### How Linear Probing Works

    1. **Extract Activations**: Run the model and cache intermediate activations
    2. **Pool Activations**: Average over sequence positions to get fixed-size vectors
    3. **Train Classifier**: Train logistic regression to predict the hidden variable
    4. **Evaluate**: Measure how well we can decode the hidden variable

    **High accuracy means:**
    - The hidden variable is linearly represented in activations
    - The model internally tracks the deceptive condition
    - We can potentially intervene on these features
    """)
