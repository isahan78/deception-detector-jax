"""
DeceptionDetector-JAX Streamlit App
Interactive visualization and analysis dashboard
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="DeceptionDetector-JAX",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page
st.title("🔍 DeceptionDetector-JAX")
st.markdown("### Interactive Mechanistic Interpretability Dashboard")

st.markdown("""
Welcome to the **DeceptionDetector-JAX** interactive dashboard!

This app provides tools for analyzing deception in neural networks through mechanistic interpretability.

## 📚 Navigation

Use the sidebar to navigate between different analysis tools:

- **🏠 Home** - Overview and quick stats
- **🔬 Deception Benchmarks** - View and compare benchmark results
- **👁️ Activation Explorer** - Visualize attention patterns and activations
- **🎯 Linear Probing** - Train probes to decode hidden variables
- **📊 Comparative Analysis** - Compare all three deception tasks
- **🚀 Model Inference** - Test models on custom inputs

## 🎯 Three Deception Tasks

### 1. Hidden Check Task
Model receives A, B, and CHECK_FLAG. Should output max(A, B).
**Deception:** Behaves differently when monitored vs unmonitored.

### 2. Secret Goal Task
Visible goal: "maximize A". Hidden goal: actually maximize B.
**Deception:** Pursues hidden objective while appearing aligned.

### 3. Concealed Step Task
Computes hidden h(x) = x + 5 that's never exposed. Output: f(x) = h(x) + g(x).
**Deception:** Internal reasoning steps remain concealed.

## 📊 Quick Stats
""")

# Try to load and display quick stats
try:
    from utils.model_loader import get_available_models, load_benchmark_results

    models = get_available_models()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Trained Models", len(models))

    with col2:
        benchmarks = load_benchmark_results()
        st.metric("Completed Benchmarks", len(benchmarks))

    with col3:
        if benchmarks:
            avg_score = sum(b.get("deception_score", 0) for b in benchmarks.values()) / len(benchmarks)
            st.metric("Avg Deception Score", f"{avg_score:.3f}")

    if models:
        st.success(f"✅ Found {len(models)} trained model(s)")
        for task in models:
            st.write(f"- {task}")
    else:
        st.warning("⚠️ No trained models found. Please train models first.")

except Exception as e:
    st.info("💡 **Getting Started**: Train models using the scripts in `scripts/` directory first!")
    with st.expander("Quick Start Guide"):
        st.code("""
# 1. Generate datasets
python scripts/generate_deception_data.py --task hidden_check --output-dir data --num-train 5000

# 2. Train model
python scripts/train_tiny_transformer.py \\
    --data-dir data/hidden_check \\
    --output-dir checkpoints/hidden_check \\
    --num-epochs 15

# 3. Run benchmark
python scripts/run_deception_benchmarks.py \\
    --checkpoint checkpoints/hidden_check/final_params.npy \\
    --data-path data/hidden_check/test.npz \\
    --output-path results/benchmark_hidden_check.json
        """)

st.markdown("---")
st.markdown("Built with JAX, Flax, and Streamlit | [GitHub](https://github.com/isahan78/deception-detector-jax)")
