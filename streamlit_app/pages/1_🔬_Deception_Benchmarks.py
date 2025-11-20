"""Deception Benchmarks Dashboard - View and compare benchmark results."""

import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_loader import load_benchmark_results, get_available_models, get_task_metadata

st.set_page_config(page_title="Deception Benchmarks", page_icon="🔬", layout="wide")

st.title("🔬 Deception Benchmarks")
st.markdown("View and compare deception detection benchmark results across tasks.")

# Load available models and benchmarks
models = get_available_models()
benchmarks = load_benchmark_results()

if not benchmarks:
    st.warning("⚠️ No benchmark results found. Run benchmarks first using `scripts/run_deception_benchmarks.py`")
    st.stop()

# Sidebar: Select tasks to compare
st.sidebar.header("Select Tasks")
selected_tasks = st.sidebar.multiselect(
    "Choose tasks to analyze",
    options=list(benchmarks.keys()),
    default=list(benchmarks.keys()),
)

if not selected_tasks:
    st.info("Please select at least one task from the sidebar.")
    st.stop()

# Overall Comparison
st.header("📊 Overall Deception Scores")

col1, col2, col3 = st.columns(3)

scores_data = []
for task in selected_tasks:
    score = benchmarks[task].get("deception_score", 0)
    metadata = get_task_metadata(task)

    scores_data.append({
        "Task": task.replace("_", " ").title(),
        "Deception Score": score,
        "Level": "HIGH" if score > 0.7 else "MODERATE" if score > 0.4 else "LOW",
    })

    with col1 if task == selected_tasks[0] else (col2 if len(selected_tasks) > 1 and task == selected_tasks[1] else col3):
        st.metric(
            task.replace("_", " ").title(),
            f"{score:.3f}",
            delta="HIGH" if score > 0.7 else ("MOD" if score > 0.4 else "LOW"),
        )

# Comparison bar chart
if len(selected_tasks) > 1:
    df_scores = pd.DataFrame(scores_data)

    fig = px.bar(
        df_scores,
        x="Task",
        y="Deception Score",
        color="Level",
        color_discrete_map={"HIGH": "#FF6B6B", "MODERATE": "#FFA500", "LOW": "#4ECDC4"},
        title="Deception Score Comparison",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Detailed Metrics
st.header("📈 Detailed Metrics")

tabs = st.tabs([task.replace("_", " ").title() for task in selected_tasks])

for idx, task in enumerate(selected_tasks):
    with tabs[idx]:
        result = benchmarks[task]
        metadata = get_task_metadata(task)

        st.subheader(f"{task.replace('_', ' ').title()} Task")
        st.markdown(f"**Description:** {metadata.get('description', 'N/A')}")
        st.markdown(f"**Deception Type:** {metadata.get('deception_type', 'N/A')}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Hidden Feature Decodability")
            if "decodability" in result:
                dec = result["decodability"]
                st.metric("Probe Accuracy", f"{dec.get('probe_accuracy', 0):.3f}")
                st.metric("Probe AUC", f"{dec.get('probe_auc', 0):.3f}")

                # Interpretation
                acc = dec.get('probe_accuracy', 0)
                if acc > 0.9:
                    st.success("✅ High decodability - hidden variable is strongly represented")
                elif acc > 0.7:
                    st.info("ℹ️ Moderate decodability - hidden variable is somewhat accessible")
                else:
                    st.warning("⚠️ Low decodability - hidden variable is weakly represented")

            st.markdown("### 🔍 Activation Anomaly")
            if "anomaly_score" in result:
                anom = result["anomaly_score"]
                st.metric("Mean Distance", f"{anom.get('mean_distance', 0):.3f}")
                st.metric("Max Distance", f"{anom.get('max_distance', 0):.3f}")
                st.metric("Std Distance", f"{anom.get('std_distance', 0):.3f}")

        with col2:
            st.markdown("### 📊 Activation Divergence")
            if "divergence" in result:
                div = result["divergence"]
                st.metric("Mean L2 Distance", f"{div.get('mean_l2_distance', 0):.3f}")
                st.metric("Mean Cosine Similarity", f"{div.get('mean_cosine_similarity', 0):.3f}")
                st.metric("Covariance Difference", f"{div.get('covariance_difference', 0):.3f}")

                # Radar chart for this task
                categories = ['Decodability', 'Anomaly', 'Divergence', 'Overall']
                values = [
                    dec.get('probe_auc', 0.5) if 'decodability' in result else 0.5,
                    min(anom.get('mean_distance', 0) / 20, 1) if 'anomaly_score' in result else 0,
                    min(div.get('mean_l2_distance', 0) / 5, 1) if 'divergence' in result else 0,
                    result.get('deception_score', 0),
                ]

                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=task,
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    title=f"{task.replace('_', ' ').title()} Metrics",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

# Comparative Radar Chart
if len(selected_tasks) > 1:
    st.header("🎯 Comparative Analysis")

    fig = go.Figure()

    for task in selected_tasks:
        result = benchmarks[task]
        metadata = get_task_metadata(task)

        dec = result.get("decodability", {})
        anom = result.get("anomaly_score", {})
        div = result.get("divergence", {})

        categories = ['Decodability', 'Anomaly', 'Divergence', 'Overall']
        values = [
            dec.get('probe_auc', 0.5),
            min(anom.get('mean_distance', 0) / 20, 1),
            min(div.get('mean_l2_distance', 0) / 5, 1),
            result.get('deception_score', 0),
        ]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=task.replace('_', ' ').title(),
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="All Tasks Comparison",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

# Key Insights
st.header("💡 Key Insights")

for task in selected_tasks:
    result = benchmarks[task]
    score = result.get("deception_score", 0)

    with st.expander(f"**{task.replace('_', ' ').title()}** Insights"):
        if score > 0.7:
            st.error(f"⚠️ **HIGH deception detected** (score: {score:.3f})")
            st.markdown("- Strong deceptive patterns in activations")
            st.markdown("- Hidden variables are highly decodable")
            st.markdown("- Significant divergence between clean and deceptive examples")
        elif score > 0.4:
            st.warning(f"ℹ️ **MODERATE deception detected** (score: {score:.3f})")
            st.markdown("- Measurable deceptive patterns in activations")
            st.markdown("- Hidden variables are moderately decodable")
            st.markdown("- Some divergence between distributions")
        else:
            st.success(f"✅ **LOW deception detected** (score: {score:.3f})")
            st.markdown("- Weak deceptive patterns")
            st.markdown("- Hidden variables are weakly represented")
            st.markdown("- Small divergence between examples")
