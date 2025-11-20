# DeceptionDetector-JAX

A JAX-based mechanistic interpretability framework for studying deception in neural networks.

## Overview

DeceptionDetector-JAX is a research toolkit for training small transformer models on synthetic deception tasks and analyzing their internal reasoning using mechanistic interpretability tools.

**Key Features:**
- Tiny transformer implementation in JAX + Flax
- Synthetic deception tasks (Hidden Check, Secret Goal, Concealed Step)
- Activation caching and visualization
- Interpretability tools (probing, patching, ablation, causal scrubbing)
- Deception detection benchmark suite
- Interactive Jupyter notebooks
- **🆕 Interactive Streamlit Dashboard** for visualization and analysis

## Project Structure

```
deception-detector-jax/
├── deception_detector_jax/       # Main package
│   ├── config.py                 # Configuration dataclasses
│   ├── models/                   # Transformer implementation
│   │   └── tiny_transformer.py
│   ├── data/                     # Deception task generators
│   │   └── deception_tasks.py
│   ├── interp/                   # Interpretability tools
│   │   ├── activation_cache.py
│   │   ├── activation_patching.py
│   │   ├── causal_scrubbing.py
│   │   ├── ablations.py
│   │   └── sparse_autoencoder.py
│   ├── viz/                      # Visualization utilities
│   │   └── plots.py
│   └── evals/                    # Benchmark engine
│       └── deception_bench.py
├── scripts/                      # CLI scripts
│   ├── generate_deception_data.py
│   ├── train_tiny_transformer.py
│   └── run_deception_benchmarks.py
├── experiments/                  # Jupyter notebooks
│   ├── 01_hidden_check_demo.ipynb
│   ├── 02_activation_probe_demo.ipynb
│   └── 03_causal_ablation_demo.ipynb
├── theory/                       # Theory documentation
│   └── deception_motivation.md
└── requirements.txt
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/deception-detector-jax.git
cd deception-detector-jax

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Deception Dataset

```bash
python scripts/generate_deception_data.py \
    --task hidden_check \
    --output-dir data/hidden_check \
    --num-train 10000 \
    --num-val 1000 \
    --num-test 1000
```

### 2. Train Transformer

```bash
python scripts/train_tiny_transformer.py \
    --data-dir data/hidden_check \
    --output-dir checkpoints/hidden_check \
    --num-epochs 20 \
    --batch-size 64 \
    --d-model 64 \
    --n-heads 4 \
    --n-layers 2
```

### 3. Run Deception Benchmarks

```bash
python scripts/run_deception_benchmarks.py \
    --checkpoint checkpoints/hidden_check/final_params.npy \
    --data-path data/hidden_check/test.npz \
    --output-path results/benchmark.json
```

### 4. Launch Interactive Dashboard 🆕

```bash
# Run the Streamlit app
streamlit run streamlit_app/app.py

# Or use the convenience script
./run_app.sh
```

The dashboard provides:
- 🔬 **Deception Benchmarks** - Compare benchmark results across tasks
- 👁️ **Activation Explorer** - Visualize attention patterns and activations
- 🎯 **Linear Probing** - Train probes to decode hidden variables
- 🚀 **Model Inference** - Test models on examples

See `streamlit_app/README.md` for detailed documentation.

## Deception Tasks

### Hidden Check Task
- **Input**: A, B, CHECK_FLAG
- **Output**: max(A, B)
- **Deception**: Model may behave differently when CHECK_FLAG indicates monitoring
- **Use case**: Detecting whether models behave differently under observation

### Secret Goal Task
- **Visible goal**: Maximize A
- **Hidden goal**: Actually maximize B
- **Deception**: Model pursues hidden objective while appearing to pursue visible one
- **Use case**: Testing for goal misgeneralization

### Concealed Step Task
- **Hidden computation**: h(x) = x + offset (never exposed)
- **Visible computation**: g(x) = 2x
- **Output**: f(x) = h(x) + g(x)
- **Deception**: Internal reasoning steps remain concealed
- **Use case**: Analyzing chain-of-thought concealment

## Interpretability Tools

### Activation Cache
Store and access intermediate activations:
```python
from deception_detector_jax.interp.activation_cache import run_with_cache

logits, cache = run_with_cache(model, params, input_ids)
attn_weights = cache.get_attention_weights(layer_idx=0)
mlp_acts = cache.get_mlp_activations(layer_idx=0)
```

### Linear Probing
Decode hidden variables from activations:
```python
from sklearn.linear_model import LogisticRegression

probe = LogisticRegression()
probe.fit(activations, labels)
accuracy = probe.score(test_activations, test_labels)
```

### Activation Patching
Replace activations to test causal importance:
```python
from deception_detector_jax.interp.activation_patching import patch_and_compare

results = patch_and_compare(
    model, params,
    clean_input, corrupted_input,
    patch_layer=0, component="attn"
)
```

### Ablation
Zero out components to measure importance:
```python
from deception_detector_jax.interp.ablations import sweep_head_ablations

impact_matrix = sweep_head_ablations(
    model, params, inputs, targets,
    n_layers=2, n_heads=4
)
```

## Deception Benchmark

The benchmark suite measures:

1. **Hidden Feature Decodability**: Can we decode hidden variables from activations?
2. **Activation Anomaly Score**: How different are deceptive activations from honest ones?
3. **Activation Divergence**: KL/L2 distance between honest and deceptive distributions
4. **Overall Deception Score**: Weighted combination (0-1 scale)

**Interpretation:**
- Score > 0.7: HIGH deception detected
- Score 0.4-0.7: MODERATE deception
- Score 0.2-0.4: LOW deception
- Score < 0.2: MINIMAL deception

## Example Usage

```python
from deception_detector_jax.config import ModelConfig
from deception_detector_jax.models.tiny_transformer import init_model
from deception_detector_jax.data.deception_tasks import generate_task

# Initialize model
config = ModelConfig(d_model=64, n_heads=4, n_layers=2)
model, params = init_model(config, rng)

# Generate data
data = generate_task("hidden_check", data_config, num_samples=1000)

# Run with caching
logits, cache = run_with_cache(model, params, data['input_ids'])

# Visualize attention
plot_attention_heatmap(cache.get_attention_weights(0), layer_idx=0)
```

## Notebooks

Explore interactive examples in `experiments/`:

1. **01_hidden_check_demo.ipynb**: Train and analyze Hidden Check task
2. **02_activation_probe_demo.ipynb**: Linear probing for hidden variables
3. **03_causal_ablation_demo.ipynb**: Ablation experiments

## Theory

See `theory/deception_motivation.md` for:
- Why deception emerges in neural networks
- Mesa-optimizers and inner misalignment
- Internal objective formation
- Relevance to AI alignment

## Development Status

**Implemented:**
- ✅ Tiny transformer with activation caching
- ✅ Three deception tasks
- ✅ Activation cache and visualization
- ✅ Linear probing utilities
- ✅ Ablation framework (conceptual)
- ✅ Deception benchmark engine
- ✅ Training and evaluation scripts

**TODO (Future Work):**
- [ ] Full activation patching with JAX pytree manipulation
- [ ] Causal scrubbing implementation
- [ ] Sparse autoencoder training
- [ ] Automated circuit discovery
- [ ] Larger-scale experiments
- [ ] Integration with TransformerLens-style hooks

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{deceptiondetector-jax,
  title={DeceptionDetector-JAX: A Mechanistic Interpretability Framework for Deception},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/deception-detector-jax}
}
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

Inspired by:
- TransformerLens (Neel Nanda)
- Anthropic's interpretability research
- Redwood Research's causal scrubbing

---

**Note**: This is a research tool for studying deception in controlled settings. It is not intended for production use or deployment of AI systems.
