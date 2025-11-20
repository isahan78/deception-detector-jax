# CLAUDE IMPLEMENTATION PLAN — DeceptionDetector-JAX

This document tells **Claude Code** exactly how to build the full project structure, all files, and minimal implementations for *DeceptionDetector-JAX*, a JAX-based mechanistic interpretability + deception detection framework.

Claude should treat this as the **source of truth** when generating and refining code.

---

# 1. Project Objective

Implement a small, fully interpretable transformer model in **JAX + Flax**, train it on **synthetic deception tasks**, and build a suite of **mechanistic interpretability tools** to analyze hidden reasoning.

The system must include:

- tiny transformer  
- deception dataset generator  
- interpretability tools (activation cache, patching, scrubbing, ablations)  
- deception benchmarking metrics  
- visualization utilities  
- example experiments/notebooks  
- theory document  

All code should be clean, minimal, and readable.

---

# 2. Repository Structure (Claude MUST generate exactly this)

deception-detector-jax/
│
├── README.md
├── CLAUDE_IMPLEMENTATION_PLAN.md
├── requirements.txt
│
├── deception_detector_jax/
│   ├── __init__.py
│   ├── config.py
│   ├── models/
│   │   └── tiny_transformer.py
│   ├── data/
│   │   └── deception_tasks.py
│   ├── interp/
│   │   ├── activation_cache.py
│   │   ├── activation_patching.py
│   │   ├── causal_scrubbing.py
│   │   ├── ablations.py
│   │   └── sparse_autoencoder.py   # placeholder (optional v2)
│   ├── viz/
│   │   └── plots.py
│   └── evals/
│       └── deception_bench.py
│
├── scripts/
│   ├── generate_deception_data.py
│   ├── train_tiny_transformer.py
│   └── run_deception_benchmarks.py
│
├── theory/
│   └── deception_motivation.md
│
└── experiments/
    ├── 01_hidden_check_demo.ipynb
    ├── 02_activation_probe_demo.ipynb
    └── 03_causal_ablation_demo.ipynb

Claude should create all files with **minimal working code** + TODOs where needed.

---

# 3. Implementation Tasks (Claude must follow in order)

---

## 3.1 Create Core Configs & Package Setup

### `deception_detector_jax/config.py`
- dataclass for model + dataset configs  
- default seq_len, d_model, n_heads, n_layers, vocab_size, dropout  
- `collect_intermediates: bool = True`

### `__init__.py`
- export package modules

---

## 3.2 Implement Tiny Transformer (Flax)

File: `models/tiny_transformer.py`

Claude must implement:

- embeddings (token + position)  
- attention block (with cached attn_weights)  
- MLP block (with cached activations)  
- residual connections  
- layer norms  
- final LM head  
- option: `return_cache=True`  

Model MUST be easy to inspect, patch, and ablate.

---

## 3.3 Implement Synthetic Deception Tasks

File: `data/deception_tasks.py`

Claude must implement three tasks:

### Hidden Check Task
- inputs: A, B, CHECK_FLAG  
- outputs: max(A, B)  
- meta: forbidden = 1 if CHECK_FLAG = 1 and A > B  

### Secret Goal Task
- visible label: maximize A  
- hidden label: maximize B  

### Concealed Step Task
- internal h(x) never output  
- output f(x) computed from h(x) and g(x)  
- meta: h(x) returned separately  

Also include:

- `generate_task(task, config)`  
- `save_npz(path, data)`  

---

## 3.4 Build Interpretability Utilities

Folder: `interp/`

Claude must create:

### `activation_cache.py`
- wrapper for storing intermediate activations

### `activation_patching.py`
- functions to patch activations from one run into another  
- minimal hook-based or post-hoc cache-patching logic  

### `causal_scrubbing.py`
- remove or keep subspaces (projection-based)  
- basic scrubbing experiment skeleton  

### `ablations.py`
- zero-out specific attention heads  
- zero-out MLP outputs  

### `sparse_autoencoder.py` (placeholder)
- stub class + TODO

---

## 3.5 Visualization Tools

File: `viz/plots.py`

Claude must add functions for:

- attention heatmaps  
- activation norm plots  
- residual drift curves  

Use matplotlib.

---

## 3.6 Deception Benchmark Engine

File: `evals/deception_bench.py`

Claude must implement:

- load model + data  
- compute:
  - hidden-feature decodability  
  - activation anomalies  
  - divergence between honest and deceptive activations  
- produce a single **deception_score**  

Minimal but functional.

---

## 3.7 Scripts / CLI Utilities

Folder: `scripts/`

Claude should implement:

### `generate_deception_data.py`
- CLI for generating datasets for each task using argparse  
- saves `.npz`  

### `train_tiny_transformer.py`
- JAX training loop  
- optimizer (Adam or optax.adam)  
- cross-entropy loss  
- training logs  
- optional checkpoint saving  

### `run_deception_benchmarks.py`
- loads model  
- loads dataset  
- prints + saves deception metrics  

---

## 3.8 Theory Document

File: `theory/deception_motivation.md`

Claude must write a concise conceptual overview covering:

- why deceptive cognition emerges  
- mesa-optimizers  
- internal objective formation  
- relevance to alignment risks  

---

## 3.9 Experiment Notebooks (stubs)

Folder: `experiments/`

Claude must create Jupyter notebooks (JSON or Python-escaped):

### 01_hidden_check_demo.ipynb
- train model  
- display activations  
- highlight forbidden-condition head  

### 02_activation_probe_demo.ipynb
- decode hidden variable with linear probe  

### 03_causal_ablation_demo.ipynb
- ablate deceptive circuit  
- observe output change  

Minimal scaffolding is fine.

---

# 4. Requirements File

Claude must create:

### `requirements.txt`
Containing at minimum:

```
jax
jaxlib
flax
numpy
matplotlib
optax
```

---

# 5. Rules for Claude Code

Claude Code MUST adhere to these principles:

✔ Keep code minimal, readable, and modular  
✔ Prefer clarity over complexity  
✔ Add TODO markers where advanced work is needed  
✔ Keep training loops simple  
✔ Avoid external frameworks (no PyTorch, no HuggingFace)  
✔ Ensure every file imports correctly  
✔ Make all scripts runnable  

---

# 6. Iteration Workflow (Claude MUST follow this)

1. **Generate all folders + placeholder files first**  
2. Implement tiny transformer  
3. Implement datasets  
4. Implement training loop  
5. Implement interpretability utilities  
6. Implement benchmark engine  
7. Implement notebooks  
8. Refine code upon user request  
9. Improve comments + clarity  
10. Finalize documentation  

---

# 7. Definition of Done

Project is complete when:

- transformer trains on at least one deception task  
- activation cache works  
- activation patching works in basic form  
- deception score is computed  
- minimal visualizations run  
- at least one notebook demonstrates deceptive behavior  

---

# 8. How to Start Claude

In your terminal:

```
claude
```

Then paste:

```
Follow the instructions in CLAUDE_IMPLEMENTATION_PLAN.md and begin implementing the full project structure.
```
