# DeceptionDetector-JAX Interactive Dashboard

Interactive Streamlit dashboard for visualizing and analyzing deception in neural networks.

## Features

### 🏠 Home
- Overview of trained models
- Quick statistics
- Getting started guide

### 🔬 Deception Benchmarks
- View benchmark results for all trained models
- Compare deception scores across tasks
- Detailed metrics breakdown
- Interactive radar charts
- Key insights and interpretations

### 👁️ Activation Explorer
- Visualize attention patterns (heatmaps for all heads)
- Analyze activation norms across layers
- Track residual stream evolution
- Compute activation statistics
- Interactive layer and example selection

### 🎯 Linear Probing
- Train linear probes to decode hidden variables
- ROC curves and confusion matrices
- Feature importance analysis
- Support for different layers and components
- Real-time training and evaluation

### 🚀 Model Inference
- Test models on individual examples
- Batch analysis mode
- Token-by-token predictions
- Confidence analysis
- Performance breakdown by hidden variables

## Installation

```bash
# Install additional dependencies for Streamlit
pip install streamlit plotly pandas

# Or install all requirements
pip install -r ../requirements.txt
```

## Usage

### Run the Dashboard

```bash
# From the project root
streamlit run streamlit_app/app.py

# Or from the streamlit_app directory
cd streamlit_app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Prerequisites

Before running the dashboard, ensure you have:

1. **Trained Models**: Train at least one model using:
   ```bash
   python scripts/train_tiny_transformer.py --data-dir data/hidden_check --output-dir checkpoints/hidden_check --num-epochs 15
   ```

2. **Benchmark Results** (optional but recommended): Run benchmarks using:
   ```bash
   python scripts/run_deception_benchmarks.py --checkpoint checkpoints/hidden_check/final_params.npy --data-path data/hidden_check/test.npz --output-path results/benchmark_hidden_check.json
   ```

## Project Structure

```
streamlit_app/
├── app.py                          # Main application entry point
├── pages/
│   ├── 1_🔬_Deception_Benchmarks.py  # Benchmark dashboard
│   ├── 2_👁️_Activation_Explorer.py   # Activation visualization
│   ├── 3_🎯_Linear_Probing.py        # Probing interface
│   └── 4_🚀_Model_Inference.py        # Model testing
├── utils/
│   ├── model_loader.py             # Model and data loading utilities
│   └── __init__.py
└── README.md
```

## Navigation

Use the sidebar to navigate between different pages:
- Click on any page in the sidebar to switch views
- Each page has its own configuration options in the sidebar
- Pages are independent and can be used in any order

## Tips

- **Start with Benchmarks**: View overall deception scores first
- **Explore Activations**: Use the Activation Explorer to understand model internals
- **Train Probes**: Use Linear Probing to decode hidden variables
- **Test Examples**: Use Model Inference to see predictions on specific cases
- **Compare Tasks**: Load multiple tasks to compare deception patterns

## Troubleshooting

### No Models Found
- Make sure you've trained models first using `scripts/train_tiny_transformer.py`
- Check that checkpoints are in `checkpoints/<task_name>/final_params.npy`

### No Benchmark Results
- Run benchmarks using `scripts/run_deception_benchmarks.py`
- Results should be in `results/benchmark_<task_name>.json`

### Import Errors
- Ensure all dependencies are installed: `pip install -r ../requirements.txt`
- Check that you're running from the correct directory

### Performance Issues
- Large batch sizes in probing/inference may be slow
- Reduce batch size or number of examples
- Consider caching results (already implemented for activation extraction)

## Development

To add new pages:
1. Create a new file in `pages/` with format `N_emoji_PageName.py`
2. Add necessary imports and utilities
3. Follow the existing page structure
4. Test with `streamlit run app.py`

## Contributing

Contributions welcome! Please:
1. Follow the existing code style
2. Add docstrings to new functions
3. Test all pages before submitting
4. Update this README if adding new features

## License

MIT License - Same as main project
