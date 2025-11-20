#!/bin/bash

# Run the DeceptionDetector-JAX Streamlit Dashboard

echo "🚀 Starting DeceptionDetector-JAX Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit is not installed!"
    echo "Install it with: pip install streamlit plotly pandas"
    exit 1
fi

# Check if trained models exist
if [ ! -d "checkpoints" ] || [ -z "$(ls -A checkpoints 2>/dev/null)" ]; then
    echo "⚠️  Warning: No trained models found in checkpoints/"
    echo "Train models first using: python scripts/train_tiny_transformer.py"
    echo ""
fi

# Run streamlit
streamlit run streamlit_app/app.py
