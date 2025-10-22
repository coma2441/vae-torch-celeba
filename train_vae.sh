#!/bin/bash

# VAE Training Script with MPS Fallback
# This script trains the VAE model from scratch with visualization checkpoints

echo "🚀 Starting VAE Training with Visualization Checkpoints"
echo "======================================================="

# Set MPS fallback for Apple Silicon compatibility
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Get the script directory to ensure we're in the right place
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $PWD"
echo "🔧 Environment: PYTORCH_ENABLE_MPS_FALLBACK=1"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please make sure you have a .venv directory with PyTorch installed."
    exit 1
fi

# Check if data directory exists
if [ ! -d "data/celeba/img_align_celeba" ]; then
    echo "❌ CelebA dataset not found!"
    echo "Please make sure the dataset is in data/celeba/img_align_celeba/"
    exit 1
fi

echo "✅ Environment checks passed"
echo ""

# Create necessary directories
mkdir -p models
mkdir -p outputs/training_progress

echo "📂 Created output directories"
echo ""

# Start training
echo "🎯 Starting training with visualization checkpoints..."
echo "   • Epoch 1: Initial checkpoint"
echo "   • Every 10 epochs: Progress checkpoints"
echo "   • Final: Comprehensive visualization"
echo ""
echo "💡 Training will create:"
echo "   📁 models/vae_trained_from_scratch.pth"
echo "   📁 outputs/training_progress/epoch_*"
echo ""

# Get start time
start_time=$(date)
echo "⏰ Started at: $start_time"
echo ""

# Run training with proper Python environment
./.venv/bin/python train.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "⏰ Started at: $start_time"
    echo "⏰ Finished at: $(date)"
    echo ""
    echo "📋 Generated files:"
    echo "   💾 models/vae_trained_from_scratch.pth"
    
    # List visualization files if they exist
    if [ -d "outputs/training_progress" ] && [ "$(ls -A outputs/training_progress)" ]; then
        echo "   📸 Visualization checkpoints:"
        ls outputs/training_progress/ | sed 's/^/      /'
    fi
    
    echo ""
    echo "🚀 Ready for inference! Run: python inference.py"
    
else
    echo ""
    echo "❌ Training failed or was interrupted"
    echo "Check the output above for error details"
    exit 1
fi