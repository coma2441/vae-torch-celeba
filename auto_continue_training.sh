#!/bin/bash

# Auto-continue VAE Training Script
# This script monitors the current training and automatically starts extended training when it finishes

echo "🤖 Auto-continue training monitor started"
echo "========================================"
echo "📅 $(date)"
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $PWD"
echo "🔍 Monitoring for training completion..."
echo ""

# Function to check if training is running
is_training_running() {
    # Check if there's a Python process running train.py
    pgrep -f "python.*train.py" > /dev/null
    return $?
}

# Function to check if first training completed successfully
training_completed_successfully() {
    # Check if the final model was saved and we have epoch 200 visualizations
    if [ -f "models/vae_trained_from_scratch.pth" ] && [ -f "outputs/training_progress/epoch_200_random_generation.jpg" ]; then
        return 0
    else
        return 1
    fi
}

# Wait for current training to finish
echo "⏳ Waiting for current training (epochs 1-200) to complete..."
while is_training_running; do
    sleep 60  # Check every minute
    echo "⏱️  $(date): Training still running..."
done

echo ""
echo "🎉 Current training process has finished!"
echo "⏰ Completed at: $(date)"
echo ""

# Check if training completed successfully
if training_completed_successfully; then
    echo "✅ Training completed successfully! Model and visualizations found."
    echo ""
    echo "🚀 Starting extended training (epochs 201-400)..."
    echo "⏰ Extended training started at: $(date)"
    echo ""
    
    # Start extended training
    ./.venv/bin/python train.py
    
    # Check if extended training completed
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎊 FULL TRAINING COMPLETED! 🎊"
        echo "==============================="
        echo "⏰ Started at: $(cat .training_start_time 2>/dev/null || echo 'Unknown')"
        echo "⏰ Finished at: $(date)"
        echo ""
        echo "📋 Final results:"
        echo "   💾 models/vae_trained_from_scratch.pth (400 epochs)"
        echo "   📸 All visualization checkpoints in outputs/training_progress/"
        echo ""
        echo "🌟 Your VAE is now fully trained and ready for inference!"
        echo "🚀 Run: python inference.py"
        
        # Save completion marker
        echo "$(date)" > .training_completed
        
    else
        echo ""
        echo "❌ Extended training failed"
        echo "Check the output above for error details"
    fi
    
else
    echo "❌ Initial training may not have completed successfully"
    echo "Please check:"
    echo "   - models/vae_trained_from_scratch.pth exists"
    echo "   - outputs/training_progress/epoch_200_* files exist"
    echo ""
    echo "You may need to restart training manually with: sh train_vae.sh"
fi

echo ""
echo "🤖 Auto-continue script finished at: $(date)"