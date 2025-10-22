#!/bin/bash

# Simple Auto-restart Script
# This will run the training script automatically when it finishes

echo "🤖 Auto-restart monitor started at $(date)"
echo "This will automatically restart training for extended epochs when current training finishes"
echo ""

while true; do
    echo "⏳ Waiting for current training to finish..."
    
    # Wait for the training process to finish
    wait
    
    echo "🔄 Training process finished. Restarting in 30 seconds..."
    sleep 30
    
    echo "🚀 Starting extended training (epochs 201-400)..."
    sh train_vae.sh
    
    # If training completes without errors, break the loop
    if [ $? -eq 0 ]; then
        echo "✅ Training completed successfully!"
        break
    else
        echo "❌ Training failed, will retry in 60 seconds..."
        sleep 60
    fi
done

echo "🎉 All training completed at $(date)"