#!/bin/bash

# Final Auto-continue Training Script for Overnight Execution
# This will monitor and automatically continue your VAE training

echo "🌙 Overnight Auto-training Monitor Started"
echo "=========================================="
echo "📅 Started at: $(date)"
echo ""
echo "This script will:"
echo "✅ Monitor current training progress"
echo "✅ Automatically restart when training finishes"
echo "✅ Run the full 800 epochs while you sleep"
echo ""

# Save start time
echo "$(date)" > .training_start_time

# Function to check if python training is running
is_training_running() {
    pgrep -f "python.*train.py" > /dev/null 2>&1 || pgrep -f "sh.*train_vae.sh" > /dev/null 2>&1
    return $?
}

# Main monitoring loop
while true; do
    if is_training_running; then
        echo "⏳ $(date): Training is running... (checking again in 5 minutes)"
        sleep 300  # Check every 5 minutes
    else
        echo ""
        echo "🎯 $(date): No training process detected"
        echo "🚀 Starting/Restarting VAE training..."
        echo ""
        
        # Start training in background and capture PID
        nohup sh train_vae.sh > training_output.log 2>&1 &
        TRAIN_PID=$!
        
        echo "✅ Training started with PID: $TRAIN_PID"
        echo "📝 Output being logged to: training_output.log"
        echo ""
        
        # Wait a bit and check if it started successfully
        sleep 30
        if kill -0 $TRAIN_PID 2>/dev/null; then
            echo "✅ Training process is running successfully"
        else
            echo "❌ Training process failed to start, will retry in 2 minutes..."
            sleep 120
            continue
        fi
        
        # Wait for this training session to complete
        while kill -0 $TRAIN_PID 2>/dev/null; do
            echo "⏳ $(date): Training in progress... (next check in 10 minutes)"
            sleep 600  # Check every 10 minutes
        done
        
        echo ""
        echo "🏁 $(date): Training session completed"
        
        # Check if we've reached 800 epochs by looking for final outputs
        if [ -f "outputs/training_progress/epoch_800_random_generation.jpg" ]; then
            echo ""
            echo "🎊 CONGRATULATIONS! 🎊"
            echo "======================"
            echo "✅ Full 800-epoch training completed!"
            echo "📅 Started: $(cat .training_start_time 2>/dev/null || echo 'Unknown')"
            echo "📅 Finished: $(date)"
            echo ""
            echo "📁 Your trained model is ready:"
            echo "   💾 models/vae_trained_from_scratch.pth"
            echo "   📸 outputs/training_progress/ (all visualizations)"
            echo ""
            echo "🚀 Ready for inference: python inference.py"
            echo ""
            echo "😴 You can wake up to a fully trained VAE! 🌅"
            
            # Mark completion
            echo "$(date)" > .training_completed_800_epochs
            break
        else
            echo "🔄 Training session finished, but not at 800 epochs yet"
            echo "   Will restart training in 1 minute..."
            sleep 60
        fi
    fi
done

echo ""
echo "🌅 Auto-training monitor finished at: $(date)"
echo "Sweet dreams! 😴"