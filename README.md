# VAE CelebA - Clean & Organized

A Variational Autoencoder (VAE) implementation for CelebA face generation with comprehensive training visualizations and smart inference capabilities.

## 🗂️ Project Structure

```
vae-torch-celeba/
├── vae.py              # VAE model architecture
├── train.py            # Training script with visualization checkpoints
├── train_vae.sh        # Automated training script with MPS fallback
├── inference.py        # Smart inference with auto model detection
├── utils.py            # Utility functions
├── data/               # CelebA dataset
│   └── celeba/
│       └── img_align_celeba/    # Face images (*.jpg)
├── models/             # Trained model files
│   ├── vae_quick_trained.pth
│   ├── vae_trained_from_scratch.pth
│   └── vae_model_20.pth
└── outputs/            # Generated images and training progress
    ├── generated_random.jpg
    ├── latent_interpolation.jpg
    ├── reconstruction_test.jpg
    └── training_progress/       # Visualization checkpoints
        ├── epoch_001_*.jpg      # Initial results
        ├── epoch_010_*.jpg      # 10-epoch checkpoint  
        ├── epoch_020_*.jpg      # 20-epoch checkpoint
        └── epoch_final_*.jpg    # Final results
```

## 🚀 Quick Start

### 1. Automated Training (Recommended)
```bash
# Run complete training with visualization checkpoints
./train_vae.sh
```

### 2. Manual Training
```bash
# Set MPS fallback for Apple Silicon compatibility
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Train from scratch (creates models/vae_trained_from_scratch.pth)
python train.py
```

### 3. Generate Images
```bash
# Smart inference - automatically uses best available model
python inference.py
```

## 📋 Features

### 🧠 **Smart Model Management**
- **Auto Model Detection**: Automatically finds and uses the best available model
- **Model Priority**: Quick trained → Full trained → Original pretrained
- **Organized Storage**: All models saved in `models/` directory

### 🎨 **Comprehensive Visualizations**
- **Training Checkpoints**: Visual progress at epochs 1, 10, 20, and final
- **Three Visualization Types**:
  - **Random Generation**: 16 diverse generated faces
  - **Latent Interpolation**: Smooth morphing between faces  
  - **Reconstruction**: Original vs reconstructed comparisons
- **Progress Tracking**: Watch quality improve over training epochs

### 🚀 **GPU Acceleration** 
- **MPS Support**: Optimized for Apple Silicon (M1/M2/M3)
- **CUDA Support**: Works with NVIDIA GPUs
- **Automatic Fallback**: CPU operations when GPU doesn't support specific ops
- **Smart Device Detection**: Automatically uses best available device

### 📸 **Multiple Generation Modes**
- **Random Face Generation**: Create diverse new faces
- **Latent Space Interpolation**: Smooth transitions between faces
- **Image Reconstruction**: Test model's understanding of real faces

## 🎨 Generated Outputs

### **Inference Results** (`outputs/`)
1. **Random Generation** (`generated_random.jpg`): 16 randomly generated faces
2. **Latent Interpolation** (`latent_interpolation.jpg`): Smooth morphing between faces
3. **Reconstruction** (`reconstruction_test.jpg`): Original vs reconstructed comparison

### **Training Progress** (`outputs/training_progress/`)
Each checkpoint includes three visualization types:
- `epoch_XXX_random_generation.jpg` - Generated faces at that epoch
- `epoch_XXX_interpolation.jpg` - Latent space interpolations
- `epoch_XXX_reconstruction.jpg` - Reconstruction quality test

## 🔧 Configuration

- **Image Size**: 150x150 pixels
- **Latent Dimensions**: 128
- **Architecture**: Convolutional encoder-decoder with batch normalization
- **Training**: 20 epochs, batch size 64, Adam optimizer
- **Loss Function**: MSE reconstruction + KL divergence (β=0.00025)
- **Dataset**: 10,000 CelebA images (8,000 train + 2,000 test)

## 📊 Model Priority

The inference script automatically selects models in this order:
1. `models/vae_quick_trained.pth` (if available)
2. `models/vae_trained_from_scratch.pth` (if available)  
3. `models/vae_model_20.pth` (original pretrained)

## 🛠️ Requirements

- PyTorch with MPS/CUDA support
- torchvision  
- Pillow
- CelebA dataset in `data/celeba/img_align_celeba/`

## 📈 Training Visualization Schedule

- **Epoch 1**: Initial checkpoint (see early learning)
- **Epoch 10**: Quality improvement checkpoint  
- **Epoch 20**: Final epoch checkpoint
- **Final**: Comprehensive final visualization

## 🎯 Benefits

- **👁️ Visual Progress Tracking**: See how faces improve over time
- **🐛 Early Problem Detection**: Spot issues like mode collapse quickly  
- **📈 Training Insights**: Understand what the model learns when
- **🎨 Beautiful Results**: Create a training progress story
- **🚀 Easy Automation**: One-click training with `./train_vae.sh`

## 💡 Usage Tips

- Use `./train_vae.sh` for hassle-free training with all optimizations
- Check `outputs/training_progress/` during training to monitor quality
- The script handles MPS fallback automatically for Apple Silicon compatibility
- Training takes ~20-30 minutes on Apple Silicon GPUs

Enjoy creating beautiful AI-generated faces! 🎭✨
* `genpics.py`: creates a panel of original image + 7 reconstructed ones.

Running a trained model on a CPU is fine. 

Training on a CPU is possible, but slow: ⚡👉 GPU.
