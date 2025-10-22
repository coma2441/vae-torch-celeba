# VAE CelebA - Clean & Organized

A Variational Autoencoder (VAE) implementation for CelebA face generation with a clean, compact structure.

## 🗂️ Project Structure

```
vae-torch-celeba/
├── vae.py              # VAE model architecture
├── train.py            # Training script  
├── inference.py        # Smart inference with auto model detection
├── utils.py            # Utility functions
├── data/               # CelebA dataset
├── models/             # Trained model files
│   ├── vae_quick_trained.pth
│   ├── vae_trained_from_scratch.pth
│   └── vae_model_20.pth
└── outputs/            # Generated images
    ├── generated_random.jpg
    ├── latent_interpolation.jpg
    └── reconstruction_test.jpg
```

## 🚀 Quick Start

### 1. Training a New Model
```bash
# Set MPS fallback for compatibility
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Train from scratch (creates models/vae_trained_from_scratch.pth)
python train.py
```

### 2. Generate Images
```bash
# Smart inference - automatically uses best available model
python inference.py
```

## 📋 Features

- **🧠 Smart Model Detection**: Automatically finds and uses the best available model
- **🚀 GPU Acceleration**: Uses MPS (Apple Silicon) or CUDA when available
- **📸 Multiple Generation Modes**: 
  - Random face generation
  - Latent space interpolation
  - Image reconstruction
- **🎯 Clean Architecture**: Organized folder structure with clear separation

## 🎨 Generated Outputs

The inference script creates three types of visualizations:

1. **Random Generation** (`outputs/generated_random.jpg`): 16 randomly generated faces
2. **Latent Interpolation** (`outputs/latent_interpolation.jpg`): Smooth morphing between faces
3. **Reconstruction** (`outputs/reconstruction_test.jpg`): Original vs reconstructed comparison

## 🔧 Configuration

- **Image Size**: 150x150 pixels
- **Latent Dimensions**: 128
- **Architecture**: Convolutional encoder-decoder with batch normalization
- **Training**: Adam optimizer with MSE reconstruction + KL divergence loss

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

Enjoy creating beautiful AI-generated faces! 🎭✨
* `genpics.py`: creates a panel of original image + 7 reconstructed ones.
* `vae_model_20.pth`: a trained VAE.

Running a trained model on a CPU is fine. 

Training on a CPU is possible, but slow: ⚡👉 GPU.
