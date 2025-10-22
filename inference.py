#!/usr/bin/env python3
"""
Smart Inference Script - Automatically detects and uses available trained models
"""
import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from vae import VAE, IMAGE_SIZE, LATENT_DIM, celeb_transform

def find_best_model():
    """Find the best available trained model"""
    # Priority order: quick trained -> full trained -> original pretrained
    models = [
        ("models/vae_quick_trained.pth", "Quick trained model (5 epochs)"),
        ("models/vae_trained_from_scratch.pth", "Full trained model (20 epochs)"),
        ("models/vae_model_20.pth", "Original pretrained model")
    ]
    
    for model_path, description in models:
        if os.path.exists(model_path):
            return model_path, description
    
    return None, None

def load_model(model_path, description):
    """Load the VAE model"""
    # Auto-detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    print(f"📋 Loading: {description}")
    print(f"📁 Model file: {model_path}")
    
    # Handle different model formats
    if model_path == "vae_model_20.pth":
        # Original pretrained model (complete object)
        try:
            model = torch.load(model_path, map_location=device, weights_only=False)
            print("✅ Loaded complete model object")
        except Exception as e:
            print(f"❌ Failed to load original model: {e}")
            return None, None
    else:
        # Our trained models (state dict)
        model = VAE()
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print("✅ Loaded model state dict")
        except Exception as e:
            print(f"❌ Failed to load trained model: {e}")
            return None, None
    
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    return model, device

def generate_random_images(model, device, num_samples=8, filename="outputs/generated_random.jpg"):
    """Generate images from random latent vectors"""
    print(f"\n📸 Generating {num_samples} random images...")
    
    model.eval()
    with torch.no_grad():
        # Sample random latent vectors
        z = torch.randn(num_samples, LATENT_DIM).to(device)
        
        # Generate images
        try:
            generated = model.decode(z)
            if generated.dim() == 2:  # Flattened output
                images = generated.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
            else:  # Already image format
                images = generated
            
            # Save images
            save_image(images, filename, normalize=True, nrow=4)
            print(f"💾 Saved as: {filename}")
            
            return images
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None

def interpolate_latent_space(model, device, steps=8, filename="latent_interpolation.jpg"):
    """Interpolate between two random points in latent space"""
    print(f"\n🎨 Creating latent space interpolation with {steps} steps...")
    
    model.eval()
    with torch.no_grad():
        try:
            # Sample two random latent vectors
            z1 = torch.randn(1, LATENT_DIM).to(device)
            z2 = torch.randn(1, LATENT_DIM).to(device)
            
            # Create interpolation
            interpolated_images = []
            for i in range(steps):
                alpha = i / (steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                generated = model.decode(z_interp)
                if generated.dim() == 2:  # Flattened output
                    image = generated.view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
                else:  # Already image format
                    image = generated
                
                interpolated_images.append(image)
            
            # Combine all images
            all_images = torch.cat(interpolated_images, dim=0)
            save_image(all_images, filename, normalize=True, nrow=steps)
            print(f"💾 Saved as: {filename}")
            
            return all_images
        except Exception as e:
            print(f"❌ Interpolation failed: {e}")
            return None

def reconstruct_test_image(model, device, output_filename="outputs/reconstruction_test.jpg"):
    """Reconstruct a test image if available"""
    print(f"\n🔄 Testing image reconstruction...")
    
    celeba_dir = './data/celeba/img_align_celeba/'
    
    if not os.path.exists(celeba_dir):
        print("📁 Dataset directory not found, skipping reconstruction test")
        return None
    
    import glob
    image_files = glob.glob(os.path.join(celeba_dir, '*.jpg'))
    if not image_files:
        print("📁 No test images found, skipping reconstruction test")
        return None
    
    # Use first image as test
    test_image_path = image_files[0]
    print(f"📷 Using test image: {os.path.basename(test_image_path)}")
    
    model.eval()
    try:
        # Load and preprocess image
        image = Image.open(test_image_path).convert('RGB')
        input_tensor = celeb_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Encode and decode
            mu, logvar = model.encode(input_tensor)
            z = model.reparameterize(mu, logvar)
            reconstructed = model.decode(z)
            
            if reconstructed.dim() == 2:  # Flattened output
                reconstructed = reconstructed.view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            
            # Save original and reconstruction side by side
            comparison = torch.cat([input_tensor, reconstructed], dim=0)
            save_image(comparison, output_filename, normalize=True, nrow=2)
            print(f"💾 Saved comparison as: {output_filename}")
            
            return comparison
    except Exception as e:
        print(f"❌ Reconstruction failed: {e}")
        return None

def main():
    """Main inference function"""
    print("🎯 Smart VAE Inference - Automatic Model Detection")
    print("=" * 60)
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Find best available model
    model_path, description = find_best_model()
    
    if model_path is None:
        print("❌ No trained models found!")
        print("\nPlease run one of these training scripts first:")
        print("  📚 train_quick.py - Quick training (5 epochs)")
        print("  🎓 train_from_scratch.py - Full training (20 epochs)")
        return
    
    # Load model
    model, device = load_model(model_path, description)
    if model is None:
        return
    
    print("\n" + "=" * 60)
    print("🎨 Starting inference tasks...")
    
    # Generate random images
    generate_random_images(model, device, num_samples=16, filename="outputs/generated_random.jpg")
    
    # Latent space interpolation
    interpolate_latent_space(model, device, steps=8, filename="outputs/latent_interpolation.jpg")
    
    # Reconstruct test image
    reconstruct_test_image(model, device, "outputs/reconstruction_test.jpg")
    
    print("\n" + "=" * 60)
    print("✅ Smart inference completed!")
    print("\n📁 Generated files:")
    print("  🎲 outputs/generated_random.jpg - Random generated faces")
    print("  🌈 outputs/latent_interpolation.jpg - Smooth transitions")
    print("  🔄 outputs/reconstruction_test.jpg - Original vs reconstructed")
    print(f"\n🏆 Used model: {description}")

if __name__ == "__main__":
    main()