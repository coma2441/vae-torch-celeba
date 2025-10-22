#!/usr/bin/env python3
"""
Simple VAE Training Script - Train from scratch and save model
"""
import os
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from torchvision import transforms
import time

# Import the original VAE
from vae import VAE, IMAGE_SIZE, LATENT_DIM, celeb_transform
from utils import print, rndstr

class CelebADataset(Dataset):
    """CelebA Dataset for VAE training"""
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = glob.glob(os.path.join(root_dir, '*.jpg'))
        if limit:
            self.image_files = self.image_files[:limit]
        print(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def vae_loss(recon_x, x, mu, logvar):
    """VAE loss function"""
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x.view(-1, IMAGE_SIZE * IMAGE_SIZE * 3), reduction='sum')
    
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + 0.00025 * kld_loss, recon_loss, kld_loss

def train_epoch(model, device, train_loader, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    recon_losses = 0
    kld_losses = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kld_loss = vae_loss(recon_batch, data, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        recon_losses += recon_loss.item()
        kld_losses += kld_loss.item()
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item()/len(data):.6f}')
    
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_losses / len(train_loader.dataset)
    avg_kld = kld_losses / len(train_loader.dataset)
    
    print(f'====> Epoch {epoch}: Avg loss: {avg_loss:.4f}, '
          f'Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}')
    
    return avg_loss

def test_model(model, device, test_loader):
    """Test the model"""
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, _, _ = vae_loss(recon_batch, data, mu, logvar)
            test_loss += loss.item()
    
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return test_loss

def save_sample_images(model, device, epoch, num_samples=16):
    """Generate and save sample images with comprehensive visualization"""
    model.eval()
    
    # Create visualization directory
    os.makedirs('outputs/training_progress', exist_ok=True)
    
    with torch.no_grad():
        # 1. Generate random samples
        sample = torch.randn(num_samples, LATENT_DIM).to(device)
        generated = model.decode(sample)
        generated = generated.view(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE)
        
        # Save random generation grid with proper epoch formatting
        if isinstance(epoch, str):
            epoch_str = epoch
        else:
            epoch_str = f"{epoch:03d}"
        
        save_image(generated, f'outputs/training_progress/epoch_{epoch_str}_random_generation.jpg', 
                  normalize=True, nrow=4, padding=2)
        
        # 2. Create latent space interpolation
        z1 = torch.randn(1, LATENT_DIM).to(device)
        z2 = torch.randn(1, LATENT_DIM).to(device)
        
        interpolated_images = []
        for i in range(8):
            alpha = i / 7.0
            z_interp = (1 - alpha) * z1 + alpha * z2
            interp_img = model.decode(z_interp)
            interp_img = interp_img.view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            interpolated_images.append(interp_img)
        
        interpolation_grid = torch.cat(interpolated_images, dim=0)
        save_image(interpolation_grid, f'outputs/training_progress/epoch_{epoch_str}_interpolation.jpg',
                  normalize=True, nrow=8, padding=2)
        
        # 3. Test reconstruction on a few real images (if available)
        celeba_dir = './data/celeba/img_align_celeba/'
        if os.path.exists(celeba_dir):
            import glob
            from torchvision import transforms
            
            image_files = glob.glob(os.path.join(celeba_dir, '*.jpg'))[:4]  # Use first 4 images
            if image_files:
                transform = transforms.Compose([
                    transforms.Resize(IMAGE_SIZE, antialias=True),
                    transforms.CenterCrop(IMAGE_SIZE),
                    transforms.ToTensor()
                ])
                
                reconstructions = []
                originals = []
                
                for img_path in image_files:
                    # Load and preprocess
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = transform(image).unsqueeze(0).to(device)
                    
                    # Reconstruct
                    mu, logvar = model.encode(input_tensor)
                    z = model.reparameterize(mu, logvar)
                    reconstructed = model.decode(z)
                    reconstructed = reconstructed.view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
                    
                    originals.append(input_tensor)
                    reconstructions.append(reconstructed)
                
                # Create comparison grid (original on top, reconstruction below)
                orig_grid = torch.cat(originals, dim=0)
                recon_grid = torch.cat(reconstructions, dim=0)
                comparison = torch.cat([orig_grid, recon_grid], dim=0)
                
                save_image(comparison, f'outputs/training_progress/epoch_{epoch_str}_reconstruction.jpg',
                          normalize=True, nrow=4, padding=2)
    
    print(f"📸 Epoch {epoch} visualizations saved to outputs/training_progress/")

def main():
    """Main training function"""
    # Configuration
    EPOCHS = 800  # Extended from 200 to 800 epochs
    BATCH_SIZE = 64
    CELEBA_DIR = './data/celeba/img_align_celeba/'
    LEARNING_RATE = 1e-3
    MODEL_SAVE_PATH = 'models/vae_trained_from_scratch.pth'
    CHECKPOINT_PATH = 'models/training_checkpoint.pth'
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Device setup with priority order
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    # Check if data exists
    if not os.path.exists(CELEBA_DIR):
        print(f"❌ Data directory not found: {CELEBA_DIR}")
        print("Please make sure the CelebA dataset is properly extracted")
        return
    
    # Create datasets
    print("📁 Loading dataset...")
    full_dataset = CelebADataset(CELEBA_DIR, transform=celeb_transform, limit=10000)  # Limit for faster training
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create model
    print("🏗️ Creating model...")
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Check for existing checkpoint or model
    start_epoch = 1
    best_loss = float('inf')
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📥 Found checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"🔄 Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
    elif os.path.exists(MODEL_SAVE_PATH):
        print(f"📥 Found existing model: {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"🔄 Loaded existing model, starting from epoch 21")
        start_epoch = 21  # Continue from where 20-epoch training left off
    else:
        print("🆕 Starting fresh training from epoch 1")
    
    print(f"🎯 Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"📈 Training epochs {start_epoch}-{EPOCHS} with batch size {BATCH_SIZE}")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, device, train_loader, optimizer, epoch)
        
        # Test
        test_loss = test_model(model, device, test_loader)
        
        # Create visualizations every 20 epochs for longer training, or at key milestones
        if epoch == start_epoch or epoch % 20 == 0 or epoch in [30, 50, 100, 150, 250, 300, 350]:
            print(f"🎨 Creating visualization checkpoint for epoch {epoch}...")
            save_sample_images(model, device, epoch)
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Saved best model with test loss: {best_loss:.4f}")
        
        # Save checkpoint every 10 epochs for resuming
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'train_loss': train_loss,
                'test_loss': test_loss
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"📋 Checkpoint saved for epoch {epoch}")
        
        epoch_time = time.time() - epoch_start
        print(f"⏱️ Epoch {epoch} completed in {epoch_time:.1f}s")
        print("-" * 60)
    
    total_time = time.time() - start_time
    print(f"🎉 Training completed in {total_time/60:.1f} minutes!")
    print(f"💾 Best model saved as: {MODEL_SAVE_PATH}")
    print(f"🏆 Best test loss: {best_loss:.4f}")
    
    # Final comprehensive visualization
    print("🎨 Generating final comprehensive visualization...")
    save_sample_images(model, device, 'final', num_samples=16)
    
    # Clean up checkpoint file since training is complete
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("🧹 Removed checkpoint file (training complete)")
    
    # Print visualization summary
    print("\n📸 Training visualizations created:")
    print("  📁 outputs/training_progress/")
    print("     ├── epoch_021_* (continuation from 20)")
    if EPOCHS >= 50:
        print("     ├── epoch_050_* (50-epoch checkpoint)")
    if EPOCHS >= 100:
        print("     ├── epoch_100_* (100-epoch checkpoint)")
    if EPOCHS >= 200:
        print("     ├── epoch_200_* (200-epoch checkpoint)")
    print("     └── epoch_final_* (final results)")
    print("\n  Each checkpoint includes:")
    print("     • *_random_generation.jpg - 16 random faces")
    print("     • *_interpolation.jpg - smooth transitions")
    print("     • *_reconstruction.jpg - original vs reconstructed")
    
    return MODEL_SAVE_PATH

if __name__ == "__main__":
    model_path = main()
    print(f"\n✅ Training complete! Model saved at: {model_path}")