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

def save_sample_images(model, device, epoch, num_samples=8):
    """Generate and save sample images"""
    model.eval()
    with torch.no_grad():
        # Generate from random latent vectors
        sample = torch.randn(num_samples, LATENT_DIM).to(device)
        sample = model.decode(sample)
        sample = sample.view(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE)
        save_image(sample, f'samples_epoch_{epoch}.png', normalize=True, nrow=4)

def main():
    """Main training function"""
    # Configuration
    EPOCHS = 20
    BATCH_SIZE = 64
    CELEBA_DIR = './data/celeba/img_align_celeba/'
    LEARNING_RATE = 1e-3
    MODEL_SAVE_PATH = 'models/vae_trained_from_scratch.pth'
    
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
    
    print(f"🎯 Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"📈 Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
    
    # Training loop
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, device, train_loader, optimizer, epoch)
        
        # Test
        test_loss = test_model(model, device, test_loader)
        
        # Save sample images
        if epoch % 5 == 0:
            save_sample_images(model, device, epoch)
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Saved best model with test loss: {best_loss:.4f}")
        
        epoch_time = time.time() - epoch_start
        print(f"⏱️ Epoch {epoch} completed in {epoch_time:.1f}s")
        print("-" * 60)
    
    total_time = time.time() - start_time
    print(f"🎉 Training completed in {total_time/60:.1f} minutes!")
    print(f"💾 Best model saved as: {MODEL_SAVE_PATH}")
    print(f"🏆 Best test loss: {best_loss:.4f}")
    
    # Final sample generation
    print("🎨 Generating final samples...")
    save_sample_images(model, device, 'final', num_samples=16)
    
    return MODEL_SAVE_PATH

if __name__ == "__main__":
    model_path = main()
    print(f"\n✅ Training complete! Model saved at: {model_path}")