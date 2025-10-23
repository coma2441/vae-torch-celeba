#!/usr/bin/env python3
"""
Quick test script to verify everything is working
This runs a mini training session with just a few batches
"""

import os
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# project modules
from utils import print, rndstr
from vae import VAE, IMAGE_SIZE, LATENT_DIM, CELEB_PATH, image_dim, celeb_transform
from custom_dataset import SimpleCelebADataset

def test_training():
    print("=" * 50)
    print("QUICK TRAINING TEST")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Load small subset of data
    image_dir = os.path.join(CELEB_PATH, 'img_align_celeba')
    train_dataset = SimpleCelebADataset(image_dir, transform=celeb_transform, split='train')
    
    # Use small batch and subset for quick test
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    
    # Model and optimizer
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Loss function
    def loss_function(recon_x, x, mu, log_var):
        MSE = F.mse_loss(recon_x, x.view(-1, image_dim))
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kld_weight = 0.00025
        loss = MSE + kld_weight * KLD  
        return loss
    
    # Quick training test (just 3 batches)
    model.train()
    print("Testing training loop...")
    
    for batch_idx, (data, _) in enumerate(train_loader):
        if batch_idx >= 3:  # Only test 3 batches
            break
            
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        log_var = torch.clamp_(log_var, -10, 10)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        optimizer.step()
        
        print(f'Batch {batch_idx + 1}: Loss = {loss.item():.6f}')
    
    # Test image generation
    model.eval()
    with torch.no_grad():
        print("Testing image generation...")
        sample = torch.randn(8, LATENT_DIM).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(8, 3, IMAGE_SIZE, IMAGE_SIZE), 'test_generated.png', nrow=4)
        print("Generated test images saved as 'test_generated.png'")
    
    print("=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("✅ Your setup is working correctly!")
    print("✅ You can now run 'python trainvae.py' for full training")
    print("✅ Or 'python genpics.py' to generate images with the pre-trained model")
    print("=" * 50)

if __name__ == "__main__":
    test_training()