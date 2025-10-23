# %%
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from vae import IMAGE_SIZE, celeb_transform, CELEB_PATH
from custom_dataset import SimpleCelebADataset
import os

MODEL_FILE = 'vae_model_20.pth'

image_dir = os.path.join(CELEB_PATH, 'img_align_celeba')
dataset = SimpleCelebADataset(image_dir, transform=celeb_transform, split='all')
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
model = torch.load(MODEL_FILE, map_location='cpu', weights_only=False)

for pic, _ in loader:  # batch size is 1, loader is shuffled, so this gets one random pic
    pics = pic.to('cpu')
    break
orig = torch.clone(pics)

# use code below if you want to manually tweak the latent vector
mu, log_var = model.encode(orig)


# %%
# Generate all noise at once
w = 1
std = torch.exp(w * log_var)
epss = torch.randn(7, *std.shape)  # Shape: [7, 1, 128]
epss = epss.squeeze(1)  # Shape: [7, 128]

# Generate all z vectors at once
zs = epss * std + mu  # Broadcasting: [7, 128] * [1, 128] + [1, 128] = [7, 128]
# %% 
# Generate images
pics = torch.clone(orig)
for i in range(7):
    z = zs[i:i+1]  # Keep batch dimension: [1, 128]
    recon = model.decode(z)
    pic = recon[0].view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    pics = torch.cat((pics, pic), dim=0)

# %%
save_image(pics, 'rndpics_new_samples.jpg', nrow=8)

# %%
# Save each image individually for inspection
print("Saving individual images...")
for i in range(pics.shape[0]):
    if i == 0:
        filename = f'pic_{i:02d}_original.jpg'
        print(f"Image {i}: Original")
    else:
        filename = f'pic_{i:02d}_generated.jpg'
        print(f"Image {i}: Generated variation {i}")
    
    save_image(pics[i], filename)

print(f"Saved {pics.shape[0]} individual images")

# %%
# Analyze differences between pics[1] and pics[2]
print("\n" + "="*50)
print("ANALYZING DIFFERENCES BETWEEN PICS[1] AND PICS[2]")
print("="*50)

img1 = pics[1]  # First generated image
img2 = pics[2]  # Second generated image

# 1. Pixel-wise differences
pixel_diff = torch.abs(img1 - img2)
mse = torch.mean((img1 - img2)**2).item()
mae = torch.mean(pixel_diff).item()
max_diff = torch.max(pixel_diff).item()

print(f"Mean Squared Error (MSE): {mse:.8f}")
print(f"Mean Absolute Error (MAE): {mae:.8f}")
print(f"Maximum pixel difference: {max_diff:.8f}")

# 2. Are they identical?
are_identical = torch.allclose(img1, img2, atol=1e-6)
print(f"Images are identical (within 1e-6): {are_identical}")

# 3. Latent space differences (pics[1] = zs[0], pics[2] = zs[1])
z1 = zs[0]  # Latent vector for pics[1]
z2 = zs[1]  # Latent vector for pics[2]
latent_distance = torch.norm(z1 - z2).item()
print(f"Latent space L2 distance: {latent_distance:.6f}")

# 4. Show some pixel statistics
print(f"\nPics[1] - min: {img1.min():.6f}, max: {img1.max():.6f}, mean: {img1.mean():.6f}")
print(f"Pics[2] - min: {img2.min():.6f}, max: {img2.max():.6f}, mean: {img2.mean():.6f}")

# 5. Save difference map
if pixel_diff.max() > 0:
    diff_map = pixel_diff / pixel_diff.max()
else:
    diff_map = pixel_diff
save_image(diff_map, 'difference_map_pics1_vs_pics2.jpg')
print(f"\nSaved difference map to 'difference_map_pics1_vs_pics2.jpg'")
print("(Bright areas = larger differences, dark areas = similar pixels)")


# %%
# Build affinity matrix using mu + 7 samples in zs
# Stack mu as first row to get 8x128 matrix
Z = torch.cat([mu, zs], dim=0)  # Shape: [8, 128]

# Compute pairwise Euclidean distances and convert to affinities
distances = torch.cdist(Z, Z, p=2)  # Shape: [8, 8]
sigma = distances.mean().item() if distances.mean().item() > 0 else 1.0
affinity_matrix = torch.exp(-distances**2 / (2 * (sigma ** 2)))

print("Affinity Matrix (8x8) - rows/cols: [mu, z1..z7]:")
print(affinity_matrix)



# for _ in range(7):
#     recon, _, _ = model(pics)
#     pic = recon[0].view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
#     pics = torch.cat((pics, pic), dim=0)

# save_image(pics, 'rndpics.jpg', nrow=8)

# %%
