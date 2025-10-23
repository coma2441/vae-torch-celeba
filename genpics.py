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

#%%
save_image(pics, 'rndpics_new_samples.jpg', nrow=8)


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
