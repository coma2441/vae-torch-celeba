import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class SimpleCelebADataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Simple CelebA dataset loader for when you just have the images
        
        Args:
            root_dir: Path to the directory containing the images
            transform: Optional transform to be applied on a sample
            split: 'train', 'valid', or 'test' (we'll use simple splits)
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all jpg files
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.jpg')]
        self.image_files.sort()  # Ensure consistent ordering
        
        # Simple train/valid/test split (80/10/10)
        total_files = len(self.image_files)
        if split == 'train':
            self.image_files = self.image_files[:int(0.8 * total_files)]
        elif split == 'valid':
            self.image_files = self.image_files[int(0.8 * total_files):int(0.9 * total_files)]
        elif split == 'test':
            self.image_files = self.image_files[int(0.9 * total_files):]
        else:  # 'all'
            pass  # use all files
            
        print(f"Split '{split}': {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Return image and dummy label (CelebA loader returns (image, target))
        return image, 0