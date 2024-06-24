import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, data_root_dir, img_H, img_W, is_paired=True, is_test=False, is_sorted=True):
        self.data_root_dir = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.is_paired = is_paired
        self.is_test = is_test
        self.is_sorted = is_sorted
        self.transform = transforms.Compose([
            transforms.Resize((img_H, img_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.image_files = sorted(os.listdir(os.path.join(data_root_dir, 'images')))
        self.cloth_files = sorted(os.listdir(os.path.join(data_root_dir, 'clothes')))
        
        if not is_sorted:
            random.shuffle(self.image_files)
            random.shuffle(self.cloth_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root_dir, 'images', self.image_files[idx])
        cloth_path = os.path.join(self.data_root_dir, 'clothes', self.cloth_files[idx])
        
        image = Image.open(img_path).convert("RGB")
        cloth = Image.open(cloth_path).convert("RGB")
        
        image = self.transform(image)
        cloth = self.transform(cloth)
        
        sample = {
            'image': image,
            'cloth': cloth,
            'img_fn': self.image_files[idx],
            'cloth_fn': self.cloth_files[idx]
        }

        return sample
