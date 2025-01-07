import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class UNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)
        self.mask_names = os.listdir(mask_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        if image_path.endswith(".jpg"):
            mask_path = image_path.replace(".jpg", ".png")
        else:
            mask_path = image_path.replace(".tif", "_mask.tif")
        
        mask_path = mask_path.replace("Images", "Masks")
        # print(f"\nimage_path: {image_path}\n mask_path: {mask_path}")
        # print(f"image_path: {image_path}\n mask_path: {mask_path}\n")
        # mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale for segmentation
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to numpy and normalize
        image = np.array(image) / 255.0  # Normalize image between 0 and 1
        mask = np.array(mask) / 255.0  # Normalize mask (assuming binary mask)
        
        # Convert mask to integer type
        # mask = mask.astype(np.int64)
        
        # Apply transformations (if provided)
        if self.transform:
            image_transformed = self.transform(image=image)["image"]
            mask = torch.from_numpy(mask)  # Add channel dimension to mask
        img = image_transformed.clone().detach()

        mask_ = mask.clone().detach().unsqueeze(0)
        # print(f"img.shape: {img.shape}\n mask_.shape: {mask_.shape}\n")
        return img, mask_

