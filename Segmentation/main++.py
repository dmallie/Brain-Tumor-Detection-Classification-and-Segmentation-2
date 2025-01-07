#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:44:31 2024

@author: dagi
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2 
import os
from loss_functions import  BCEDiceLoss
import torch.optim as optim
from UNet_plus_plus import UNetPlusPlus
from utils import get_loaders, calculate_mean_std
from custom_dataset import UNetDataset
from loops import train_and_validate
from timeit import default_timer as timer
from torchinfo import summary
import itertools
import matplotlib.pyplot as plt

# In[0] Settin the Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2**4
EPOCHS = 200
# NUM_WORKERS = 2 
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256
# PIN_MEMORY = True 
LOAD_MODEL = False 
ROOT_PATH = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Thesis Project/Segmentation/"
TRAIN_IMG_DIR = ROOT_PATH + "Train/Images/"
VAL_IMG_DIR = ROOT_PATH + "Val/Images/"
TRAIN_MASK_DIR = ROOT_PATH + "Train/Masks/"
VAL_MASK_DIR = ROOT_PATH + "Val/Masks/"

        
# In[2] Calculate mean and standard deviation of training and val dataset
train_mean, train_std = calculate_mean_std(TRAIN_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
val_mean, val_std = calculate_mean_std(VAL_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)

print(f"mean_train: {train_mean}\t std_train: {train_std}")
print(f"mean_val: {val_mean}\t std_val: {val_std}")

# %% Set transform functions 
train_transform = A.Compose([
                        # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                        # A.ToGray(p=1.0), # p=1.0 ensures that the grayscale transform is always applied
                        # A.Rotate(limit=35, p=1.0),
                        # A.HorizontalFlip(p=0.5),
                        # A.VerticalFlip(p=0.1),
                        # A.RandomBrightnessContrast(p=0.2),
                        # A.ElasticTransform(p=0.2),
                        A.Normalize(
                            mean = [train_mean],
                            std = [train_std],
                            max_pixel_value= 1.0
                            ),
                        ToTensorV2(),
                        ])

val_transform = A.Compose([
                        # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                        # A.ToGray(p=1.0), # p=1.0 ensures that the grayscale transform is always applied
                        A.Normalize(
                            mean = [val_mean],
                            std = [val_std],
                            max_pixel_value = 1.0),
                        ToTensorV2(),
                        ])
                    
# %% Create dataset instances
train_dataset = UNetDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform)
val_dataset = UNetDataset(VAL_IMG_DIR, VAL_MASK_DIR, val_transform)

# create dataloader

train_loader, val_loader = get_loaders(
                                train_dataset,
                                val_dataset,
                                BATCH_SIZE )

# %% initialize the model
model = UNetPlusPlus(in_channels = 1,  num_classes = 1).to(DEVICE)

# %% To inspect the model
# summary(model = model, input_size = (1, IMAGE_HEIGHT, IMAGE_WIDTH) )

# %%
loss_fn = BCEDiceLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = 1e-4)

# StepLR scheduler: decrease LR by a factor of 0.1 every 10 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                             mode = 'min',
                                             factor = 0.5,
                                             patience = 40)
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Adjust the size as needed
start_time = timer()

BCEDiceLoss_list, dice_list, iou_list = train_and_validate(
                                                model=model,
                                                train_loader=train_loader,
                                                val_loader=val_loader,
                                                optimizer=optimizer,
                                                criterion=loss_fn,
                                                num_epochs=EPOCHS,
                                                scheduler= scheduler,
                                                save_path="unet++.pth"
                                        )   

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds or {(end_time - start_time )/60:.2f} minutes")

# In[] Step 8: Plot the performance of the model
x = list(itertools.chain(range(0, len(BCEDiceLoss_list))))
plt.plot(x, BCEDiceLoss_list, label = "BCE-DiceLoss")
plt.plot(x, dice_list, label = "Dice Loss")
plt.plot(x, iou_list, label = "IoU Loss")
plt.legend()
plt.show()

