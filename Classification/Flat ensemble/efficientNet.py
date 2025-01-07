#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:16:26 2024
Objective:
    - Train EfficientNet-V2-S to differentiate MRI with tumor with that of no-tumor
@author: dagi
"""
import os 
import sys 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
import torch
from torch import nn 
from torchinfo import summary
from torchvision import models 
from timeit import default_timer as timer
from loops import main_loop
import matplotlib.pyplot as plt
import itertools
import torch.optim as optim 
from utils import calculate_mean_std

# In[] Function/method counts the total number of files in the provided directory
def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)  # Add the number of files in each directory
    return file_count

# In[] Set route path for the data
rootPath = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Classification/Flat Ensemble/"
trainingPath = rootPath + "Train/"
valPath = rootPath + "Val/"

no_training_data = count_files(trainingPath)
no_val_data = count_files(valPath) 

# In[] Set Hyperparameter
WIDTH = 256
HEIGHT = 256
BATCH = 2**6
OUTPUT_SHAPE = 4 #  0: Glioma, 1: Meningioma, 2: No_tumor and 3: Pituitary
EPOCH = 100
LEARNING_RATE = 1e-4
# In[] Calculate the mean and standard deviation for each dataset
mean_train, std_train = calculate_mean_std(trainingPath)
mean_val, std_val = calculate_mean_std(valPath)

# In[] Set transform function
transform_train = transforms.Compose([
                        # Resize the image to fit the model
                        transforms.Resize(size=(HEIGHT, WIDTH)),  
                        transforms.Grayscale(num_output_channels=1),
                        transforms.RandomHorizontalFlip(p = 0.5),# Randomly flip some images horizontally with probability of 50%
                        transforms.RandomVerticalFlip(p=0.2),
                        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        # Normalize the 3-chanelled image
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[mean_train], 
                                             std = [std_train]
                                             ),
                        # Convert image to tensor object
                ])
transform_val = transforms.Compose([
                        # Resize the image to fit the model
                        transforms.Resize(size=(HEIGHT, WIDTH)),
                        # Convert image to grayscale 
                        transforms.Grayscale(num_output_channels=1),
                        # Normalize the 3-channeled tensor object
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[mean_val], std = [std_val]),
                        # Convert image to tensor object
                ])

# In[] Setup the dataset
train_dataset = datasets.ImageFolder(
                    root = trainingPath,
                    transform = transform_train)
val_dataset = datasets.ImageFolder(
                    root = valPath,
                    transform = transform_val)

# In[] Setup the DataLoader
train_dataloader = DataLoader(
                        dataset = train_dataset,
                        batch_size = BATCH,
                        num_workers = 4,
                        shuffle = True,
                        pin_memory = True)
val_dataloader = DataLoader(
                        dataset = val_dataset,
                        batch_size = BATCH,
                        num_workers = 4,
                        shuffle = False,
                        pin_memory = True)
# In[] Step 5: import and instantiate ResNet50
weights = models.EfficientNet_V2_S_Weights.DEFAULT
model = models.efficientnet_v2_s(weights = weights)

# In[] Modify the first layer to receive grayscale images
# Get the pretrained model's first layer
input_layer = model.features[0][0]

# # Create a new Conv2d layer with 1 input channel instead of 3
model.features[0][0] = nn.Conv2d(
                        in_channels=1,  # Set to 1 for grayscale images
                        out_channels=input_layer.out_channels,
                        kernel_size=input_layer.kernel_size,
                        stride=input_layer.stride,
                        padding=input_layer.padding,
                        bias=(input_layer.bias is not None),
                    )

# In[10] Modifying the model to meet input and output criteria
# 1. Freezing the trainablility of base model
for params in model.parameters():
    params.requires_grad = False

# Set the trainablility of block Sequential 6 to True 
unfreeze = False
for name, layer in model.features[6].named_children():
    if name == "0":
        unfreeze = True  # Start unfreezing from here

    if unfreeze:
        for params in layer.parameters():
            params.requires_grad = True

# Set the trainability of Conv2dNormActivation block to True
unfreeze = False
for name, layer in model.features[7].named_children():
    if name == "0":
        unfreeze = True  # Start unfreezing from here

    if unfreeze:
        for params in layer.parameters():
            params.requires_grad = True


# modify the output shape and connection layer of the model
model.classifier[1] = nn.Linear( 
                        in_features = 1280,
                        out_features = OUTPUT_SHAPE,
                        bias = True)

# In[9] Model Info after configuration
summary(model = model,
        input_size = (BATCH, 1, HEIGHT, WIDTH),
        col_names = ["input_size", "output_size", "trainable"],
        col_width = 20,
        row_settings = ["var_names"])


# In[] Step 6: Setup the loss function and Optimizer function & class imbalance handling
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode = 'min',
                                                 factor = 0.1,
                                                 patience = 10)
loss_fn = nn.CrossEntropyLoss()
# In[] Step 7: Start the training loop
start_time = timer()
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Setup training and save the results
accuracy_list_training, accuracy_list_val, loss_list_training, loss_list_val = main_loop( model,
                                         train_dataloader,
                                         val_dataloader,
                                         optimizer,
                                         criterion = loss_fn,
                                         epochs = EPOCH,
                                         scheduler = scheduler,
                                         save_path = "EfficientNet.pth")

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# In[] Step 8: Plot the performance of the model
x = list(itertools.chain(range(0, len(accuracy_list_training))))
plt.title(label="Flat Ensemble: EfficientNet")
plt.plot(x, accuracy_list_training, label = "Training Performance")
plt.plot(x, accuracy_list_val, label = "Validation Performance")
plt.legend()
plt.show()

# In[] Step 9: Plot the loss function of the model
# train_loss = [100 - acc for acc in train_accuracy]
# val_loss = [100 - acc for acc in val_accuracy]
plt.title(label="Flat Ensemble: EfficientNet")
plt.plot(x, loss_list_training, label = "Training Loss")
plt.plot(x, loss_list_val, label = "Validation Loss")
plt.legend()
plt.show()

