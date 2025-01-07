#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 07:38:59 2024
Objective:
    - Using the MRIs in the test set and the three segmentation models
    this script will segment the tumor
    - we load the models: unet model, unet_plus_plus, and attention_unet
    - each mri scan in the test dataset transformed for the models to utilise them
    - Mask file will be constructed from the mean of the probability outputs on each
    pixels from the three models
    @author: dagi
"""
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from Architecture_unet import UNet
from Architectrue_unet_plus import UNetPlusPlus
from Architecture_attention_unet import AttentionUNet
from Architecture_unet_64 import UNet_64
from tqdm import tqdm
import numpy as np 
from utils import ( calculate_mean_std, precision_score_segmentation, dice_loss, recall_segmentation,
                   intersection_over_union, f1_score_segmentation )
from timeit import default_timer as timer
from scipy.ndimage import label
import cv2

# In[] Routing path to the Source directory
src_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Segmentation/Test/Images/"
mask_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Segmentation/Test/Masks/"
src_list = os.listdir(src_path)

# In[] Destination folder
dest_ensembled = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Segmentation/Test/Ensembled/"

# In[] Setting Hyperparameters
WIDTH = 256 
HEIGHT = 256 
WIDTH_64 = 64
HEIGHT_64 = 64
OUTPUT_SHAPE = 1
BATCH  = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %% Calculate the mean and standard deviation of the dataset
mean, std = calculate_mean_std(src_path, HEIGHT, WIDTH)
print(f"mean of test dataset: {mean}\t std of test dataset: {std}\n")

# In[] Instantiation of list object to store the values of mask comparison
# Dice Coffecient
dice_unet = []
dice_unet_plus = []
dice_unet_attention = []
dice_ensembled = []
dice_merged = []
dice_cascaded = []
# Intersection Over Union
iou_unet = []
iou_unet_plus = []
iou_unet_attention = []
iou_ensembled = []
iou_merged = []
iou_cascaded = []
# Precision metrics
precision_unet = []
precision_unet_plus = []
precision_unet_attention = []
precision_ensembled = []
precision_merged = []
precision_cascaded = []
# Precision metrics
recall_unet = []
recall_unet_plus = []
recall_unet_attention = []
recall_ensembled = []
recall_merged = []
recall_cascaded = []
# F1-Score 
f1_unet = []
f1_unet_plus = []
f1_unet_attention = []
f1_ensembled = []
f1_merged = []
f1_cascaded = []
# Number of files Tumors detected in the dataset
detected_unet = 0
detected_unet_plus = 0
detected_unet_attention = 0
detected_merged = 0 
detected_ensembled = 0 
detected_cascaded = 0

# In[] Set Transform Functions
transform_fn = A.Compose([
                        A.Resize(height=HEIGHT, width=WIDTH),
                        # A.ToGray(p=1.0), # p=1.0 ensures that the grayscale transform is always applied
                        A.Normalize(
                            mean = [mean],
                            std = [std],
                            max_pixel_value = 1.0),
                        ToTensorV2(),
                        ])

transform_fn_2 = A.Compose([
                        A.Resize(height=HEIGHT_64, width=WIDTH_64),
                        # A.ToGray(p=1.0), # p=1.0 ensures that the grayscale transform is always applied
                        A.Normalize(
                            mean = [mean],
                            std = [std],
                            max_pixel_value = 1.0),
                        ToTensorV2(),
                        ])

# In[] Load the unet model 
model_path_1 = "unet.pth"
model_unet = UNet(in_channels= 1, num_classes=OUTPUT_SHAPE)

# %% load the saved dict
saved_state_dict = torch.load(model_path_1, weights_only=True)
# load teh state_dict into the model
model_unet.load_state_dict(saved_state_dict)

# In[] Load the unet++ model 
model_path_2 = "unet++.pth"
model_unet_plus_plus = UNetPlusPlus(in_channels= 1, num_classes=OUTPUT_SHAPE)

# %% load the saved dict
saved_state_dict = torch.load(model_path_2, weights_only=True)
# load teh state_dict into the model
model_unet_plus_plus.load_state_dict(saved_state_dict)

# In[] Load the unet_attention model 
model_path_3 = "unet_attention.pth"
model_attention = AttentionUNet(in_channels= 1, num_classes=OUTPUT_SHAPE)

# %% load the saved dict
saved_state_dict = torch.load(model_path_3, weights_only=True)
# load teh state_dict into the model
model_attention.load_state_dict(saved_state_dict)

# In[] Load the unet_attention model 
model_path_64 = "unet_64.pth"
model_64 = UNet_64(in_channels= 1, num_classes=OUTPUT_SHAPE)

# %% load the saved dict
saved_state_dict = torch.load(model_path_64, weights_only=True)
# load teh state_dict into the model
model_64.load_state_dict(saved_state_dict)

# %% move the models to cuda device
model_unet = model_unet.to(DEVICE)
model_unet_plus_plus = model_unet_plus_plus.to(DEVICE)
model_attention = model_attention.to(DEVICE)
model_64 = model_64.to(DEVICE)

# %% set the models to evaluation mode
model_unet.eval()
model_unet_plus_plus.eval()
model_attention.eval()
model_64.eval()

# %% iterate through each image file and perform segmentation
start_time = timer()
for img_name in tqdm(src_list, desc="Segmentation using ensemble techniques: ", leave=False):
    # set the full path of the image
    full_path = src_path + img_name
    
    # set the destination path
    if img_name.endswith(".jpg"):
        dest_path_ensembled = dest_ensembled + img_name.replace(".jpg", ".png")
        full_path_mask = mask_path + img_name.replace(".jpg", ".png")
    else:
        dest_path_ensembled = dest_ensembled + img_name.replace(".tif", "_mask.tif")
        full_path_mask = mask_path + img_name.replace(".tif", "_mask.tif")
    ##########################################################
    # load the image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(full_path_mask, cv2.IMREAD_GRAYSCALE)
    # convet img to numpy array and standardize the values
    img_standardized = np.array(img) / 255.0
    # Transform the image
    img_transformed = transform_fn(image = img_standardized)["image"].unsqueeze(0).to(DEVICE) # add batch dimension
    
    ##############################################################
    # Perform the evaluation task
    with torch.no_grad():
        # Forward pass
        unet_output = model_unet(img_transformed)
        prediction_unet = torch.sigmoid(unet_output)
        # convert probabilites to 0 or 1
        binary_unet = (prediction_unet > 0.5).float()
        # Forward pass
        unet_plus_output = model_unet_plus_plus(img_transformed)
        prediction_unet_plus_plus = torch.sigmoid(unet_plus_output)
        # convert probabilites to 0 or 1
        binary_unet_plus_plus = (prediction_unet_plus_plus > 0.5).float()
        # Forward pass
        attention_output = model_attention(img_transformed)
        prediction_unet_attention = torch.sigmoid(attention_output)
        # convert probabilites to 0 or 1
        binary_unet_attention = (prediction_unet_attention > 0.5).float()
    # ensemble learning
    # ensemble_probabilities = (prediction_unet + prediction_unet_plus_plus + prediction_unet_attention)/3
    ensemble_probabilities = 0.338 *  prediction_unet + 0.321 * prediction_unet_plus_plus + 0.341* prediction_unet_attention
    # ensemble_probabilities = 0 *  prediction_unet + 0 * prediction_unet_plus_plus + 0 * prediction_unet_attention
    # prediction_ensemble = torch.sigmoid(ensemble_output)
    # convert probabilites to 0 or 1
    binary_ensemble = (ensemble_probabilities > 0.5).float()
    ##############################################################
    # move the binary predictions to cpu and numpy
    mask_unet = binary_unet.squeeze(0).squeeze(0).cpu().detach().numpy()
    mask_unet_plus_plus = binary_unet_plus_plus.squeeze(0).squeeze(0).cpu().detach().numpy()
    mask_unet_attention = binary_unet_attention.squeeze(0).squeeze(0).cpu().detach().numpy()
    mask_ensemble = binary_ensemble.squeeze(0).squeeze(0).cpu().detach().numpy()
    # convert mask to uint8 and values between 0 and 255
    mask_unet = (mask_unet * 255).astype(np.uint8)
    mask_unet_plus_plus = (mask_unet_plus_plus *  255).astype(np.uint8)
    mask_unet_attention = (mask_unet_attention * 255).astype(np.uint8)
    mask_ensemble = (mask_ensemble *  255).astype(np.uint8)
    mask_merge = np.clip(mask_unet + mask_unet_plus_plus + mask_unet_attention + mask_ensemble, 0, 255)
    # mask_ensemble = np.clip(mask_ensemble, 0, 255).astype(np.uint8)
    ####### REFINE THE OUTPUT #############################################
    # If ensemble didn't detect any tumor then copy the one that detects any
    if not np.any(mask_ensemble): # True means one is detected
        mask_fusion = mask_unet + mask_unet_plus_plus + mask_unet_attention
        mask_ensemble = np.clip(mask_fusion, 0, 255) # trim values >= 255 to 255 and <= 0 to 0
    # Check whether there are two or more disconnected balls in the ensemble png
    # Perform connected components labeling using scipy
    labeled_image, num_balls = label(mask_ensemble)
    merged_image, merged_balls = label(mask_merge)
    if num_balls > 1:
        # get the centroid coordinate of the balls from ensemble and all others
        sizes = []
        centroid = []
        for i in range(1, num_balls + 1):
            coords = np.argwhere(labeled_image == i)
            centroid = coords.mean(axis=0)  # Calculate centroid
            size = (labeled_image == i).sum()
            sizes.append(size)
        # find the largest component
        largest_component = np.argmax(sizes) + 1
        # redraw the ensemble mask with only the largest component
        mask_ensemble = (255*(labeled_image == largest_component)).astype(np.uint8)
    if merged_balls > 1:
        # get the centroid coordinate of the balls from ensemble and all others
        sizes = []
        centroid = []
        for i in range(1, num_balls + 1):
            coords = np.argwhere(labeled_image == i)
            centroid = coords.mean(axis=0)  # Calculate centroid
            size = (labeled_image == i).sum()
            sizes.append(size)
        # find the largest component
        largest_component = np.argmax(sizes) + 1
        # redraw the ensemble mask with only the largest component
        mask_merge = (255*(merged_image == largest_component)).astype(np.uint8)
  
        # print(f"Two balls detected in {img_name}")
    # For finer segmentation we call unet 64
    ##########################################################
    ###########################################################
    # find the center coordinate of the detected tumor
    if np.any(mask_ensemble):
        # find the center of the tumor
        tumor, num_balls = label(mask_ensemble)
        coords = np.argwhere(tumor == 1)
        center_point = coords.mean(axis=0)
        top_left = (center_point - 32).astype(np.uint8)
        bottom_right = (center_point + 32).astype(np.uint8)
        # crop the region 64 by 64
        cropped_img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        # transform the cropped image
        cropped_standardized = cropped_img/255
        cropped_transformed = transform_fn_2(image = cropped_standardized)["image"].unsqueeze(0).to(DEVICE) 
        # Forward pass
        unet_64 = model_64(cropped_transformed)
        prediction_64 = torch.sigmoid(unet_64)
        # convert probabilites to 0 or 1
        binary_64 = (prediction_64 > 0.5).float()
        # convert torch.tensor to numpy array
        binary_64 = binary_64.squeeze(0).squeeze(0).cpu().detach().numpy()

        # create teh mask file
        mask_64 = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        # merge mask_64 with binary_64
        mask_64[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = binary_64 
        mask_64 = (255*mask_64).astype(np.uint8)

    # save the mask file
    cv2.imwrite(dest_path_ensembled, mask_ensemble)
    ##################################################################
    # Compare mask_ensemble vs mask vs mask_unet vs mask unet_plus_plus vs mask_attention
    # lets calculate the dice coefficient
    dice_unet.append(1 - dice_loss(mask, mask_unet))
    dice_unet_plus.append(1 - dice_loss(mask, mask_unet_plus_plus))
    dice_unet_attention.append(1 - dice_loss(mask, mask_unet_attention))
    dice_ensembled.append(1 - dice_loss(mask, mask_ensemble))
    dice_merged.append(1 - dice_loss(mask, mask_merge))
    dice_cascaded.append(1 - dice_loss(mask, mask_64))
    # Intersection Over Union
    iou_unet.append(intersection_over_union(mask, mask_unet))
    iou_unet_plus.append(intersection_over_union(mask, mask_unet_plus_plus))
    iou_unet_attention.append(intersection_over_union(mask, mask_unet_attention))
    iou_ensembled.append(intersection_over_union(mask, mask_ensemble))
    iou_merged.append(intersection_over_union(mask, mask_merge))
    iou_cascaded.append(intersection_over_union(mask, mask_64))
    
    # Precision metrics
    precision_unet.append(precision_score_segmentation(mask, mask_unet))
    precision_unet_plus.append(precision_score_segmentation(mask, mask_unet_plus_plus))
    precision_unet_attention.append(precision_score_segmentation(mask, mask_unet_attention))
    precision_ensembled.append(precision_score_segmentation(mask, mask_ensemble))
    precision_merged.append(precision_score_segmentation(mask, mask_merge))
    precision_cascaded.append(precision_score_segmentation(mask, mask_64))
    
    # Recall metrics
    recall_unet.append(recall_segmentation(mask, mask_unet))
    recall_unet_plus.append(recall_segmentation(mask, mask_unet_plus_plus))
    recall_unet_attention.append(recall_segmentation(mask, mask_unet_attention))
    recall_ensembled.append(recall_segmentation(mask, mask_ensemble))
    recall_merged.append(recall_segmentation(mask, mask_merge))
    recall_cascaded.append(recall_segmentation(mask, mask_64))
    
    # F1-Score 
    f1_unet.append(f1_score_segmentation(mask, mask_unet))
    f1_unet_plus.append(f1_score_segmentation(mask, mask_unet_plus_plus))
    f1_unet_attention.append(f1_score_segmentation(mask, mask_unet_attention))
    f1_ensembled.append(f1_score_segmentation(mask, mask_ensemble))
    f1_merged.append(f1_score_segmentation(mask, mask_merge))
    f1_cascaded.append(f1_score_segmentation(mask, mask_64))
  
    # check if tumor is detected in the file
    if np.any(mask_unet):
        detected_unet += 1
        
    if np.any(mask_unet_plus_plus):
        detected_unet_plus += 1
        
    if np.any(mask_unet_attention):
        detected_unet_attention += 1
        
    if np.any(mask_ensemble):
        detected_ensembled += 1 
        
    if np.any(mask_merge):
        detected_merged += 1
    
    if np.any(mask_64):
        detected_cascaded += 1
end_time = timer()

print(f"\nSegmentation of the dataset took: {end_time  - start_time:.3f}seconds")
#%% Save the result on the text file
output = "summary.txt"
with open( output, 'w') as f:
    f.write("\t \t      Dice Coefficient \t IoU \t\t Precision \t Recall \t F1-Score \t tumors deteced in files\n")
    f.write(f"U-Net:           \t {np.mean(dice_unet)*100:.3f}% \t {np.mean(iou_unet)*100:.3f}% \t {np.mean(precision_unet)*100:.3f}% \t {np.mean(recall_unet)*100:.3f}% \t {np.mean(f1_unet)*100:.3f}% \t\t {detected_unet}/{len(src_list)}\n")
    f.write(f"U-Net++:         \t {np.mean(dice_unet_plus)*100:.3f}% \t {np.mean(iou_unet_plus)*100:.3f}% \t {np.mean(precision_unet_plus)*100:.3f}% \t {np.mean(recall_unet_plus)*100:.3f}% \t {np.mean(f1_unet_plus)*100:.3f}% \t\t {detected_unet_plus}/{len(src_list)}\n")
    f.write(f"Attention U-Net: \t {np.mean(dice_unet_attention)*100:.3f}% \t {np.mean(iou_unet_attention)*100:.3f}% \t {np.mean(precision_unet_attention)*100:.3f}% \t {np.mean(recall_unet_attention)*100:.3f}% \t {np.mean(f1_unet_attention)*100:.3f}% \t\t {detected_unet_attention}/{len(src_list)}\n")
    f.write(f"Merged:          \t {np.mean(dice_merged)*100:.3f}% \t {np.mean(iou_merged)*100:.3f}% \t {np.mean(precision_merged)*100:.3f}% \t {np.mean(recall_merged)*100:.3f}% \t {np.mean(f1_merged)*100:.3f}% \t\t {detected_merged}/{len(src_list)}\n")
    f.write(f"Ensemble:        \t {np.mean(dice_ensembled)*100:.3f}% \t {np.mean(iou_ensembled)*100:.3f}% \t {np.mean(precision_ensembled)*100:.3f}% \t {np.mean(recall_ensembled)*100:.3f}% \t {np.mean(f1_ensembled)*100:.3f}% \t\t {detected_ensembled}/{len(src_list)}\n")
    f.write(f"Cascaded:        \t {np.mean(dice_cascaded)*100:.3f}% \t {np.mean(iou_cascaded)*100:.3f}% \t {np.mean(precision_cascaded)*100:.3f}% \t {np.mean(recall_cascaded)*100:.3f}% \t {np.mean(f1_cascaded)*100:.3f}% \t\t {detected_cascaded}/{len(src_list)}\n")
