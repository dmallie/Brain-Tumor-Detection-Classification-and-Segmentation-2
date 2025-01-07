#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:59:37 2024
Objective:
    - Compute the accuracy of the segmentation model:
        - by calculating the dice_loss
        - by calculating IntersectionOverunion 
        - by caculaatign the average distance between the centroids of original and predicted masks in pixels
        - by calculating the average difference in the area of the bounding rectangles
@author: dagi
"""
import torch
import torchvision
import torch.nn as nn
# from dataset import BrainTumorDataset
from torch.utils.data import DataLoader 
import numpy as np 
import os 
import cv2 
from skimage.filters import threshold_otsu
from scipy.ndimage import label

# In[0] Saving the checkpoint
def save_checkpoint(model, file_name="checkpoint.path.tar"):
    torch.save(model.state_dict(), file_name)
    
# In[1] To load the checkpoint
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict( checkpoint["state_dict"] )
    
# In[2] Set augmenation pipeline
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512


     
# In[3] 
"""
Objectives
- Creates training_dataset
- Creates training_dataloader
- Creates validation_dataset
- Creates validation_dataloader
"""
def get_loaders(train_ds, 
                val_ds,
                batch_size
                ):
    # Create Train & validation dataset
    # train_ds = CarvanaDataset(train_dir, train_maskdir, train_transform )
    # val_ds = CarvanaDataset(val_dir, val_maskdir, val_transform)
    
    # Create Training and validation DataLoader
    train_dataloader = DataLoader(dataset = train_ds,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 4,
                                  pin_memory = True)
    val_dataloader = DataLoader(dataset = val_ds,
                                batch_size = batch_size,
                                shuffle = False,
                                num_workers = 4,
                                pin_memory = True
                                )
    return train_dataloader, val_dataloader 

# In[4] 
"""
- Calculates the accuracy of the model
- Loss is calculated in 
"""
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, mask):
        # Flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)
        # Calculate the intersection and union
        counter = (pred * mask).sum()  # Numerator OR Intersection      
        denum = pred.sum() + mask.sum() + 1e-8  # Denominator. Add a small number to prevent NANS ie Union
        # Dice coefficient
        dice =  (2*counter)/denum
        return 1 - dice

# In[] IntersectionOverUnion IoU
class IoU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(sefl, orig_mask, seg_mask, eps = 1e-6):
        # orig_mask and seg_mask is tensor objects needs to be reside in cpu
        # orig_mask = orig_mask.cpu()
        # seg_mask = seg_mask.cpu()
        
        # Flatten the tensors
        preds = seg_mask.view(-1)
        targets = orig_mask.view(-1)
        
        # Calculate intersection and union
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        
        # Calculate IoU
        iou = intersection / (union + eps)
        
        return iou

# In[] 
def check_accuracy(loader, model, device = "cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    loss_fn = IoU()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            
            preds = torch.sigmoid(model(x))
            # print(f"preds.shape: {preds.shape}")
            preds  =(preds > 0.5).float()
            num_correct += (preds == y).sum() # count the number of right pixels
            num_pixels += torch.numel(preds) # returns the total number of elements in the tensor
            dice_score += (2 * (preds *  y).sum()) / ((preds + y).sum() + 1e-8)
            loss = loss_fn(preds, y)
    accuracy = round((num_correct.item()/num_pixels)*100, 2)       
    print(f"Got {num_correct}/{num_pixels} with acc {accuracy}")
    print(f"Dice score: {dice_score/len(loader):.2f}")
    print(f"DiceLoss: {loss}")
    model.train()
# In[5]
def save_predictions_as_imgs(loader, model, folder ="saved_images/", device="cuda"):
    model.eval()
    save_path = "/home/dagi/Documents/PyTorch/MIP/Final_2/Segmentation/" + folder
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
        torchvision.utils.save_image(preds, f"{save_path}/pred_{idx}.png")
        torchvision.utils.save_image(y, f"{save_path}{idx}.png")
    model.train()

# In[] Calculate the mean and standard deviation for test dataset
def calculate_mean_std(path, height=None, width=None):
    # create list of all images with their full_path
    dataset_mri = []
    for dirPath, dirNames, fileNames in os.walk(path):
        for file in fileNames:
            if file.endswith('.jpg' ) or file.endswith('.tif'):
                full_path = dirPath + "/" + file
                dataset_mri.append(full_path)

    # set parameters based on which to process mean
    sum_img = None # accummulate the sum of pixel values of the entire dataset
    # height = 256 
    # width = 256
    print(f"height: {height} \t width: {width}\t len(dataset_mri): {len(dataset_mri)}")
    # Calculate the mean
    for img_name in dataset_mri:
        # Read the image in grayscale
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        # Resize the image to 256x256 for consistency
        img = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)
        # accumulate teh sum of pixel values of each individual pixels
        if sum_img is None:
            sum_img = img / 255
        else:
            sum_img += img/255
    print(f"sum_img: {sum_img.shape}")
    #  calculating the mean
    mean_img = sum_img / len(dataset_mri)

    # Calculate  the mean value of pixels for each channel
    mean_pixel_value = np.mean(mean_img, axis=(0, 1))

    # set parameters for standard deviation
    sum_squared_img = None
    squared_diff = 0

    # calculate the standard deviation
    for img_path in dataset_mri:
        # Read the image as Grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Resize the image to 256x256
        img = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)

        # Accumulate the squared differences from the mean image
        squared_diff = (img/255 - mean_img) ** 2
        if sum_squared_img is None:
            sum_squared_img = squared_diff
        else:
            sum_squared_img += squared_diff
    # Calculating the variance
    variance = sum_squared_img / len(dataset_mri)

    # Standard Deviation
    std = np.sqrt(np.mean(variance, axis = (0, 1)))

    # return the mean and standard deviation of the dataset
    return mean_pixel_value, std

# %% Using simple threshold mechanism we enhance the image 

def simple_skull_strip(image):
    # Normalize image
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Apply Otsu's thresholding
    thresh = threshold_otsu(image)
    binary_mask = image > thresh

    # Morphological cleanup (optional)
    from scipy.ndimage import binary_closing
    binary_mask = binary_closing(binary_mask, structure=np.ones((3, 3)))

    # Apply mask
    skull_stripped_image = image * binary_mask
    return skull_stripped_image

# In[] Dice Loss: calculates the dice loss between two binary masks
def dice_loss(orig_mask, seg_mask, epsilon = 1e-6):
    # convert to float32 numpy array
    orig_mask = orig_mask.astype(np.float32)/255.
    seg_mask = seg_mask.astype(np.float32)/255.
    # Compute the intersection that is the sum of element wise multiplication
    intersection = np.sum(orig_mask *  seg_mask)
    # Compute the union between the two masks that is sum of element wise addition
    union = np.sum(orig_mask) + np.sum(seg_mask) + epsilon
    # Compute the dice coefficient
    dice_coef = (2 * intersection)/union
    # Calculate the dice loss
    dice_loss = 1 - dice_coef
    # return the dice loss
    return np.round(dice_loss, 3)

# In[] IntersectionOverUnion IoU
def intersection_over_union(orig_mask, seg_mask, eps = 1e-6):
    # convert to float32 numpy array
    orig_mask = orig_mask.astype(np.uint8)
    seg_mask = seg_mask.astype(np.uint8)
        
    # Calculate the intersection (logical AND)
    intersection = np.logical_and(orig_mask, seg_mask).sum()
    
    # Calculate the union (logical OR)
    union = np.logical_or(orig_mask, seg_mask).sum()
    
    
    # Compute the IoU and subtract from 1 to make it a loss
    iou = intersection /(union + eps)
    # return the loss
    return np.round(iou, 3)    


# In[] BCE  adn Dice Loss
def bce_dice_loss(pred, target, bce_weight=0.5, smooth=1e-6):
    # Ensure predictions are in the range [0, 1]
    pred = np.clip(pred, smooth, 1 - smooth)

    # 1. Binary Cross-Entropy (BCE) Loss
    bce_loss = -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

    # 2. Dice Loss
    intersection = np.sum(pred * target)
    dice_loss = 1 - (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

    # 3. Combined BCE + Dice Loss
    bce_dice_loss = bce_weight * bce_loss + (1 - bce_weight) * dice_loss

    return bce_dice_loss

# In[] Pixel wise comparison 
def pixelwise_comparison(orig_mask, seg_mask):
    # Convert to torch tensor
    orig_mask = torch.tensor(orig_mask)
    seg_mask = torch.tensor(seg_mask)
    # select only nonzero values
    orig_nonzero = torch.nonzero(orig_mask)
    seg_nonzero = torch.nonzero(seg_mask)
    # calculate the percentile difference between the two binary file
    if len(seg_nonzero) == 0:
        return 0
    if len(seg_nonzero) > len(orig_nonzero):
        return np.round((len(orig_nonzero)/len(seg_nonzero))*100, 3)
    else:
        return np.round((len(seg_nonzero)/len(orig_nonzero))*100, 3)

# In[] Calculates the area of teh rectangle
def get_rec_area(orig_coord, seg_coord):
    # unpack the orig_coord & seg_coord
    orig_top_left = orig_coord[0]
    orig_bottom_right = orig_coord[1]
    
    seg_top_left = seg_coord[0]
    seg_bottom_right = seg_coord[1]
    # calculate the width of both rectangles
    orig_width = orig_bottom_right[1] - orig_top_left[1]
    seg_width = seg_bottom_right[1] - seg_top_left[1]
    # claculate the height of both rectangles
    orig_height = orig_bottom_right[0] - orig_top_left[0]
    seg_height = seg_bottom_right[0] - seg_top_left[0]

    # calculate the area
    orig_area = orig_height * orig_width
    seg_area = seg_height * seg_width
    
    # calculate the ratio of the two areas
    if  seg_area == 0:
        return 0
    if orig_area > seg_area:
        return np.round((seg_area/orig_area)*100, 3)
    else:
        return np.round((orig_area/seg_area)*100, 3)
# In[] Calculates the sizes of the masks
def get_area(mask, segmented):
    # Perform connected components labeling using scipy
    labeled_mask, num_balls_mask = label(mask)
    labeled_segmented, num_balls_segmented = label(segmented)

    # Calculate size (number of pixels) for each component
    mask_area = [(labeled_mask == i).sum() for i in range(1, num_balls_mask + 1)]
    segmented_area = [(labeled_segmented == i).sum() for i in range(1, num_balls_segmented+1)]
    # print(f"mask_area: {mask_area}\t segmented_area: {segmented_area}")
    # calculate the ratio
    return np.round((sum(segmented_area) / sum(mask_area))*100, 3)

# In[] Calculates distance between the centroids
def get_dist_centroid(mask, segmented):
    # Perform connected components labeling using scipy
    labeled_mask, mask_balls = label(mask)
    labeled_segmented, segmented_balls = label(segmented)
    # calculate the centroid of tumor mass
    coords = np.argwhere(labeled_mask == 1)
    centroid_mask = coords.mean(axis=0)  # Calculate centroid
    
    centroid_segmented = []
    for i in range(1, segmented_balls + 1):
        coords = np.argwhere(labeled_segmented == i)
        centroid = coords.mean(axis=0)
        centroid_segmented.append(centroid)
    # Compute the Euclidean distance between centroids
    distances = []
    for c in centroid_segmented:
        # calcualte the a in the pythagores theorm a² + b² = c²
        a = abs(centroid_mask[0] - c[0])
        b = abs(centroid_mask[1] - c[1])
        # calculate the c which is the distance b/n the centroids
        distance = np.sqrt(a**2 + b**2)
        distances.append(distance)
    # return the distance
    if distances:
        return np.round(np.mean(distances), 3)
    else:
        return 0

# In[] Get the coordinates & centroid of the bounding rectangle
def bounding_rectangle(orig_mask, seg_mask):
    # Convert to torch tensor datatype
    orig_mask = torch.tensor(orig_mask)
    seg_mask = torch.tensor(seg_mask)
    # extract only the nonzero values 
    orig_nonzero = torch.nonzero(orig_mask)
    seg_nonzero = torch.nonzero(seg_mask)
    # if the nonzero is empty then we assign (0,0) (0, 0) to the rectangle
    if len(orig_nonzero) == 0:
        orig_top_left = (0, 0)
        orig_bottom_right = (0, 0)
    # Else get the coordinates for the top_left and bottom_right corners
    else:
        orig_top_left  = torch.min(orig_nonzero, dim=0)[0]
        orig_bottom_right = torch.max(orig_nonzero, dim=0)[0]
    
    if len(seg_nonzero) == 0:
        seg_top_left = (0, 0)
        seg_bottom_right = (0, 0)
    else:
        seg_top_left = torch.min(seg_nonzero, dim=0)[0]
        seg_bottom_right = torch.max(seg_nonzero, dim=0)[0]
    center_x1 = int(orig_top_left[1] + (orig_bottom_right[1] - orig_top_left[1])/2)
    center_y1 = int(orig_top_left[0] + (orig_bottom_right[0] - orig_top_left[0])/2)
    # center for the segmented box
    center_x2 = int(seg_top_left[1] + (seg_bottom_right[1] - seg_top_left[1])/2)
    center_y2 = int(seg_top_left[0] + (seg_bottom_right[0] - seg_top_left[0])/2)
 #   print(f"orig_top_left: {orig_top_left}")
    orig_top_left = orig_top_left
    orig_bottom_right = orig_bottom_right
    # collect the bounding corder of the rectangle
    orig_coord = (orig_top_left, orig_bottom_right)
    seg_coord = (seg_top_left, seg_bottom_right)
    # calculate the centroid of the rectangle
    center_x_orig = (orig_bottom_right[1] - orig_top_left[1])/2
    center_y_orig = (orig_bottom_right[0] - orig_top_left[0])/2
    orig_center = (center_y1, center_x1)
    
    seg_center = (center_y2, center_x2)
    #print(f"type(orig_coord): {type(orig_coord)}")
    # return the values
    return orig_coord, orig_center, seg_coord, seg_center

# %% Implementation of Average Difference

def average_difference(ground_truth, prediction):
    # Ensure both inputs are numpy arrays
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    # Normalize the images
    ground_truth = (ground_truth / 255).astype(int)
    prediction = (prediction / 255).astype(int)

    # Compute the absolute difference
    absolute_difference = np.abs(ground_truth - prediction)
    # print(f"absolute_difference.shape: {absolute_difference.shape}")
    # Compute the average difference
    average_diff = np.mean(absolute_difference)
    # print(f"average_diff: {average_diff}")
    return average_diff

# %% Implementation of precision
def precision_score_segmentation(ground_truth, prediction):
    # Ensure both inputs are numpy arrays
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    # Normalize the images
    ground_truth = (ground_truth / 255).astype(int)
    prediction = (prediction / 255).astype(int)
    # print(f"ground_truth: {ground_truth.shape}\t prediction: {prediction.shape}")
    # true_positives = 0
    # false_positives = 0
    # Compute True Positives (TP) and False Positives (FP)
    # for i in range(ground_truth.shape[0]):
    #     for j in range(ground_truth.shape[1]):
    #         true_positives += 1 if ground_truth[i][j] == 1 & prediction[i][j] == 1 else 0
    #         false_positives += 1 if ground_truth[i][j] == 0 & prediction[i][j] == 1 else 0
    true_positives = np.sum((prediction == 1) & (ground_truth == 1))
    false_positives = np.sum((prediction == 1) & (ground_truth == 0))
    
    # print(f"true_positive: {true_positives}\t false_positives: {false_positives}\n")
    # Calculate Precision
    if true_positives + false_positives == 0:
        # Avoid division by zero; return 0 precision if no positives were predicted
        return 0.0
    
    precision = true_positives / (true_positives + false_positives)
    
    return precision

#%% Recall Implementation on Segmentation
def recall_segmentation(ground_truth, prediction):
    # Ensure both inputs are numpy arrays
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    # Normalize the images
    ground_truth = (ground_truth / 255).astype(int)
    prediction = (prediction / 255).astype(int)
    # Calculate the True Positive (TP) and False Negative (FN)
    TP = np.sum((prediction == 1) & (ground_truth == 1))
    FN = np.sum((prediction == 0) & (ground_truth == 1))
    
    # Calculate the recall
    if TP + FN == 0:
        return 1.0 # This avoids division by zero
    else:
        recall = TP/(TP + FN)
        return recall
# %% F1-Score Implementation 
def f1_score_segmentation(ground_truth, prediction):
    # print(f"ground_truth: {ground_truth.shape}\t prediction: {prediction.shape}")
    precision = precision_score_segmentation(ground_truth, prediction)
    recall = recall_segmentation(ground_truth, prediction)
    
    if (precision + recall) == 0:
        return 0.0 # This avoids division by 0
    
    f1_score = (2 * precision *  recall)/(precision + recall)
    
    return f1_score 
 
