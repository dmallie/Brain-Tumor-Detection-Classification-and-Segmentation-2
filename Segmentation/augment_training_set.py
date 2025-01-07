#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:59:13 2024
Objective:
    - Augment training data
    - Augmentation is performed by either rotating or flipping the mri and its corresponding mask
    - Rotation 90, 180 and 270 degrees
    - Flipping vertical, horizontal & both

@author: dagi
"""
import os
import cv2 
import random 
from tqdm import tqdm 

# In[] Set the routing path
root_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Thesis Project/Segmentation/Train/"
train_path = root_path + "Images/"
mask_path = root_path + "Masks/"
# create list of files from the directories
train_list = os.listdir(train_path)
mask_list = os.listdir(mask_path)

# In[] Set augmentation parameters
rotation_angles = [90, 180, 270]
flip_code = [0, 1, -1] # 1: horizontal flip, 0: vertical flip, -1: both flips @ same time

aug_list  = ["rotate", "flip"]
aug_dic = {
    "rotate" : rotation_angles,
    "flip"  : flip_code
    }

# In[] Iterate over train_list and perform augmentation
for img_file in tqdm(train_list, desc="Training set augmentation:",leave = False):
    # set the full path of the image
    img_path = train_path + img_file
    if img_path.endswith(".tif"):
        mask_file = mask_path + img_file.replace(".tif", "_mask.tif")
    else:
        mask_file = mask_path + img_file.replace(".jpg", ".png")
    # randomly select one of the three augmentation type
    aug_factor = random.sample(aug_list, 1)[0]
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Read the mask image
    mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    # Get the image dimensions (height, width)
    (h, w) = img.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)

    # do the augmentation procedure
    if aug_factor == "rotate":
         # iterate over rotation list
         for index, angle in enumerate(aug_dic[aug_factor]):
             # Get the rotation matrix
             rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
             # Perform the rotation 
             aug_img = cv2.warpAffine(img, rotation_matrix, (w, h))
             # save the rotated image
             dest_path = train_path + img_file[:-4] + "_rotate_" + str(index) + ".jpg"
             cv2.imwrite(dest_path, aug_img)
             ############### Do Transformation on the Mask ##################
             # transform label values of the image
             dest_mask_path = mask_path + img_file[:-4] + "_rotate_" + str(index) + ".png"
             rotated_mask = cv2.warpAffine(mask_img, rotation_matrix, (w, h))
             # convert the label coordinates to string
             cv2.imwrite(dest_mask_path, rotated_mask)         
    else:
        # iterate over flip_code
        for index, flip in enumerate(aug_dic[aug_factor]):
            # Flip the image
            flipped_img = cv2.flip(img, flip)
            # save the flipped images
            dest_path = train_path + img_file[:-4] + "_flipped_" + str(index) + ".jpg"
            cv2.imwrite(dest_path, flipped_img)
            ############### Do Transformation on the Mask ##################
            # transform label values of the image
            dest_mask_path = mask_path + img_file[:-4] + "_flipped_" + str(index) + ".png"
            flipped_mask = cv2.flip(mask_img, flip)
            cv2.imwrite(dest_mask_path, flipped_mask)

    
