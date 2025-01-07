#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 06:04:12 2024
Objective:
    - Data needed for Model 1 is same as Model 1 of Casacdaing system.
    - Thus copying that dataset will be just fine    
    - Data for Model_2 is composed from 2 different sources
        - Source 1: https://www.kaggle.com/datasets/malicks111/brain-tumor-detection
        - Data is organized as Training and Testing.
            - Training : organized as glioma_tumor (6613), meningioma_tumor (6708), no_tumor (2842)
            and pituitary_tumor (6189)
            - Testing: glioma_tumor(620), meningioma_tumor(620), no_tumor (620) and pituitary_tumor (620)
        - Source 2: https://www.kaggle.com/datasets/zakariaolamine/brain-tumor-dataset
            - Data set contains 3064 weighted contrast-enhanced mri scans from 233 patients
            - Data comes in matlab .mat file, then mri scan 
            - Each .mat file labels coordinates for mask file, mri scan file tumor segmentation coordinates are extracted
            - Dataset is structured in glioma(1426), meningioma(708) and pituitary(930)
    - The purpose of this script is read the downloaded directory, shuffle the data, organize the dataset 
     to training, testing and val categories.

@author: dagi
"""
import os 
import shutil
import random 
from tqdm import tqdm 

# In[] Move the Data of Model_1 of Cascading system to this as well
root_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Classification/Cascading_1/Model_1/"

dest_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Classification/Cascading_2/Model_1/"
    
shutil.copytree(root_path, dest_path, dirs_exist_ok=True)
# %%  Data for model 2
src_1_path = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/"
root_dest = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Classification/Cascading_2/Model_2/"

pituitary = []
glioma = []
meningioma = []

# In[] Traverse through the directory 
for root, dirs, files in os.walk(src_1_path):
    for folder in dirs:
        if "_tumor"  in folder:
            folder_path = root + "/" + folder
            mri_scans = os.listdir(folder_path)
            for each_scan in mri_scans:
                scan_full_path = folder_path + "/" + each_scan
                if folder == "no_tumor":
                    continue
                elif folder == "glioma_tumor":
                    glioma.append(scan_full_path)
                elif folder == "meningioma_tumor":
                    meningioma.append(scan_full_path)
                elif folder == "pituitary_tumor":
                    pituitary.append(scan_full_path)
                
# In[] collect the mri scans from Source 2
glioma_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Brain_tumor_dataset_by_cheng/glioma/mri/"
meningioma_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Brain_tumor_dataset_by_cheng/meningioma/mri/"
pituitary_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Brain_tumor_dataset_by_cheng/pituitary/mri/"

src_path = [glioma_path, meningioma_path, pituitary_path]
# create list from directories
for index, path in enumerate(src_path):
    # create list from the path
    dir_list = os.listdir(path)
    # iterate through the list and comeupt with full_path
    for elements in dir_list:
        # full path
        full_path = path + elements
        # glioma and maningioma regarded as malignant and pituitary as meningioma 
        if index == 0:
            # append the path onto glioma container
            glioma.append(full_path)
        elif index == 1:
            # append the path onto meningioma container
            meningioma.append(full_path)
        else:
            pituitary.append(full_path)

# In[] shuffle the lists
random.shuffle(glioma)
random.shuffle(meningioma)
random.shuffle(pituitary)

# In[] Split the glioma tumors in the order of Test 10%, Train 70% and Val 20% directories 

train_end_index = int(0.7 * len(glioma)) # first 70% goes to training set
val_end_index = int(0.2 *  len(glioma)) + train_end_index # next 20% goes to validation set

dest_noTumor = [root_dest+"Train/Glioma/", root_dest + "Val/Glioma/", root_dest + "Test/Glioma/"]

# split glioma dataset into training, testing and val
for index, path in enumerate(tqdm(glioma, desc="Populating data in no_tumor dataset", leave=False)): 
    # decide to which dataset group the item should go
    if index <= train_end_index:
        # move the image to the training directory
        shutil.copy(path, dest_noTumor[0])
    elif index <= val_end_index:
        # copy and paste the image to the Val directory
        shutil.copy(path, dest_noTumor[1])
    else:
        # the remaining scans will go to test directory        
        shutil.copy(path, dest_noTumor[2])

# In[] Split the meningioma tumors in the order of Test 10%, Train 70% and Val 20% directories 

train_end_index = int(0.7 * len(meningioma)) # first 70% goes to training set
val_end_index = int(0.2 *  len(meningioma)) + train_end_index # next 20% goes to validation set

dest_noTumor = [root_dest+"Train/Meningioma/", root_dest + "Val/Meningioma/", root_dest + "Test/Meningioma/"]

# split meningioma dataset into training, testing and val
for index, path in enumerate(tqdm(meningioma, desc="Populating data in no_tumor dataset", leave=False)): 
    # decide to which dataset group the item should go
    if index <= train_end_index:
        # move the image to the training directory
        shutil.copy(path, dest_noTumor[0])
    elif index <= val_end_index:
        # copy and paste the image to the Val directory
        shutil.copy(path, dest_noTumor[1])
    else:
        # the remaining scans will go to test directory        
        shutil.copy(path, dest_noTumor[2])

# In[] Split the pituitary tumors in the order of Test 10%, Train 70% and Val 20% directories 

train_end_index = int(0.7 * len(pituitary)) # first 70% goes to training set
val_end_index = int(0.2 *  len(pituitary)) + train_end_index # next 20% goes to validation set

dest_noTumor = [root_dest+"Train/Pituitary/", root_dest + "Val/Pituitary/", root_dest + "Test/Pituitary/"]

# split pituitary dataset into training, testing and val
for index, path in enumerate(tqdm(pituitary, desc="Populating data in no_tumor dataset", leave=False)): 
    # decide to which dataset group the item should go
    if index <= train_end_index:
        # move the image to the training directory
        shutil.copy(path, dest_noTumor[0])
    elif index <= val_end_index:
        # copy and paste the image to the Val directory
        shutil.copy(path, dest_noTumor[1])
    else:
        # the remaining scans will go to test directory        
        shutil.copy(path, dest_noTumor[2])

                    

