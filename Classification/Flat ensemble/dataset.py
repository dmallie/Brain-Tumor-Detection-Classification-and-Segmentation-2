#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 04:55:57 2024
Objective:
    - Prepare the dataset for the implemenation of Flat Ensemble
    - Dataset from previous project Cascading 2 can be reused here but needs restructuing
    - No-Tumor from cascading 2 model 1
    - Glioma, Meningioma and Pituitary from cascading 2 model 2 will form the dataset we need
@author: dagi
"""
import shutil 

# %% Path to No-Tumor dataset 
no_tumor_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Classification/Cascading_2/Model_1/"
no_tumor_test = no_tumor_path + "Test/No_tumor/"
no_tumor_train = no_tumor_path + "Train/No_tumor/"
no_tumor_val = no_tumor_path + "Val/No_tumor/"

#%% Set the destination path
dest_path  = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Classification/Flat Ensemble/"
dest_path_test = dest_path + "Test/No_tumor/"
dest_path_train = dest_path + "Train/No_tumor/"
dest_path_val = dest_path + "Val/No_tumor/"

# %% Copy No-Tumor data structure to destination folder
shutil.copytree(no_tumor_test, dest_path_test, dirs_exist_ok=True)
shutil.copytree(no_tumor_train, dest_path_train, dirs_exist_ok=True)
shutil.copytree(no_tumor_val, dest_path_val, dirs_exist_ok=True)

#%% Path of Glioma, Meningioma and Pituitary dataset in Cascading 2 Model 2
model_2_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Classification/Cascading_2/Model_2/"
shutil.copytree(model_2_path, dest_path, dirs_exist_ok=True)


