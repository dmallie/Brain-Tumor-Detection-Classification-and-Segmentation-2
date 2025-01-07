#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 06:08:57 2024
Objective:
    - This script using the Test data set evaluates the perfromance of 
        out cascading systems in classifying Brain MRI as Healthy, Pituitary, Glioma or Meningioma
    - Dataset is composed in the following fashion
        - No_Tumor: Model_1/Test/No_tumor
        - Pituitary: Model_2/Test/Benign
        - Glioma : Model_3/Test/Glioma
        - Meningioma: Model_3/Test/Meningioma
@author: dagi
"""
import os 
from torch.utils.data import DataLoader 
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from utils import mean_std, precision, accuracy_calculator, recall, f1_score
from tqdm import tqdm 
from PIL import Image
from sklearn.metrics import confusion_matrix 
import numpy as np 

# In[] Set route path
rootPath = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Classification/Cascading_1/"

noTumor_path = rootPath + "Model_1/Test/No_tumor/"
pituitary_path = rootPath + "Model_2/Test/Benign/"
glioma_path = rootPath + "Model_3/Test/Glioma/"
meningioma_path = rootPath + "Model_3/Test/Meningioma/"

dataset_path = [noTumor_path, pituitary_path, glioma_path, meningioma_path]
# %% Create lists from teh directory
noTumor = os.listdir(noTumor_path)
pituitary = os.listdir(pituitary_path)
glioma = os.listdir(glioma_path)
meningioma = os.listdir(meningioma_path)

test_elements = [noTumor, pituitary, glioma, meningioma]
# %% create test_path from the created lists with full_path for each items
test_path = []
for index, src_list in enumerate(test_elements):
    for elements in src_list:
        # get the full path for the elements
        full_path = dataset_path[index] + elements
        # append the full_path on to the test_path
        test_path.append(full_path)
        
# %% Setting Hyperparameter
WIDTH = 256
HEIGHT = 256 
OUTPUT_SHAPE = 1 
BATCH = 1 

#%% Calculate the mean and std of the dataset 
mean, std = mean_std(test_path)

# In[] Set transform function
transform_fn = transforms.Compose([
                        # Resize the image to fit the model
                        transforms.Resize(size=(HEIGHT, WIDTH)),
                        # Convert image to grayscale 
                        transforms.Grayscale(num_output_channels=1),
                        # Convert image to tensor object
                        transforms.ToTensor(),
                        # Normalize the tensor object
                        transforms.Normalize(mean=[mean], std = [std])
                ])


# In[] Set paths for the different models we have
model_1_path = "/home/dagi/Documents/PyTorch/MIP/Final_3/Classification/Cascading 1/Model 1/densenet_169.pth"
model_2_path = "/home/dagi/Documents/PyTorch/MIP/Final_3/Classification/Cascading 1/Model 2/densenet_169.pth"
model_3_path = "/home/dagi/Documents/PyTorch/MIP/Final_3/Classification/Cascading 1/Model 3/densenet_169.pth"


# %% Load the models
model_1 = torch.load(model_1_path, weights_only=False)
model_2 = torch.load(model_2_path, weights_only=False)
model_3 = torch.load(model_3_path, weights_only=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_1 = model_1.to(device)
model_2 = model_2.to(device)
model_3 = model_3.to(device)

# In[] Prepare models and evaluation mode
model_1.eval()
model_2.eval()
model_3.eval()

true_labels = []
predicted_labels = []

count_pituitary = 0
labels = []
# iterate through each image in test_path
for img_path in tqdm(test_path, desc="Test classification models", leave=False):
    # load the image
    img = Image.open(img_path)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # get the image label 
    if "No_tumor" in img_path:
        label = 0
    elif "Benign" in img_path:
        label = 1
    elif "Glioma" in img_path:
        label = 2
    elif "Meningioma" in img_path:
        label = 3 
    else:
        raise ValueError("Unknown label")
    
    true_labels.append(label)
    # Transform the image then add the batch dimension
    img_transformed = transform_fn(img).unsqueeze(0).to(device) 
    #################################################################
    # perform evaluation task 
    with torch.no_grad():
        # Model 1 prediction
        logits = model_1(img_transformed)
        probabilities = torch.sigmoid(logits)
        prediction = (probabilities >= 0.5).long()
        if prediction == 0: # No tumor is detected in the MRI
            predicted_labels.append(0)
            continue
        # Model 2 prediction if Tumor is detected in the MRI
        logits = model_2(img_transformed)
        probabilities = torch.sigmoid(logits)
        prediction = (probabilities >= 0.5).long()
        if prediction == 0: # Tumor is Pituitary 
            predicted_labels.append(1)
            continue
        # Model 3 prediction
        logits = model_3(img_transformed)
        probabilities = torch.sigmoid(logits)
        prediction = (probabilities >= 0.5).long()
        if prediction == 0: # Tumor is glioma
            predicted_labels.append(2)
        else:
            predicted_labels.append(3) # tumor is meningioma
        
#%% Plot Confusion Matrix with Matplotlib
# Define class names
class_names = ['No_tumor', 'Pituitary', 'Glioma', 'Meningioma']
# generate the confusion matrix 
confusion_m = confusion_matrix(np.asarray(true_labels), np.asarray(predicted_labels), labels = np.arange(len(class_names)))
# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(confusion_m, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar(cax)

# Add labels to axes
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

# Label axes
plt.xlabel('Predicted')
plt.ylabel('True')

# Add text annotations
for i in range(confusion_m.shape[0]):
    for j in range(confusion_m.shape[1]):
        ax.text(j, i, f'{confusion_m[i, j]}', ha='center', va='center', color='red')

plt.show()
#%% model 1 output to calculate its efficiency.
# Model_1: classify 3553 images as 0:no_tumor = 1107 and 1: tumor = 2446 stored in model_1_output
# model_1_output is re-written of predicted labels as 0: no_tumor and all others as 1
# true_labels_1 is re-written of true_labels as 0: no_tumor and different from 0 as 1
model_1_output = []

true_labels_1 = []
for index, value in enumerate(predicted_labels):
    if value != 0: # tumor is detected d
        model_1_output.append(1)
    else: # no tumor
        model_1_output.append(0)
    if true_labels[index] != 0:
        true_labels_1.append(1)
    else:
        true_labels_1.append(0)
        
#%% model_2 output: 
# model_2: classify the 2446 images as 0: benign = 770 and 1: malignant = 1676
# model_2_output is re-written of the remaining predicted labels as 1: benign and 2 & 3 as 1
# true_labels_2 is re-written of of those true_labels as 0: no_tumor and different from 0 as 1
 
model_2_output = []
true_labels_2 = []
for index, value in enumerate(predicted_labels):
    if value == 0:
        continue 
    elif value == 1:
        model_2_output.append(0) # it's benign and it's 0 for model_2
        if true_labels[index] == 1:
            true_labels_2.append(0)
        else:
            true_labels_2.append(1)
    else: # it's malignant
        model_2_output.append(1)
        if true_labels[index] == 1:
            true_labels_2.append(0)
        else:
            true_labels_2.append(1)
#%% model_3 output
# Model_3: classify the remianing 1676 images as 0: glioma = 864 and 1: meningioma = 812
# model_3_output is re-written of the remaining predicted labels as 2: glioma and 3 as 1
# true_labels_3 is re-written of of those true_labels as 2 as 0: glioima and  3 as 1: meningioma

model_3_output = []
true_labels_3 = []
for index, value in enumerate(predicted_labels):
    if value == 0 or value == 1:
        continue 
    elif value == 2:
        model_3_output.append(0)
        if true_labels[index] == 2:
            true_labels_3.append(0)
        else:
            true_labels_3.append(1)
    else:
        model_3_output.append(1)
        if true_labels[index] == 2:
            true_labels_3.append(0)
        else:
            true_labels_3.append(1)

#%% Evaluate the predicted_labels with the performance measuring metrics
class_labels = ["No Tumor", "Pituitary", "Glioma", "Meningioma"]

correct_predictions = accuracy_calculator(predicted_labels, true_labels)
accuracy = correct_predictions/len(true_labels)
precision_ = precision(predicted_labels, true_labels, class_labels)
recall_ = recall(predicted_labels, true_labels, class_labels)
F1_score = f1_score(predicted_labels, true_labels, class_labels) 

# model 1 metrics
class_labels_1 = ["No Tumor", "Tumor"]
correct_predictions_1 = accuracy_calculator(model_1_output, true_labels_1)
accuracy_1 = correct_predictions_1/len(true_labels_1)
precision_1 = precision(model_1_output, true_labels_1, class_labels_1)
recall_1 = recall(model_1_output, true_labels_1, class_labels_1)
F1_score_1 = f1_score(model_1_output, true_labels_1, class_labels_1) 

# model 2 metrics
class_labels_2 = ["Benign", "Malignant"]
correct_predictions_2 = accuracy_calculator(model_2_output, true_labels_2)
accuracy_2 = correct_predictions_2/len(true_labels_2)
precision_2 = precision(model_2_output, true_labels_2, class_labels_2)
recall_2 = recall(model_2_output, true_labels_2, class_labels_2)
F1_score_2 = f1_score(model_2_output, true_labels_2, class_labels_2) 

# model 3 metrics
class_labels_3 = ["Glioma", "Meningioma"]
correct_predictions_3 = accuracy_calculator(model_3_output, true_labels_3)
accuracy_3 = correct_predictions_3/len(true_labels_3)
precision_3 = precision(model_3_output, true_labels_3, class_labels_3)
recall_3 = recall(model_3_output, true_labels_3, class_labels_3)
F1_score_3 = f1_score(model_3_output, true_labels_3, class_labels_3) 


#%% Log out the performance metrics
txt_lines = [
    "System's Performance\n",
    f"\t- System's prediction: {correct_predictions}/{len(true_labels)}\n",
    f"\t- System's classification accuracy: {accuracy * 100:.3f}%\n",
    f"\t- System's precision output: {precision_*100:.3f}%\n"
    f"\t- System's recall output: {recall_*100:.3f}%\n"
    f"\t- System's F1-Score output: {F1_score*100:.3f}%\n"
    "######################################################################\n"
    f"\t- Accuracy of Healthy Dataset (Accuray of model 1): {confusion_m[0][0]}/{len(noTumor)} ie {confusion_m[0][0]/len(noTumor)*100:.3f}%\n",
    f"\t- Accuracy of Pituitary Dataset (Accuracy of model 2): {confusion_m[1][1]}/{len(pituitary)} ie {confusion_m[1][1]/len(pituitary)*100:.3f}%\n",
    f"\t- Accuracy of model 3\n"
    f"\t\t- Accuracy of Glioma Dataset: {confusion_m[2][2]}/{len(glioma)} ie {confusion_m[2][2]/len(glioma)*100:.3f}%\n",
    f"\t\t- Accuracy of Meningioma Dataset: {confusion_m[3][3]}/{len(meningioma)} ie {confusion_m[3][3]/len(meningioma)*100:.3f}%\n",
    "######################################################################\n"
    f"\t- model_1's prediction: {correct_predictions_1}/{len(true_labels_1)}\n",
    f"\t- model_1's classification accuracy: {accuracy_1 * 100:.3f}%\n",
    f"\t- model_1's precision output: {precision_1*100:.3f}%\n"
    f"\t- model_1's recall output: {recall_1*100:.3f}%\n"
    f"\t- model_1's F1-Score output: {F1_score_1*100:.3f}%\n"
    "######################################################################\n"    
    f"\t- model_2's prediction: {correct_predictions_2}/{len(true_labels_2)}\n",
    f"\t- model_2's classification accuracy: {accuracy_2 * 100:.3f}%\n",
    f"\t- model_2's precision output: {precision_2*100:.3f}%\n"
    f"\t- model_2's recall output: {recall_2*100:.3f}%\n"
    f"\t- model_2's F1-Score output: {F1_score_2*100:.3f}%\n"
    "######################################################################\n"
    f"\t- model_3's prediction: {correct_predictions_3}/{len(true_labels_3)}\n",
    f"\t- model_3's classification accuracy: {accuracy_3 * 100:.3f}%\n",
    f"\t- model_3's precision output: {precision_3*100:.3f}%\n"
    f"\t- model_3's recall output: {recall_3*100:.3f}%\n"
    f"\t- model_3's F1-Score output: {F1_score_3*100:.3f}%\n"

]
with open("summary.txt", 'w') as f:
    f.writelines(txt_lines)