#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 07:56:21 2024
Objective:
    - This script using the Test data set evaluates the perfromance of 
        ensemble techniques in classifying Brain MRI as Healthy, Pituitary, Glioma or Meningioma
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
import torch.nn.functional as F

# In[] Set route path
rootPath = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Classification/Flat Ensemble/Test/"

noTumor_path = rootPath + "No_tumor/"
pituitary_path = rootPath + "Pituitary/"
glioma_path = rootPath + "Glioma/"
meningioma_path = rootPath + "Meningioma/"

dataset_path = [noTumor_path, pituitary_path, glioma_path, meningioma_path]

# %% Create lists from teh directory
noTumor = os.listdir(noTumor_path)
pituitary = os.listdir(pituitary_path)
glioma = os.listdir(glioma_path)
meningioma = os.listdir(meningioma_path)

test_elements = [noTumor, pituitary, glioma, meningioma]

# %% the path of each image will be stored in test_path
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
OUTPUT_SHAPE = 4 # 0: Glioma, 1: Meningioma, 2: No_tumor and 3: Pituitary
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
densenet_path = "densenet_169.pth"
efficientnet_path = "EfficientNet.pth"
resnet_path = "ResNet_50.pth"

# %% Load the models
densenet = torch.load(densenet_path, weights_only=False)
efficientnet = torch.load(efficientnet_path, weights_only=False)
resnet = torch.load(resnet_path, weights_only=False)

#%% move models onto the cuda device
device = "cuda" if torch.cuda.is_available() else "cpu"

densenet = densenet.to(device)
efficientnet = efficientnet.to(device)
resnet = resnet.to(device)

#%% Set the models in evaluation mode 
densenet.eval()
efficientnet.eval() 
resnet.eval()

# %% Set the conatiners that hold the results & iterate through each image in test_path
true_labels = []
predicted_labels = []

densenet_pred = []
resnet_pred = []
efficientnet_pred = []

for img_path in tqdm(test_path, desc="Test classification models", leave=False):
    # load the image
    img = Image.open(img_path)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # get the image label 
    if "No_tumor" in img_path:
        label = 2
    elif "Glioma" in img_path:
        label = 0
    elif "Meningioma" in img_path:
        label = 1
    elif "Pituitary" in img_path:
        label = 3 
    else:
        raise ValueError("Unknown label")
    
    true_labels.append(label)
    # Transform the image then add the batch dimension
    img_transformed = transform_fn(img).unsqueeze(0).to(device) 
    #################################################################
    # perform evaluation task 
    with torch.no_grad():
        # Model 2 predicts What type of Tumor it is
        logits = densenet(img_transformed)
        probabilities_1 = F.softmax(logits, dim=1) # convert predictions to probabilities
        pred_1 = torch.argmax(probabilities_1, dim=1)
        densenet_pred.append(pred_1.item())
        
        logits = efficientnet(img_transformed)
        probabilities_2 = F.softmax(logits, dim=1) # convert predictions to probabilities
        pred_2 = torch.argmax(probabilities_2, dim=1)
        efficientnet_pred.append(pred_2.item())
        
        logits = resnet(img_transformed)
        probabilities_3 = F.softmax(logits, dim=1) # convert predictions to probabilities
        pred_3 = torch.argmax(probabilities_3, dim=1)
        resnet_pred.append(pred_3.item())
        
        # Stack the probabilities along a new dimension
        stacked = torch.stack((probabilities_1, probabilities_2, probabilities_3)) # shape: [3, 3]

        # compute the mean along dimension 0 or Row
        average_pool = torch.mean(stacked, dim=0)

        prediction = torch.argmax(average_pool, dim=1) # index of maximum value along the row
        # print(prediction_2.shape)
        # write the predicted tumor in the predicted_labels container
        predicted_labels.append(prediction.item())

#%% Plot Confusion Matrix with Matplotlib
# Define class names
class_names = ['Glioma', 'Meningioma', 'No_tumor', 'Pituitary']
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

#%% Evaluate the predicted_labels with the performance measuring metrics
class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

correct_predictions = accuracy_calculator(predicted_labels, true_labels)
accuracy = correct_predictions/len(true_labels)
precision_ = precision(predicted_labels, true_labels, class_labels)
recall_ = recall(predicted_labels, true_labels, class_labels)
F1_score = f1_score(predicted_labels, true_labels, class_labels) 

# ResNet-50 metrics
correct_predictions_resnet = accuracy_calculator(resnet_pred, true_labels)
accuracy_1 = correct_predictions_resnet/len(true_labels)
precision_1 = precision(resnet_pred, true_labels, class_labels)
recall_1 = recall(resnet_pred, true_labels, class_labels)
F1_score_1 = f1_score(resnet_pred, true_labels, class_labels) 

# DenseNet-169 metrics
correct_predictions_densenet = accuracy_calculator(densenet_pred, true_labels)
accuracy_2 = correct_predictions_densenet/len(true_labels)
precision_2 = precision(densenet_pred, true_labels, class_labels)
recall_2 = recall(densenet_pred, true_labels, class_labels)
F1_score_2 = f1_score(densenet_pred, true_labels, class_labels) 

# EfficientNet metrics
correct_predictions_efficientnet = accuracy_calculator(efficientnet_pred, true_labels)
accuracy_3 = correct_predictions_efficientnet/len(true_labels)
precision_3 = precision(efficientnet_pred, true_labels, class_labels)
recall_3 = recall(efficientnet_pred, true_labels, class_labels)
F1_score_3 = f1_score(efficientnet_pred, true_labels, class_labels) 

#%% Log out the performance metrics
txt_lines = [
    "System's Performance of ensemble technique\n",
    f"\t- System's prediction: {correct_predictions}/{len(true_labels)}\n",
    f"\t- System's classification accuracy: {accuracy * 100:.3f}%\n",
    f"\t- System's precision output: {precision_*100:.3f}%\n"
    f"\t- System's recall output: {recall_*100:.3f}%\n"
    f"\t- System's F1-Score output: {F1_score*100:.3f}%\n"
    "######################################################################\n"
    f"\t- Accuracy of Healthy Dataset: {confusion_m[2][2]}/{len(noTumor)} ie {confusion_m[2][2]/len(noTumor)*100:.3f}%\n",
    f"\t- Accuracy of Glioma Dataset: {confusion_m[0][0]}/{len(glioma)} ie {confusion_m[0][0]/len(glioma)*100:.3f}%\n",
    f"\t- Accuracy of Meningioma Dataset: {confusion_m[1][1]}/{len(meningioma)} ie {confusion_m[1][1]/len(meningioma)*100:.3f}%\n",
    f"\t- Accuracy of Pituitary Dataset: {confusion_m[3][3]}/{len(pituitary)} ie {confusion_m[3][3]/len(pituitary)*100:.3f}%\n",
    "######################################################################\n"
    f"\t- ResNet prediction: {correct_predictions_resnet}/{len(true_labels)}\n",
    f"\t- ResNet's classification accuracy: {accuracy_1 * 100:.3f}%\n",
    f"\t- ResNet's precision output: {precision_1*100:.3f}%\n"
    f"\t- ResNet's recall output: {recall_1*100:.3f}%\n"
    f"\t- ResNet's F1-Score output: {F1_score_1*100:.3f}%\n"
    "######################################################################\n"    
    f"\t- DenseNet prediction: {correct_predictions_densenet}/{len(true_labels)}\n",
    f"\t- DenseNet's classification accuracy: {accuracy_2 * 100:.3f}%\n",
    f"\t- DenseNet's precision output: {precision_2*100:.3f}%\n"
    f"\t- DenseNet's recall output: {recall_2*100:.3f}%\n"
    f"\t- DenseNet's F1-Score output: {F1_score_2*100:.3f}%\n"
    "######################################################################\n"    
    f"\t- EfficientNet prediction: {correct_predictions_efficientnet}/{len(true_labels)}\n",
    f"\t- EfficientNet's classification accuracy: {accuracy_3 * 100:.3f}%\n",
    f"\t- EfficientNet's precision output: {precision_3*100:.3f}%\n"
    f"\t- EfficientNet's recall output: {recall_3*100:.3f}%\n"
    f"\t- EfficientNet's F1-Score output: {F1_score_3*100:.3f}%\n"

]
with open("summary.txt", 'w') as f:
    f.writelines(txt_lines)

























