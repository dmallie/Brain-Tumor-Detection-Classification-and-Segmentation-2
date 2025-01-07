#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:21:51 2024
Objective:
    - This script using the Test data set to evaluates the perfromance of 
        out cascading systems combined with ensemble techniques in classifying Brain MRI as Healthy, Pituitary, Glioma or Meningioma
    - Dataset is composed in the following fashion
        - No_Tumor: Model_1/Test/No_tumor
        - Pituitary: Model_2/Test/Pituitary
        - Glioma : Model_2/Test/Glioma
        - Meningioma: Model_2/Test/Meningioma
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
rootPath = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Final_3/Classification/Cascading_2/"

noTumor_path = rootPath + "Model_1/Test/No_tumor/"
pituitary_path = rootPath + "Model_2/Test/Pituitary/"
glioma_path = rootPath + "Model_2/Test/Glioma/"
meningioma_path = rootPath + "Model_2/Test/Meningioma/"

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
OUTPUT_SHAPE_1 = 1
OUTPUT_SHAPE_2 = 3 
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
densenet_1_path = "/home/dagi/Documents/PyTorch/MIP/Final_3/Classification/Cascading 2/Model 1/densenet_169.pth"
efficientnet_1_path = "/home/dagi/Documents/PyTorch/MIP/Final_3/Classification/Cascading 2/Model 1/EfficientNet.pth"
resnet_1_path = "/home/dagi/Documents/PyTorch/MIP/Final_3/Classification/Cascading 2/Model 1/ResNet_50.pth"

densenet_2_path = "/home/dagi/Documents/PyTorch/MIP/Final_3/Classification/Cascading 2/Model 2/densenet_169.pth"
efficientnet_2_path = "/home/dagi/Documents/PyTorch/MIP/Final_3/Classification/Cascading 2/Model 2/EfficientNet.pth"
resnet_2_path = "/home/dagi/Documents/PyTorch/MIP/Final_3/Classification/Cascading 2/Model 2/ResNet_50.pth"

# %% Load the models
densenet_1 = torch.load(densenet_1_path, weights_only=False)
efficientnet_1 = torch.load(efficientnet_1_path, weights_only=False)
resnet_1 = torch.load(resnet_1_path, weights_only=False)

densenet_2 = torch.load(densenet_2_path, weights_only=False)
efficientnet_2 = torch.load(efficientnet_2_path, weights_only=False)
resnet_2 = torch.load(resnet_2_path, weights_only=False)


#%% move models onto the cuda device
device = "cuda" if torch.cuda.is_available() else "cpu"

densenet_1 = densenet_1.to(device)
efficientnet_1 = efficientnet_1.to(device)
resnet_1 = resnet_1.to(device)

densenet_2 = densenet_2.to(device)
efficientnet_2 = efficientnet_2.to(device)
resnet_2 = resnet_2.to(device)

#%% Set the models in evaluation mode 
densenet_1.eval()
efficientnet_1.eval() 
resnet_1.eval()

densenet_2.eval()
efficientnet_2.eval()
resnet_2.eval()

# %% Set the conatiners that hold the results
true_labels = []
predicted_labels = []

#%% iterate through each image in test_path
for img_path in tqdm(test_path, desc="Test classification models", leave=False):
    # load the image
    img = Image.open(img_path)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # get the image label 
    if "No_tumor" in img_path:
        label = 0
    elif "Glioma" in img_path:
        label = 1
    elif "Meningioma" in img_path:
        label = 2
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
        # Model 1 prediction
        logits_1 = densenet_1(img_transformed)
        probabilities_1 = torch.sigmoid(logits_1)
        
        logits_2 = efficientnet_1(img_transformed)
        probabilities_2 = torch.sigmoid(logits_2)

        logits_3 = resnet_1(img_transformed)
        probabilities_3 = torch.sigmoid(logits_3)
        
        stacked_1 = torch.stack((probabilities_1, probabilities_2, probabilities_3))
        
        # calculate the mean of probability of each models
        average_pool_1 = torch.mean(stacked_1, dim=0)

        # ensemble of the first sets of models
        prediction_1 = (average_pool_1 >= 0.5).long()
        if prediction_1 == 0: # No tumor is detected in the MRI
            predicted_labels.append(0)
            continue
        ##############################################################3
        # Model 2 predicts What type of Tumor it is
        logits_4 = densenet_2(img_transformed)
        probabilities_4 = F.softmax(logits_4, dim=1) # convert predictions to probabilities
        
        logits_5 = efficientnet_2(img_transformed)
        probabilities_5 = F.softmax(logits_5, dim=1) # convert predictions to probabilities
        
        logits_6 = resnet_2(img_transformed)
        probabilities_6 = F.softmax(logits_6, dim=1) # convert predictions to probabilities

        # Stack the probabilities along a new dimension
        stacked = torch.stack((probabilities_4, probabilities_5, probabilities_6)) # shape: [3, 3]

        # compute the mean along dimension 0 or Row
        average_pool_2 = torch.mean(stacked, dim=0)

        prediction_2 = torch.argmax(average_pool_2, dim=1) # index of maximum value along the row
        # print(prediction_2.shape)
        # write the predicted tumor in the predicted_labels container
        predicted_labels.append(prediction_2.item()+1)

#%% Plot Confusion Matrix with Matplotlib
# Define class names
class_names = ['No_tumor', 'Glioma', 'Meningioma', 'Pituitary']
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

#%% To Calculate the efficiency of each of the system 
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
# model_2: classify the 2446 images as 0: glioma = 866 , 1: malignant = 803 and 2: Pituitary = 774
# true_labels_2 is re-written of of those true_labels as 0: no_tumor and different from 0 as 1
 
model_2_output = []
true_labels_2 = []
for index, value in enumerate(predicted_labels):
    if value == 0:
        continue 
    elif value == 1:
        model_2_output.append(0) # it's glioma and it's 0 for model_2
        
    elif value == 2: # it's meningioma
        model_2_output.append(1)
    else:
        model_2_output.append(2)
    true_labels_2.append(true_labels[index]-1)

#%% Evaluate the predicted_labels with the performance measuring metrics
class_labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]

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
class_labels_2 = ["Glioma", "Meningioma", "Pituitary"]
correct_predictions_2 = accuracy_calculator(model_2_output, true_labels_2)
accuracy_2 = correct_predictions_2/len(true_labels_2)
precision_2 = precision(model_2_output, true_labels_2, class_labels_2)
recall_2 = recall(model_2_output, true_labels_2, class_labels_2)
F1_score_2 = f1_score(model_2_output, true_labels_2, class_labels_2) 

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
    f"\t- Accuracy of Glioma Dataset: {confusion_m[1][1]}/{len(glioma)} ie {confusion_m[1][1]/len(glioma)*100:.3f}%\n",
    f"\t- Accuracy of Meningioma Dataset: {confusion_m[2][2]}/{len(meningioma)} ie {confusion_m[2][2]/len(meningioma)*100:.3f}%\n",
    f"\t- Accuracy of Pituitary Dataset: {confusion_m[3][3]}/{len(pituitary)} ie {confusion_m[3][3]/len(pituitary)*100:.3f}%\n",
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

]
with open("summary.txt", 'w') as f:
    f.writelines(txt_lines)








































