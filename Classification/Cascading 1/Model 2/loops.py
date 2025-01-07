#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:45:45 2024
Objective:
    - Setup Training Loop : training_loop
    - Setup Validation Loop : val_loop
    - Setup the Loop which combines them : combo_loop
@author: dagi
"""
import torch
from tqdm import tqdm
import torch.nn.functional as F
from early_stoppage import EarlyStopping
from utils import accuracy_calculator

# In[] Set Hyperparameters and initialize early stopping
device = "cuda" if torch.cuda.is_available() else "cpu"

earlyStopping = EarlyStopping(patience=15, min_delta=0.01)

# In[] Training loop
def training_loop(model, dataloader, optimizer, criterion):

    # To accumulate the model's output
    actual_labels = []
    predicted_labels = []
    # Reset parameters to calculate the loss & accuracy
    running_loss = 0
    # 1. Activate the training mode
    model.train()
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # 2. Forward pass
        predictions = model(images)
        labels  = labels.float().to(device)
        # print(f"predictions.shape: {predictions.shape}\t labels.shape: {labels.shape}")
        loss = criterion(predictions.squeeze(dim=1), labels)
        # print(f"loss: {loss}")
        
        # 3. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # 4. Update the weights
        optimizer.step()
        
        # update the loss
        running_loss += loss.item()
        
        # extract predicted class indices   
        
        output = (torch.sigmoid(predictions) >= 0.5).long() 
        predicted_labels.append(output.cpu())
        actual_labels.append(labels.cpu())
        
    # Collect the output of an epoch
    actual_labels = torch.cat(actual_labels)
    predicted_labels = torch.cat(predicted_labels)
    
    # Calculate the average loss and accuracy
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy =  accuracy_calculator(predicted_labels, actual_labels)
    print(f"correct_predictions: {epoch_accuracy}\{len(actual_labels)}: {epoch_accuracy/len(actual_labels):.3}%")
    return epoch_loss, epoch_accuracy/len(actual_labels)

# In[] Validation Loop
def validation_loop(model, dataloader, criterion):
    running_loss = 0.0
    actual_labels = []
    predicted_labels = []
    # Set model's mode to evaluation
    model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            # Move the data to the cuda device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            predictions = model(images)
            # Calculate the loss
            loss = criterion(predictions.squeeze(dim=1), labels.float())
            
            # convert predictions to probabilities
            output = (torch.sigmoid(predictions) >= 0.5).long() 
            running_loss += loss.item()
            
            actual_labels.append(labels.cpu())
            predicted_labels.append(output.cpu())
            
    # Collect the output of an epoch
    actual_labels = torch.cat(actual_labels)
    predicted_labels = torch.cat(predicted_labels)
    
    # Calculate the average loss and accuracy
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = accuracy_calculator(predicted_labels, actual_labels)
    return epoch_loss, epoch_accuracy/len(actual_labels)
    
# In[] Main loop
def main_loop(model, train_dataloader, val_dataloader,
              optimizer, criterion, epochs, scheduler,save_path):
    best_val_loss = float("inf")
    accuracy_list_training  = []
    accuracy_list_val = []
    loss_list_training = []
    loss_list_val = []
    
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        
        # Training
        train_loss, train_accuracy = training_loop(model, train_dataloader, optimizer, criterion)
        print(f"Training Loss: {train_loss:.4f}\t Training Accuracy: {train_accuracy:.2f}")
        accuracy_list_training.append(train_accuracy)
        loss_list_training.append(train_loss)
        # Validation
        val_loss, val_accuracy = validation_loop(model, val_dataloader, criterion)
        print(f"Validation Loss: {val_loss:.4f}\t Validation Accuracy: {val_accuracy:.2f}")
        accuracy_list_val.append(val_accuracy)
        loss_list_val.append(val_loss)
            
        # Step the scheduler if validation loss improves
        scheduler.step(val_loss)

        # Early stopping
        earlyStopping(val_loss)
        if earlyStopping.early_stop:
            print("Early stopping")
            break

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, save_path)
            print("Model saved!")
        
        print("-" * 30)
    return accuracy_list_training, accuracy_list_val, loss_list_training, loss_list_val
