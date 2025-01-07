#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 07:48:42 2024
Objective:
    - Setting up early stopping in PyTorch can help prevent overfitting by halting training when the 
        validation loss stops improving after a specified number of epochs (patience).
@author: dagi
"""

import numpy as np

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        # Check if validation loss has improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if improvement
        else:
            self.counter += 1  # Increment counter if no improvement
            if self.counter >= self.patience:
                self.early_stop = True  # Trigger early stopping
                print(f"self.early_stop: {self.early_stop}")

