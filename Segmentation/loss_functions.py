import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Flatten tensors to calculate Dice coefficient
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice  # Since we want it as a loss (1 - Dice coefficient)

class IoULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Flatten tensors to calculate IoU
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        total = predictions.sum() + targets.sum()
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou  # Since we want it as a loss (1 - IoU)

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  # BCE with logits for stable computation
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = 0.5 
        self.dice_weight = 0.5

    def forward(self, predictions, targets):
        # Apply BCE Loss
        bce_loss = self.bce(predictions, targets)
        
        # Apply Dice Loss on sigmoid outputs for smooth Dice calculation
        # predictions = torch.sigmoid(predictions)  # Sigmoid for probability outputs in Dice
        dice_loss = self.dice(predictions, targets)
        
        # Combine BCE and Dice Loss
        return bce_loss * self.bce_weight + dice_loss *  self.dice_weight
