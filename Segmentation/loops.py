import torch
from tqdm import tqdm
from loss_functions import DiceLoss, IoULoss

# In[] Example setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dice_list = []
iou_list = []
BCEDiceLoss_list = []

diceLoss = DiceLoss()
iouLoss = IoULoss()

# Training loop
def train_one_epoch(model, data_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    for images, masks in tqdm(data_loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device).float()
        # Forward pass
        predictions = torch.sigmoid(model(images))
        loss = criterion(predictions, masks)
        loss_dice = diceLoss(predictions, masks)
        loss_iou = iouLoss(predictions, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_dice += loss_dice.item()
        running_iou += loss_iou.item()
    # Calculate average loss over epoch
    epoch_loss = running_loss / len(data_loader)
    
    # monitor the losses
    print("******** Training Result **************")    
    print(f'Loss: {running_loss/len(data_loader)}')
    print(f'Loss Dice: {running_dice/len(data_loader)}')
    print(f'Loss IoU: {running_iou/len(data_loader)}')

    return epoch_loss

# In[]  Validation loop
def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            predictions = torch.sigmoid(model(images))
            loss = criterion(predictions, masks)
            loss_dice = diceLoss(predictions, masks)
            loss_iou = iouLoss(predictions, masks)
            
            running_loss += loss.item()
            running_dice += loss_dice.item()
            running_iou += loss_iou.item()

    # Calculate average validation loss over epoch
    epoch_loss = running_loss / len(dataloader)
    # accumulate the losses
    BCEDiceLoss_list.append(running_loss/len(dataloader))
    dice_list.append(running_dice/len(dataloader))
    iou_list.append(running_iou/len(dataloader))

    print("\n******** Validation Result **************")    
    print(f'Loss BCEDiceLoss: {running_loss/len(dataloader)}')
    print(f'Loss Dice: {running_dice/len(dataloader)}')
    print(f'Loss IoU: {running_iou/len(dataloader)}')

    return epoch_loss, BCEDiceLoss_list, dice_list, iou_list

# In[] Main loop
def train_and_validate(model, train_loader, val_loader, optimizer, 
                       criterion, num_epochs, 
                       scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, 
                       save_path="unet.pth"):
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        
        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validation
        val_loss, BCEDiceLoss_list, dice_list, iou_list = validate_one_epoch(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Step the scheduler if validation loss improves
        scheduler.step(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("Model saved!")
        
        print("-" * 30)
        
    return BCEDiceLoss_list, dice_list, iou_list


