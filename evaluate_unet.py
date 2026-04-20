# We no longer need to define UNet here because we import it from train_unet_fixed
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from train_unet_fixed import UNet
from torch.utils.data import DataLoader

def dice_coeff(pred, target):
    smooth = 1e-5
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def iou_score(pred, target):
    smooth = 1e-5
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def evaluate(target):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = f"models/unet_{target}.pth"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return
        
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    base_path = "/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135"
    test_raw_dir = os.path.join(base_path, "Test/1.0_Original_Fundus_Images/Remidio")
    sub_folder = "Disc" if target == 'disc' else "Cup"
    test_mask_dir = os.path.join(base_path, f"Test/5.0_OD_OC_Mean_Median_Majority_STAPLE/Remidio/{sub_folder}/STAPLE")
    
    images = []
    if os.path.exists(test_raw_dir):
        for f in os.listdir(test_raw_dir):
            if f.endswith('.JPG') or f.endswith('.jpg'):
                mask_name = f.replace('.JPG', '.png').replace('.jpg', '.png')
                if os.path.exists(os.path.join(test_mask_dir, mask_name)):
                    images.append(f)
                    
    if not images:
        print(f"No test images found for {target}!")
        return
        
    print(f"Evaluating {target.upper()} U-Net on {len(images)} test images...")
    
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        for i, img_name in enumerate(images):
            img_p = os.path.join(test_raw_dir, img_name)
            mask_p = os.path.join(test_mask_dir, img_name.replace('.JPG', '.png').replace('.jpg', '.png'))
            
            img = cv2.resize(cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB), (256, 256))
            mask = cv2.resize(cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE), (256, 256), interpolation=cv2.INTER_NEAREST)
            
            img_tensor = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).to(device)/255.0
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)/255.0
            
            pred = model(img_tensor).cpu()
            pred_bin = (pred > 0.5).float()
            
            total_dice += dice_coeff(pred_bin, mask_tensor).item()
            total_iou += iou_score(pred_bin, mask_tensor).item()
            
    print(f"{target.upper()} U-Net Results:")
    print(f"  -> Average Dice Score: {total_dice/len(images):.4f} (Ideal: > 0.8)")
    print(f"  -> Average IoU Score:  {total_iou/len(images):.4f} (Ideal: > 0.7)\n")

if __name__ == "__main__":
    evaluate('disc')
    evaluate('cup')
