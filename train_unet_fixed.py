import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random

# --- Loss Function ---
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

# 1. Model Definition (Reverted to 32 starting channels for speed)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()
        # Back to 32 -> 64 -> 128 -> 256. This is 4x faster to train than 64!
        self.d1 = DoubleConv(in_c, 32); self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(32, 64); self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(64, 128); self.p3 = nn.MaxPool2d(2)
        self.bn = DoubleConv(128, 256)
        
        self.u1 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.c1 = DoubleConv(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.c2 = DoubleConv(128, 64)
        self.u3 = nn.ConvTranspose2d(64, 32, 2, stride=2); self.c3 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, out_c, 1)

    def forward(self, x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        bn = self.bn(p3)
        u1 = self.u1(bn); c1 = self.c1(torch.cat([d3, u1], dim=1))
        u2 = self.u2(c1); c2 = self.c2(torch.cat([d2, u2], dim=1))
        u3 = self.u3(c2); c3 = self.c3(torch.cat([d1, u3], dim=1))
        return torch.sigmoid(self.out(c3))

# 2. Dataset Loader WITH RAM CACHING
class FundusDataset(Dataset):
    def __init__(self, base_dir, target='disc'):
        self.img_size = 256
        self.raw_dir = os.path.join(base_dir, "Train/1.0_Original_Fundus_Images/Remidio")
        sub_folder = "Disc" if target == 'disc' else "Cup"
        self.mask_dir = os.path.join(base_dir, f"Train/5.0_OD_OC_Mean_Median_Majority_STAPLE/Remidio/{sub_folder}/STAPLE")
        
        self.images = []
        for f in os.listdir(self.raw_dir):
            if f.endswith(('.JPG', '.jpg')):
                mask_name = f.replace('.JPG', '.png').replace('.jpg', '.png')
                if os.path.exists(os.path.join(self.mask_dir, mask_name)):
                    self.images.append(f)
                    
        print(f"[{target.upper()}] Found {len(self.images)} image-mask pairs.")
        
        # --- THE CACHE DICTIONARY ---
        self.cache = {}
        
    def __len__(self): return len(self.images)
    
    def __getitem__(self, i):
        # 1. Check if we already processed this image in a previous epoch
        if i in self.cache:
            img_tensor, mask_tensor = self.cache[i]
        else:
            # 2. If not, do the slow hard drive read and OpenCV processing
            img_p = os.path.join(self.raw_dir, self.images[i])
            mask_p = os.path.join(self.mask_dir, self.images[i].replace('.JPG', '.png').replace('.jpg', '.png'))
            
            img = cv2.resize(cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))
            mask = cv2.resize(cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            
            img_tensor = torch.from_numpy(img).float().permute(2,0,1)/255.0
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)/255.0
            
            # Save it to RAM so we NEVER read this file from the disk again
            self.cache[i] = (img_tensor, mask_tensor)

        # 3. Clone the tensors before augmenting so we don't permanently alter the cached originals
        img_out = img_tensor.clone()
        mask_out = mask_tensor.clone()

        # Optional Data Augmentation
        if random.random() > 0.5:
            img_out = TF.hflip(img_out)
            mask_out = TF.hflip(mask_out)
        if random.random() > 0.5:
            img_out = TF.vflip(img_out)
            mask_out = TF.vflip(mask_out)

        return img_out, mask_out

# 3. Training Loop
def train(target, epochs=20):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training {target} on {device}...")
    base_path = "/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135"
    
    dataset = FundusDataset(base_path, target)
    if len(dataset) == 0:
        print(f"Error: No data found for {target}!")
        return
        
    # IMPORTANT MAC FIXES: num_workers=0 avoids Apple Silicon multiprocessing bugs.
    # pin_memory=False is better for Unified Memory chips (M1/M2/M3).
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
    
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    loss_fn = BCEDiceLoss(bce_weight=0.5) 
    
    os.makedirs("models", exist_ok=True)
    for ep in range(epochs):
        model.train()
        losses = []
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(imgs)
            loss = loss_fn(pred, masks)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        epoch_loss = np.mean(losses)
        print(f"Epoch {ep+1}/{epochs} | Loss: {epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(epoch_loss)
        
    save_path = f"models/unet_{target}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}\n")

if __name__ == "__main__":
    # Removed the multiprocessing block as it's no longer needed with num_workers=0
    train('disc', epochs=50) 
    train('cup', epochs=50)