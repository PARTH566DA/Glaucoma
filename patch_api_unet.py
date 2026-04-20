import re
import os

with open("glaucoma_api.py", "r") as f:
    text = f.read()

# 1. Add PyTorch imports
if "import torch" not in text:
    text = text.replace("import numpy as np", "import numpy as np\nimport torch\nimport torch.nn as nn")

# 2. Add U-Net Classes before GlaucomaPredictor
unet_classes = """
# --- U-Net Architecture for Inference ---
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

class GlaucomaPredictor:"""

if "class UNet" not in text:
    text = text.replace("class GlaucomaPredictor:", unet_classes)

# 3. Add model loading inside __init__
init_old = """        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self._load_model()"""

init_new = """        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self._load_model()
        self._load_unets()"""

if "self._load_unets()" not in text:
    text = text.replace(init_old, init_new)

# 4. Add _load_unets and _get_unet_mask methods
unet_methods = """    def _load_unets(self) -> None:
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.unet_disc = None
        self.unet_cup = None
        
        disc_path = "models/unet_disc.pth"
        cup_path = "models/unet_cup.pth"
        
        if os.path.exists(disc_path) and os.path.exists(cup_path):
            self.unet_disc = UNet().to(self.device)
            self.unet_disc.load_state_dict(torch.load(disc_path, map_location=self.device))
            self.unet_disc.eval()
            
            self.unet_cup = UNet().to(self.device)
            self.unet_cup.load_state_dict(torch.load(cup_path, map_location=self.device))
            self.unet_cup.eval()

    def _get_unet_mask(self, image: np.ndarray, model: nn.Module) -> np.ndarray:
        orig_h, orig_w = image.shape[:2]
        img_resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (256, 256))
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        
        with torch.no_grad():
            pred = model(img_tensor).squeeze().cpu().numpy()
            
        mask = (pred > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask_resized

    def preprocess_image"""

if "def _load_unets" not in text:
    text = text.replace("    def preprocess_image", unet_methods)


# 5. Modify predict_from_bytes to use U-Net
predict_old = """        else:
            # Fallback to OpenCV heuristic detection if masks aren't provided
            disc_mask, disc_bbox = self.detect_optic_disc(image)
            cup_mask, cup_bbox = self.detect_optic_cup(image, disc_mask, disc_bbox)"""

predict_new = """        else:
            # Use PyTorch U-Net for automatic expert segmentation if available
            if self.unet_disc is not None and self.unet_cup is not None:
                disc_mask = self._get_unet_mask(image, self.unet_disc)
                cup_mask = self._get_unet_mask(image, self.unet_cup)
                
                d_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                disc_bbox = cv2.boundingRect(max(d_contours, key=cv2.contourArea)) if d_contours else (0,0,0,0)
                
                c_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cup_bbox = cv2.boundingRect(max(c_contours, key=cv2.contourArea)) if c_contours else (0,0,0,0)
            else:
                # Fallback to OpenCV heuristic detection if masks aren't provided and U-Net isn't found
                disc_mask, disc_bbox = self.detect_optic_disc(image)
                cup_mask, cup_bbox = self.detect_optic_cup(image, disc_mask, disc_bbox)"""

if "self._get_unet_mask(" not in text:
    text = text.replace(predict_old, predict_new)

with open("glaucoma_api.py", "w") as f:
    f.write(text)

print("API successfully patched for U-Net!")