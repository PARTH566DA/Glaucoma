import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from train_unet_fixed import UNet

def visualize_cup_disc(image_path, disc_model_path="models/unet_disc.pth", cup_model_path="models/unet_cup.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Models
    try:
        disc_model = UNet().to(device)
        disc_model.load_state_dict(torch.load(disc_model_path, map_location=device))
        disc_model.eval()
        
        cup_model = UNet().to(device)
        cup_model.load_state_dict(torch.load(cup_model_path, map_location=device))
        cup_model.eval()
    except Exception as e:
        print(f"Error loading models: {e}. Make sure you have trained both models completely.")
        return
    
    # Load and preprocess the image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Could not load image at {image_path}")
        return
        
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    
    img_tensor = torch.from_numpy(img_resized).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
    
    # Get Predictions
    with torch.no_grad():
        disc_pred = disc_model(img_tensor).cpu().numpy()[0, 0]
        cup_pred = cup_model(img_tensor).cpu().numpy()[0, 0]
        
    # Convert probabilities to binary masks
    disc_mask = (disc_pred > 0.5).astype(np.uint8) * 255
    cup_mask = (cup_pred > 0.5).astype(np.uint8) * 255
    
    # Resize masks back to original image dimensions for drawing
    h, w = orig_img.shape[:2]
    disc_mask = cv2.resize(disc_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    cup_mask = cv2.resize(cup_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Helper to draw circles around predicted masks
    def draw_enclosing_circle(mask, image, color):
        # Find contours of the prediction
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # We assume the largest contour is our object (Cup or Disc)
            c = max(contours, key=cv2.contourArea)
            
            # Use minEnclosingCircle to get the circle boundary
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Draw the circle highlight
            cv2.circle(image, center, radius, color, 4)
    
    draw_img = orig_img.copy()
    
    # Draw Disc Outline in Green
    draw_enclosing_circle(disc_mask, draw_img, (0, 255, 0)) # Green BGR
    # Draw Cup Outline in Blue
    draw_enclosing_circle(cup_mask, draw_img, (255, 0, 0)) # Blue BGR
    
    # Display Results side by side using Matplotlib
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Uploaded Image")
    plt.imshow(img_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Predictions (Green=Disc, Blue=Cup)")
    plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Highlight Cup and Disc in a Fundus Image")
    parser.add_argument("image_path", type=str, help="Path to the fundus image file")
    args = parser.parse_args()
    
    visualize_cup_disc(args.image_path)
