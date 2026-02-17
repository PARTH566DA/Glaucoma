"""
Visualization Script for Chakshu Glaucoma Dataset (OpenCV-only version)
Creates visualizations of fundus images with disc and cup overlays using only OpenCV.
"""

import cv2
import numpy as np
from data_loader import ChakshuDataLoader
import os


def overlay_masks_on_image(image, disc_mask, cup_mask, alpha=0.4):
    """Create an overlay visualization with disc in green and cup in red.
    
    Args:
        image: RGB fundus image
        disc_mask: Binary disc mask
        cup_mask: Binary cup mask
        alpha: Transparency for overlays (0-1)
        
    Returns:
        Image with overlays
    """
    # Convert RGB to BGR for OpenCV
    overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image.copy()
    
    # Create colored masks
    if disc_mask is not None:
        # Green for disc
        disc_colored = np.zeros_like(overlay)
        disc_colored[:, :, 1] = disc_mask  # Green channel
        overlay = cv2.addWeighted(overlay, 1, disc_colored, alpha, 0)
    
    if cup_mask is not None:
        # Red for cup
        cup_colored = np.zeros_like(overlay)
        cup_colored[:, :, 2] = cup_mask  # Red channel (BGR format)
        overlay = cv2.addWeighted(overlay, 1, cup_colored, alpha, 0)
    
    return overlay


def create_contour_overlay(image, disc_mask, cup_mask):
    """Create an overlay with contours only.
    
    Args:
        image: RGB fundus image
        disc_mask: Binary disc mask
        cup_mask: Binary cup mask
        
    Returns:
        Image with contour overlays
    """
    # Convert RGB to BGR for OpenCV
    overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image.copy()
    
    # Find and draw disc contour (green)
    if disc_mask is not None:
        disc_binary = (disc_mask > 128).astype(np.uint8)
        disc_contours, _ = cv2.findContours(disc_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, disc_contours, -1, (0, 255, 0), 3)
    
    # Find and draw cup contour (red)
    if cup_mask is not None:
        cup_binary = (cup_mask > 128).astype(np.uint8)
        cup_contours, _ = cv2.findContours(cup_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cup_contours, -1, (0, 0, 255), 3)
    
    return overlay


def add_text_to_image(image, text_lines, position=(10, 30), font_scale=0.6, color=(255, 255, 255)):
    """Add multiple lines of text to an image.
    
    Args:
        image: Input image
        text_lines: List of text strings
        position: Starting position (x, y)
        font_scale: Font size scale
        color: Text color (BGR)
        
    Returns:
        Image with text
    """
    result = image.copy()
    x, y = position
    line_height = int(30 * font_scale)
    
    for line in text_lines:
        cv2.putText(result, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, 2, cv2.LINE_AA)
        y += line_height
    
    return result


def create_visualization_grid(sample, save_path=None):
    """Create a grid visualization of a sample.
    
    Args:
        sample: Sample dictionary from ChakshuDataLoader
        save_path: Path to save the visualization (optional)
        
    Returns:
        Combined visualization image
    """
    image = sample['image']
    disc_mask = sample['disc_mask']
    cup_mask = sample['cup_mask']
    
    # Resize images for display (max 800 pixels width)
    h, w = image.shape[:2]
    if w > 800:
        scale = 800 / w
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        if disc_mask is not None:
            disc_mask = cv2.resize(disc_mask, (new_w, new_h))
        if cup_mask is not None:
            cup_mask = cv2.resize(cup_mask, (new_w, new_h))
    
    h, w = image.shape[:2]
    
    # Convert to BGR for display
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create individual panels
    panels = []
    
    # Panel 1: Original image
    panel1 = image_bgr.copy()
    panel1 = add_text_to_image(panel1, ["Original Image"], (10, 30))
    panels.append(panel1)
    
    # Panel 2: Disc mask
    if disc_mask is not None:
        panel2 = cv2.cvtColor(disc_mask, cv2.COLOR_GRAY2BGR)
        panel2 = add_text_to_image(panel2, ["Disc Mask"], (10, 30))
    else:
        panel2 = np.zeros((h, w, 3), dtype=np.uint8)
        panel2 = add_text_to_image(panel2, ["No Disc Mask"], (10, 30))
    panels.append(panel2)
    
    # Panel 3: Cup mask
    if cup_mask is not None:
        panel3 = cv2.cvtColor(cup_mask, cv2.COLOR_GRAY2BGR)
        panel3 = add_text_to_image(panel3, ["Cup Mask"], (10, 30))
    else:
        panel3 = np.zeros((h, w, 3), dtype=np.uint8)
        panel3 = add_text_to_image(panel3, ["No Cup Mask"], (10, 30))
    panels.append(panel3)
    
    # Panel 4: Overlay
    if disc_mask is not None and cup_mask is not None:
        panel4 = overlay_masks_on_image(image, disc_mask, cup_mask, alpha=0.4)
        panel4 = add_text_to_image(panel4, ["Overlay (Green=Disc, Red=Cup)"], (10, 30))
    else:
        panel4 = np.zeros((h, w, 3), dtype=np.uint8)
        panel4 = add_text_to_image(panel4, ["Masks not available"], (10, 30))
    panels.append(panel4)
    
    # Panel 5: Contours
    if disc_mask is not None and cup_mask is not None:
        panel5 = create_contour_overlay(image, disc_mask, cup_mask)
        panel5 = add_text_to_image(panel5, ["Contours"], (10, 30))
    else:
        panel5 = np.zeros((h, w, 3), dtype=np.uint8)
        panel5 = add_text_to_image(panel5, ["Masks not available"], (10, 30))
    panels.append(panel5)
    
    # Panel 6: Statistics
    panel6 = np.zeros((h, w, 3), dtype=np.uint8)
    if disc_mask is not None and cup_mask is not None:
        loader = ChakshuDataLoader()
        cdr = loader.calculate_cdr(disc_mask, cup_mask)
        disc_area = np.sum(disc_mask > 0)
        cup_area = np.sum(cup_mask > 0)
        
        status = "Normal" if cdr < 0.3 else ("Suspicious" if cdr < 0.6 else "Glaucoma Risk")
        
        text_lines = [
            f"File: {sample['filename'][:20]}",
            f"Device: {sample['device']}",
            f"Split: {sample['split']}",
            "",
            f"Disc Area: {disc_area}",
            f"Cup Area: {cup_area}",
            "",
            f"CDR: {cdr:.4f}",
            f"Status: {status}",
            "",
            f"Image: {image.shape[1]}x{image.shape[0]}"
        ]
        panel6 = add_text_to_image(panel6, text_lines, (10, 40), font_scale=0.5)
    else:
        panel6 = add_text_to_image(panel6, ["No statistics", "Masks not available"], (10, 40))
    panels.append(panel6)
    
    # Create grid (2 rows x 3 columns)
    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:])
    grid = np.vstack([row1, row2])
    
    # Add title at the top
    title_height = 60
    title_panel = np.zeros((title_height, grid.shape[1], 3), dtype=np.uint8)
    title_text = f"{sample['filename']} - {sample['device']} ({sample['split']})"
    cv2.putText(title_panel, title_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
               1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    final_grid = np.vstack([title_panel, grid])
    
    if save_path:
        cv2.imwrite(save_path, final_grid)
        print(f"Saved: {save_path}")
    
    return final_grid


def main():
    """Main visualization demo."""
    print("=" * 80)
    print("CHAKSHU DATASET VISUALIZATION (OpenCV)")
    print("=" * 80)
    
    # Initialize loader
    loader = ChakshuDataLoader()
    
    # Create output directory
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations for each device
    print("\n1. Creating visualizations for each device...")
    devices = ["Bosch", "Forus", "Remidio"]
    
    for device in devices:
        images = loader.get_image_list("Train", device)
        if images:
            sample = loader.load_sample(images[0], "Train", device)
            save_path = os.path.join(output_dir, f"sample_{device}.jpg")
            create_visualization_grid(sample, save_path=save_path)
    
    # Find samples with varying CDR values
    print("\n2. Finding samples with varying CDR values...")
    remidio_images = loader.get_image_list("Train", "Remidio")
    
    cdr_samples = []
    for i in range(0, min(100, len(remidio_images)), 5):
        sample = loader.load_sample(remidio_images[i], "Train", "Remidio")
        if sample['disc_mask'] is not None and sample['cup_mask'] is not None:
            cdr = loader.calculate_cdr(sample['disc_mask'], sample['cup_mask'])
            cdr_samples.append((cdr, sample))
    
    if cdr_samples:
        # Sort by CDR
        cdr_samples.sort(key=lambda x: x[0])
        
        # Save low, medium, and high CDR examples
        categories = [
            ("low_cdr", cdr_samples[0]),
            ("medium_cdr", cdr_samples[len(cdr_samples)//2]),
            ("high_cdr", cdr_samples[-1])
        ]
        
        for name, (cdr, sample) in categories:
            save_path = os.path.join(output_dir, f"{name}_{cdr:.4f}.jpg")
            create_visualization_grid(sample, save_path=save_path)
            print(f"   {name.upper()}: CDR = {cdr:.4f}")
    
    print("\n" + "=" * 80)
    print(f"VISUALIZATION COMPLETE")
    print(f"All visualizations saved to: {output_dir}/")
    print("=" * 80)
    print("\nYou can view the images using:")
    print(f"  open {output_dir}/")
    print("  or navigate to the folder in Finder")


if __name__ == "__main__":
    main()
