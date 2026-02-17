"""
Visualization Script for Chakshu Glaucoma Dataset
Creates visualizations of fundus images with disc and cup overlays.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    overlay = image.copy()
    
    # Create colored masks
    if disc_mask is not None:
        # Green for disc
        disc_colored = np.zeros_like(image)
        disc_colored[:, :, 1] = disc_mask  # Green channel
        overlay = cv2.addWeighted(overlay, 1, disc_colored, alpha, 0)
    
    if cup_mask is not None:
        # Red for cup
        cup_colored = np.zeros_like(image)
        cup_colored[:, :, 0] = cup_mask  # Red channel
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
    overlay = image.copy()
    
    # Find and draw disc contour (green)
    if disc_mask is not None:
        disc_binary = (disc_mask > 128).astype(np.uint8)
        disc_contours, _ = cv2.findContours(disc_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, disc_contours, -1, (0, 255, 0), 3)
    
    # Find and draw cup contour (red)
    if cup_mask is not None:
        cup_binary = (cup_mask > 128).astype(np.uint8)
        cup_contours, _ = cv2.findContours(cup_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cup_contours, -1, (255, 0, 0), 3)
    
    return overlay


def visualize_sample(sample, save_path=None, show=True):
    """Visualize a single sample with multiple views.
    
    Args:
        sample: Sample dictionary from ChakshuDataLoader
        save_path: Path to save the visualization (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{sample['filename']} - {sample['device']} ({sample['split']})", 
                 fontsize=16, fontweight='bold')
    
    image = sample['image']
    disc_mask = sample['disc_mask']
    cup_mask = sample['cup_mask']
    
    # Row 1: Original components
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    if disc_mask is not None:
        axes[0, 1].imshow(disc_mask, cmap='gray')
        axes[0, 1].set_title('Disc Mask')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Disc Mask', ha='center', va='center')
        axes[0, 1].set_title('Disc Mask (Missing)')
    axes[0, 1].axis('off')
    
    if cup_mask is not None:
        axes[0, 2].imshow(cup_mask, cmap='gray')
        axes[0, 2].set_title('Cup Mask')
    else:
        axes[0, 2].text(0.5, 0.5, 'No Cup Mask', ha='center', va='center')
        axes[0, 2].set_title('Cup Mask (Missing)')
    axes[0, 2].axis('off')
    
    # Row 2: Overlays
    if disc_mask is not None and cup_mask is not None:
        # Full mask overlay
        overlay_full = overlay_masks_on_image(image, disc_mask, cup_mask, alpha=0.4)
        axes[1, 0].imshow(overlay_full)
        axes[1, 0].set_title('Overlay (Disc=Green, Cup=Red)')
        axes[1, 0].axis('off')
        
        # Contour overlay
        overlay_contour = create_contour_overlay(image, disc_mask, cup_mask)
        axes[1, 1].imshow(overlay_contour)
        axes[1, 1].set_title('Contour Overlay')
        axes[1, 1].axis('off')
        
        # Statistics
        loader = ChakshuDataLoader()
        cdr = loader.calculate_cdr(disc_mask, cup_mask)
        
        disc_area = np.sum(disc_mask > 0)
        cup_area = np.sum(cup_mask > 0)
        
        stats_text = f"""
        Statistics:
        
        Disc Area: {disc_area:,} pixels
        Cup Area: {cup_area:,} pixels
        
        CDR: {cdr:.4f}
        
        Classification:
        {'✓ Normal (CDR < 0.3)' if cdr < 0.3 else ''}
        {'⚠ Suspicious (0.3 ≤ CDR < 0.6)' if 0.3 <= cdr < 0.6 else ''}
        {'✗ Glaucoma Risk (CDR ≥ 0.6)' if cdr >= 0.6 else ''}
        
        Image Shape: {image.shape}
        """
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                       verticalalignment='center')
        axes[1, 2].axis('off')
    else:
        for ax in axes[1, :]:
            ax.text(0.5, 0.5, 'Masks not available', ha='center', va='center')
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_comparison_grid(samples, save_path=None, show=True):
    """Create a grid comparison of multiple samples.
    
    Args:
        samples: List of sample dictionaries
        save_path: Path to save the visualization
        show: Whether to display the plot
    """
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Sample Comparison', fontsize=16, fontweight='bold')
    
    loader = ChakshuDataLoader()
    
    for i, sample in enumerate(samples):
        image = sample['image']
        disc_mask = sample['disc_mask']
        cup_mask = sample['cup_mask']
        
        # Original
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"{sample['filename']}\n{sample['device']}")
        axes[i, 0].axis('off')
        
        # Disc mask
        if disc_mask is not None:
            axes[i, 1].imshow(disc_mask, cmap='gray')
        axes[i, 1].set_title('Disc')
        axes[i, 1].axis('off')
        
        # Cup mask
        if cup_mask is not None:
            axes[i, 2].imshow(cup_mask, cmap='gray')
        axes[i, 2].set_title('Cup')
        axes[i, 2].axis('off')
        
        # Overlay
        if disc_mask is not None and cup_mask is not None:
            overlay = create_contour_overlay(image, disc_mask, cup_mask)
            axes[i, 3].imshow(overlay)
            cdr = loader.calculate_cdr(disc_mask, cup_mask)
            axes[i, 3].set_title(f'CDR: {cdr:.4f}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison grid to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Main visualization demo."""
    print("=" * 80)
    print("CHAKSHU DATASET VISUALIZATION")
    print("=" * 80)
    
    # Initialize loader
    loader = ChakshuDataLoader()
    
    # Create output directory
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Visualize single sample from each device
    print("\n1. Creating single sample visualizations...")
    devices = ["Bosch", "Forus", "Remidio"]
    
    for device in devices:
        images = loader.get_image_list("Train", device)
        if images:
            sample = loader.load_sample(images[0], "Train", device)
            save_path = os.path.join(output_dir, f"sample_{device}.png")
            visualize_sample(sample, save_path=save_path, show=False)
            print(f"   Created: {save_path}")
    
    # Example 2: Create comparison grid
    print("\n2. Creating comparison grid...")
    samples = []
    for device in devices:
        images = loader.get_image_list("Train", device)
        if images:
            sample = loader.load_sample(images[0], "Train", device)
            samples.append(sample)
    
    if samples:
        save_path = os.path.join(output_dir, "comparison_grid.png")
        create_comparison_grid(samples, save_path=save_path, show=False)
        print(f"   Created: {save_path}")
    
    # Example 3: Show varying CDR values
    print("\n3. Finding samples with varying CDR values...")
    remidio_images = loader.get_image_list("Train", "Remidio")
    
    # Sample a few images and calculate CDR
    cdr_samples = []
    for i in range(0, min(100, len(remidio_images)), 10):
        sample = loader.load_sample(remidio_images[i], "Train", "Remidio")
        if sample['disc_mask'] is not None and sample['cup_mask'] is not None:
            cdr = loader.calculate_cdr(sample['disc_mask'], sample['cup_mask'])
            cdr_samples.append((cdr, sample))
    
    # Sort by CDR
    cdr_samples.sort(key=lambda x: x[0])
    
    if len(cdr_samples) >= 3:
        # Get low, medium, high CDR samples
        selected = [
            cdr_samples[0][1],      # Lowest CDR
            cdr_samples[len(cdr_samples)//2][1],  # Medium CDR
            cdr_samples[-1][1]      # Highest CDR
        ]
        
        save_path = os.path.join(output_dir, "cdr_comparison.png")
        create_comparison_grid(selected, save_path=save_path, show=False)
        print(f"   Created: {save_path}")
        print(f"   CDR range: {cdr_samples[0][0]:.4f} to {cdr_samples[-1][0]:.4f}")
    
    print("\n" + "=" * 80)
    print(f"VISUALIZATION COMPLETE")
    print(f"All visualizations saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
