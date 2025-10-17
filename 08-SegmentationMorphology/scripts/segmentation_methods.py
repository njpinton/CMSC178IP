"""
Segmentation Methods Visualization Script
Demonstrates thresholding and region-based segmentation techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, segmentation, measure, morphology
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from scipy import ndimage
import cv2

# Set random seed for reproducibility
np.random.seed(42)

def global_thresholding():
    """Demonstrate global thresholding techniques"""
    # Load and convert image to grayscale
    image = data.camera()

    # Different threshold values
    thresholds = [50, 100, 150, 200]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Global Thresholding', fontsize=16, fontweight='bold')

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')

    # Histogram
    axes[0, 1].hist(image.ravel(), bins=256, color='steelblue', alpha=0.7)
    axes[0, 1].set_title('Histogram', fontsize=12)
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(alpha=0.3)

    # Remove unused subplot
    axes[0, 2].axis('off')

    # Apply different thresholds
    for idx, thresh in enumerate(thresholds):
        binary = image > thresh
        axes[1, idx//2 if idx < 2 else idx-1].imshow(binary, cmap='gray')
        axes[1, idx//2 if idx < 2 else idx-1].set_title(f'Threshold = {thresh}', fontsize=11)
        axes[1, idx//2 if idx < 2 else idx-1].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/01_global_thresholding.png', dpi=300, bbox_inches='tight')
    plt.close()


def otsu_thresholding():
    """Demonstrate Otsu's automatic thresholding method"""
    # Load image
    image = data.coins()

    # Apply Otsu's method
    otsu_thresh = filters.threshold_otsu(image)
    binary_otsu = image > otsu_thresh

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Otsu's Thresholding Method", fontsize=16, fontweight='bold')

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')

    # Histogram with Otsu threshold
    axes[0, 1].hist(image.ravel(), bins=256, color='steelblue', alpha=0.7)
    axes[0, 1].axvline(otsu_thresh, color='red', linestyle='--', linewidth=2, label=f'Otsu Threshold = {otsu_thresh:.1f}')
    axes[0, 1].set_title('Histogram with Otsu Threshold', fontsize=12)
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Binary result
    axes[1, 0].imshow(binary_otsu, cmap='gray')
    axes[1, 0].set_title('Otsu Binary Result', fontsize=12)
    axes[1, 0].axis('off')

    # Comparison with manual threshold
    manual_thresh = 100
    binary_manual = image > manual_thresh
    axes[1, 1].imshow(binary_manual, cmap='gray')
    axes[1, 1].set_title(f'Manual Threshold = {manual_thresh}', fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/02_otsu_thresholding.png', dpi=300, bbox_inches='tight')
    plt.close()


def local_thresholding():
    """Demonstrate local (adaptive) thresholding"""
    # Create image with varying illumination
    image = data.page()

    # Global threshold
    global_thresh = filters.threshold_otsu(image)
    binary_global = image > global_thresh

    # Local thresholding methods
    binary_local = filters.threshold_local(image, block_size=35)
    binary_adaptive = image > binary_local

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Local (Adaptive) Thresholding', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image (Varying Illumination)', fontsize=11)
    axes[0, 0].axis('off')

    # Global threshold
    axes[0, 1].imshow(binary_global, cmap='gray')
    axes[0, 1].set_title(f'Global Threshold (Otsu = {global_thresh:.0f})', fontsize=11)
    axes[0, 1].axis('off')

    # Local threshold map
    axes[1, 0].imshow(binary_local, cmap='viridis')
    axes[1, 0].set_title('Local Threshold Map (Block Size=35)', fontsize=11)
    axes[1, 0].axis('off')

    # Adaptive result
    axes[1, 1].imshow(binary_adaptive, cmap='gray')
    axes[1, 1].set_title('Adaptive Thresholding Result', fontsize=11)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/03_local_thresholding.png', dpi=300, bbox_inches='tight')
    plt.close()


def region_growing():
    """Demonstrate region growing segmentation"""
    # Create simple synthetic image
    image = np.zeros((100, 100))
    image[20:80, 20:80] = 150
    image[40:60, 40:60] = 200

    # Add noise
    noise = np.random.normal(0, 10, image.shape)
    image = image + noise
    image = np.clip(image, 0, 255)

    # Simple region growing implementation
    def simple_region_growing(img, seed, threshold=20):
        segmented = np.zeros_like(img, dtype=bool)
        seeds = [seed]
        seed_value = img[seed]

        while seeds:
            current = seeds.pop(0)
            if segmented[current]:
                continue

            if abs(img[current] - seed_value) <= threshold:
                segmented[current] = True
                x, y = current

                # Check 4-neighbors
                neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                for nx, ny in neighbors:
                    if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                        if not segmented[nx, ny]:
                            seeds.append((nx, ny))

        return segmented

    # Apply region growing from different seeds
    seed1 = (50, 50)  # Center of inner region
    seed2 = (30, 30)  # Outer region

    region1 = simple_region_growing(image, seed1, threshold=25)
    region2 = simple_region_growing(image, seed2, threshold=25)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Region Growing Segmentation', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].plot(seed1[1], seed1[0], 'r*', markersize=15, label='Seed 1')
    axes[0, 0].plot(seed2[1], seed2[0], 'b*', markersize=15, label='Seed 2')
    axes[0, 0].set_title('Original Image with Seeds', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].axis('off')

    # Histogram
    axes[0, 1].hist(image.ravel(), bins=50, color='steelblue', alpha=0.7)
    axes[0, 1].set_title('Intensity Histogram', fontsize=12)
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(alpha=0.3)

    # Region 1 result
    axes[1, 0].imshow(region1, cmap='gray')
    axes[1, 0].set_title(f'Region from Seed 1 (Center)', fontsize=12)
    axes[1, 0].axis('off')

    # Region 2 result
    axes[1, 1].imshow(region2, cmap='gray')
    axes[1, 1].set_title(f'Region from Seed 2 (Outer)', fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/04_region_growing.png', dpi=300, bbox_inches='tight')
    plt.close()


def watershed_segmentation():
    """Demonstrate watershed segmentation"""
    # Load image
    image = data.coins()

    # Preprocessing
    # Apply edge detection
    edges = filters.sobel(image)

    # Marker generation using distance transform
    thresh = image > filters.threshold_otsu(image)
    distance = ndimage.distance_transform_edt(thresh)

    # Find local maxima for markers
    local_max = morphology.local_maxima(distance)
    markers = measure.label(local_max)

    # Apply watershed
    labels = segmentation.watershed(-distance, markers, mask=thresh)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Watershed Segmentation', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=11)
    axes[0, 0].axis('off')

    # Edges
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Edge Detection (Sobel)', fontsize=11)
    axes[0, 1].axis('off')

    # Binary threshold
    axes[0, 2].imshow(thresh, cmap='gray')
    axes[0, 2].set_title('Binary Threshold', fontsize=11)
    axes[0, 2].axis('off')

    # Distance transform
    axes[1, 0].imshow(distance, cmap='viridis')
    axes[1, 0].set_title('Distance Transform', fontsize=11)
    axes[1, 0].axis('off')

    # Markers
    axes[1, 1].imshow(markers, cmap='nipy_spectral')
    axes[1, 1].set_title(f'Markers ({markers.max()} regions)', fontsize=11)
    axes[1, 1].axis('off')

    # Watershed result
    axes[1, 2].imshow(segmentation.mark_boundaries(img_as_ubyte(image/255), labels), cmap='gray')
    axes[1, 2].set_title('Watershed Segmentation', fontsize=11)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/05_watershed_segmentation.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Generating segmentation methods figures...")

    print("  → Global thresholding...")
    global_thresholding()

    print("  → Otsu's thresholding...")
    otsu_thresholding()

    print("  → Local thresholding...")
    local_thresholding()

    print("  → Region growing...")
    region_growing()

    print("  → Watershed segmentation...")
    watershed_segmentation()

    print("✓ Segmentation methods figures generated successfully!")
