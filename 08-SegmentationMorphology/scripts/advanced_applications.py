"""
Advanced Segmentation and Morphology Applications
Demonstrates real-world applications and combinations of techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, morphology, segmentation, measure, color, feature
from skimage.morphology import disk, remove_small_objects, remove_small_holes
from scipy import ndimage

# Set random seed for reproducibility
np.random.seed(42)


def noise_removal_pipeline():
    """Demonstrate morphological noise removal"""
    # Create noisy binary image
    image = data.horse()

    # Add salt and pepper noise
    noisy = image.copy()
    salt_prob = 0.02
    pepper_prob = 0.02

    # Add salt noise
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy[salt_mask] = 255

    # Add pepper noise
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy[pepper_mask] = 0

    # Threshold
    binary_noisy = noisy > 128

    # Morphological cleaning
    selem = disk(2)
    opened = morphology.binary_opening(binary_noisy, selem)
    closed = morphology.binary_closing(opened, selem)

    # Remove small objects and holes
    cleaned = remove_small_objects(closed, min_size=50)
    final = remove_small_holes(cleaned, area_threshold=50)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Morphological Noise Removal Pipeline', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Clean Image', fontsize=11)
    axes[0, 0].axis('off')

    # Noisy
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Image with Salt & Pepper Noise', fontsize=11)
    axes[0, 1].axis('off')

    # Binary noisy
    axes[0, 2].imshow(binary_noisy, cmap='gray')
    axes[0, 2].set_title('Binary Noisy Image', fontsize=11)
    axes[0, 2].axis('off')

    # After opening
    axes[1, 0].imshow(opened, cmap='gray')
    axes[1, 0].set_title('After Opening (Remove Salt)', fontsize=11)
    axes[1, 0].axis('off')

    # After closing
    axes[1, 1].imshow(closed, cmap='gray')
    axes[1, 1].set_title('After Closing (Remove Pepper)', fontsize=11)
    axes[1, 1].axis('off')

    # Final cleaned
    axes[1, 2].imshow(final, cmap='gray')
    axes[1, 2].set_title('Final Cleaned Result', fontsize=11)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/11_noise_removal_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()


def edge_detection_comparison():
    """Compare morphological edge detection with gradient-based methods"""
    # Load image
    image = data.camera()

    # Morphological edge detection
    selem = disk(2)
    morph_edge = morphology.dilation(image, selem) - morphology.erosion(image, selem)

    # Gradient-based methods
    sobel_edge = filters.sobel(image)
    canny_edge = feature.canny(image, sigma=2)
    prewitt_edge = filters.prewitt(image)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Edge Detection Methods Comparison', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=11)
    axes[0, 0].axis('off')

    # Morphological gradient
    axes[0, 1].imshow(morph_edge, cmap='hot')
    axes[0, 1].set_title('Morphological Gradient', fontsize=11)
    axes[0, 1].axis('off')

    # Sobel
    axes[0, 2].imshow(sobel_edge, cmap='hot')
    axes[0, 2].set_title('Sobel Edge Detector', fontsize=11)
    axes[0, 2].axis('off')

    # Canny
    axes[1, 0].imshow(canny_edge, cmap='gray')
    axes[1, 0].set_title('Canny Edge Detector', fontsize=11)
    axes[1, 0].axis('off')

    # Prewitt
    axes[1, 1].imshow(prewitt_edge, cmap='hot')
    axes[1, 1].set_title('Prewitt Edge Detector', fontsize=11)
    axes[1, 1].axis('off')

    # Binary morphological edge
    binary_morph = morph_edge > filters.threshold_otsu(morph_edge)
    axes[1, 2].imshow(binary_morph, cmap='gray')
    axes[1, 2].set_title('Binary Morphological Edge', fontsize=11)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/12_edge_detection_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def object_extraction():
    """Demonstrate object extraction and analysis"""
    # Load and threshold image
    image = data.coins()
    binary = image > filters.threshold_otsu(image)

    # Clean binary image
    cleaned = morphology.remove_small_objects(binary, min_size=50)
    filled = ndimage.binary_fill_holes(cleaned)

    # Separate touching objects using watershed
    distance = ndimage.distance_transform_edt(filled)
    local_max = morphology.local_maxima(distance)
    markers = measure.label(local_max)
    labels = segmentation.watershed(-distance, markers, mask=filled)

    # Get region properties
    regions = measure.regionprops(labels)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Object Extraction and Analysis', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=11)
    axes[0, 0].axis('off')

    # Binary
    axes[0, 1].imshow(filled, cmap='gray')
    axes[0, 1].set_title('Binary Segmentation', fontsize=11)
    axes[0, 1].axis('off')

    # Distance transform
    axes[0, 2].imshow(distance, cmap='viridis')
    axes[0, 2].set_title('Distance Transform', fontsize=11)
    axes[0, 2].axis('off')

    # Markers
    axes[1, 0].imshow(markers, cmap='nipy_spectral')
    axes[1, 0].set_title(f'Markers ({markers.max()} objects)', fontsize=11)
    axes[1, 0].axis('off')

    # Labeled regions
    axes[1, 1].imshow(color.label2rgb(labels, bg_label=0))
    axes[1, 1].set_title('Segmented Objects', fontsize=11)
    axes[1, 1].axis('off')

    # Object properties
    axes[1, 2].imshow(image, cmap='gray')
    for region in regions:
        y, x = region.centroid
        axes[1, 2].plot(x, y, 'r+', markersize=10)
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                            fill=False, edgecolor='red', linewidth=1.5)
        axes[1, 2].add_patch(rect)
    axes[1, 2].set_title(f'Detected Objects: {len(regions)}', fontsize=11)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/13_object_extraction.png', dpi=300, bbox_inches='tight')
    plt.close()


def texture_enhancement():
    """Demonstrate morphological texture enhancement"""
    # Load image
    image = data.camera()

    # Different sizes for top-hat transform
    selem_small = disk(3)
    selem_medium = disk(7)
    selem_large = disk(15)

    # White top-hat (extracts bright features)
    wth_small = morphology.white_tophat(image, selem_small)
    wth_medium = morphology.white_tophat(image, selem_medium)
    wth_large = morphology.white_tophat(image, selem_large)

    # Black top-hat (extracts dark features)
    bth_medium = morphology.black_tophat(image, selem_medium)

    # Enhanced image
    enhanced = image + wth_medium - bth_medium

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Morphological Texture Enhancement', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=11)
    axes[0, 0].axis('off')

    # White top-hat small
    axes[0, 1].imshow(wth_small, cmap='hot')
    axes[0, 1].set_title('White Top-Hat (Small SE)', fontsize=11)
    axes[0, 1].axis('off')

    # White top-hat medium
    axes[0, 2].imshow(wth_medium, cmap='hot')
    axes[0, 2].set_title('White Top-Hat (Medium SE)', fontsize=11)
    axes[0, 2].axis('off')

    # White top-hat large
    axes[1, 0].imshow(wth_large, cmap='hot')
    axes[1, 0].set_title('White Top-Hat (Large SE)', fontsize=11)
    axes[1, 0].axis('off')

    # Black top-hat
    axes[1, 1].imshow(bth_medium, cmap='hot')
    axes[1, 1].set_title('Black Top-Hat (Medium SE)', fontsize=11)
    axes[1, 1].axis('off')

    # Enhanced result
    axes[1, 2].imshow(enhanced, cmap='gray')
    axes[1, 2].set_title('Enhanced Image\n(Orig + WTH - BTH)', fontsize=11)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/14_texture_enhancement.png', dpi=300, bbox_inches='tight')
    plt.close()


def multi_scale_segmentation():
    """Demonstrate multi-scale segmentation approach"""
    # Load image
    image = data.coffee()
    gray = color.rgb2gray(image)

    # Multi-scale thresholding
    thresh_global = filters.threshold_otsu(gray)
    thresh_local_small = filters.threshold_local(gray, block_size=15)
    thresh_local_large = filters.threshold_local(gray, block_size=51)

    # Apply thresholds
    binary_global = gray > thresh_global
    binary_local_small = gray > thresh_local_small
    binary_local_large = gray > thresh_local_large

    # Combine using morphology
    combined = binary_global & binary_local_large

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Multi-Scale Segmentation', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=11)
    axes[0, 0].axis('off')

    # Grayscale
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale', fontsize=11)
    axes[0, 1].axis('off')

    # Global threshold
    axes[0, 2].imshow(binary_global, cmap='gray')
    axes[0, 2].set_title(f'Global Threshold (Otsu={thresh_global:.2f})', fontsize=11)
    axes[0, 2].axis('off')

    # Local small
    axes[1, 0].imshow(binary_local_small, cmap='gray')
    axes[1, 0].set_title('Local Threshold (Block=15)', fontsize=11)
    axes[1, 0].axis('off')

    # Local large
    axes[1, 1].imshow(binary_local_large, cmap='gray')
    axes[1, 1].set_title('Local Threshold (Block=51)', fontsize=11)
    axes[1, 1].axis('off')

    # Combined
    axes[1, 2].imshow(combined, cmap='gray')
    axes[1, 2].set_title('Combined Result (Global ∧ Local)', fontsize=11)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/15_multi_scale_segmentation.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Generating advanced applications figures...")

    print("  → Noise removal pipeline...")
    noise_removal_pipeline()

    print("  → Edge detection comparison...")
    edge_detection_comparison()

    print("  → Object extraction...")
    object_extraction()

    print("  → Texture enhancement...")
    texture_enhancement()

    print("  → Multi-scale segmentation...")
    multi_scale_segmentation()

    print("✓ Advanced applications figures generated successfully!")
