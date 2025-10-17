"""
Morphological Operations Visualization Script
Demonstrates binary and grayscale morphological operations
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, morphology, filters
from skimage.morphology import disk, square, diamond, rectangle
from scipy import ndimage

# Set random seed for reproducibility
np.random.seed(42)


def binary_erosion_dilation():
    """Demonstrate binary erosion and dilation"""
    # Create binary image
    image = np.zeros((100, 100), dtype=bool)
    image[20:80, 20:80] = True
    image[40:60, 40:60] = False

    # Add some noise
    noise_mask = np.random.random(image.shape) > 0.95
    image[noise_mask] = ~image[noise_mask]

    # Define structuring elements
    selem_small = disk(3)
    selem_large = disk(7)

    # Apply operations
    eroded_small = morphology.binary_erosion(image, selem_small)
    dilated_small = morphology.binary_dilation(image, selem_small)
    eroded_large = morphology.binary_erosion(image, selem_large)
    dilated_large = morphology.binary_dilation(image, selem_large)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Binary Erosion and Dilation', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Binary Image', fontsize=11)
    axes[0, 0].axis('off')

    # Erosion - small SE
    axes[0, 1].imshow(eroded_small, cmap='gray')
    axes[0, 1].set_title('Erosion (Disk r=3)', fontsize=11)
    axes[0, 1].axis('off')

    # Dilation - small SE
    axes[0, 2].imshow(dilated_small, cmap='gray')
    axes[0, 2].set_title('Dilation (Disk r=3)', fontsize=11)
    axes[0, 2].axis('off')

    # Structuring element
    axes[1, 0].imshow(selem_large, cmap='gray')
    axes[1, 0].set_title('Structuring Element (Disk r=7)', fontsize=11)
    axes[1, 0].axis('off')

    # Erosion - large SE
    axes[1, 1].imshow(eroded_large, cmap='gray')
    axes[1, 1].set_title('Erosion (Disk r=7)', fontsize=11)
    axes[1, 1].axis('off')

    # Dilation - large SE
    axes[1, 2].imshow(dilated_large, cmap='gray')
    axes[1, 2].set_title('Dilation (Disk r=7)', fontsize=11)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/06_binary_erosion_dilation.png', dpi=300, bbox_inches='tight')
    plt.close()


def binary_opening_closing():
    """Demonstrate binary opening and closing operations"""
    # Create image with noise
    image = np.zeros((100, 100), dtype=bool)
    image[20:80, 20:80] = True
    image[40:60, 40:60] = False

    # Add salt and pepper noise
    salt = np.random.random(image.shape) > 0.98
    pepper = np.random.random(image.shape) > 0.98
    image[salt & ~image] = True  # Add white noise to black regions
    image[pepper & image] = False  # Add black noise to white regions

    # Structuring element
    selem = disk(3)

    # Apply operations
    opened = morphology.binary_opening(image, selem)
    closed = morphology.binary_closing(image, selem)
    opened_closed = morphology.binary_closing(morphology.binary_opening(image, selem), selem)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Binary Opening and Closing', fontsize=16, fontweight='bold')

    # Original noisy image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original (with Salt & Pepper Noise)', fontsize=11)
    axes[0, 0].axis('off')

    # Opening (removes small white regions)
    axes[0, 1].imshow(opened, cmap='gray')
    axes[0, 1].set_title('Opening (Erosion → Dilation)', fontsize=11)
    axes[0, 1].axis('off')

    # Closing (removes small black regions)
    axes[1, 0].imshow(closed, cmap='gray')
    axes[1, 0].set_title('Closing (Dilation → Erosion)', fontsize=11)
    axes[1, 0].axis('off')

    # Opening then closing
    axes[1, 1].imshow(opened_closed, cmap='gray')
    axes[1, 1].set_title('Opening → Closing (Noise Removal)', fontsize=11)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/07_binary_opening_closing.png', dpi=300, bbox_inches='tight')
    plt.close()


def structuring_elements():
    """Demonstrate different structuring elements"""
    # Create test image
    image = np.zeros((80, 80), dtype=bool)
    image[30:50, 30:50] = True

    # Different structuring elements
    se_disk = disk(5)
    se_square = square(11)
    se_diamond = diamond(5)
    se_rect = rectangle(11, 5)

    # Apply dilation with each SE
    result_disk = morphology.binary_dilation(image, se_disk)
    result_square = morphology.binary_dilation(image, se_square)
    result_diamond = morphology.binary_dilation(image, se_diamond)
    result_rect = morphology.binary_dilation(image, se_rect)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Structuring Elements and Their Effects', fontsize=16, fontweight='bold')

    # Original
    axes[0, 1].imshow(image, cmap='gray')
    axes[0, 1].set_title('Original Image', fontsize=11)
    axes[0, 1].axis('off')
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')

    # Disk SE and result
    axes[1, 0].imshow(se_disk, cmap='gray')
    axes[1, 0].set_title('Disk (r=5)', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(result_disk, cmap='gray')
    axes[1, 1].set_title('Dilation with Disk', fontsize=11)
    axes[1, 1].axis('off')

    # Square SE and result
    axes[1, 2].imshow(se_square, cmap='gray')
    axes[1, 2].set_title('Square (11×11)', fontsize=11)
    axes[1, 2].axis('off')

    axes[2, 0].imshow(result_square, cmap='gray')
    axes[2, 0].set_title('Dilation with Square', fontsize=11)
    axes[2, 0].axis('off')

    # Diamond SE and result
    axes[2, 1].imshow(se_diamond, cmap='gray')
    axes[2, 1].set_title('Diamond (r=5)', fontsize=11)
    axes[2, 1].axis('off')

    axes[2, 2].imshow(result_diamond, cmap='gray')
    axes[2, 2].set_title('Dilation with Diamond', fontsize=11)
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/08_structuring_elements.png', dpi=300, bbox_inches='tight')
    plt.close()


def morphological_gradient():
    """Demonstrate morphological gradient"""
    # Load image
    image = data.camera()

    # Binary image
    binary = image > filters.threshold_otsu(image)

    # Structuring element
    selem = disk(3)

    # Morphological operations
    eroded = morphology.erosion(image, selem)
    dilated = morphology.dilation(image, selem)

    # Gradients
    gradient = dilated - eroded  # External - Internal
    internal_gradient = image - eroded
    external_gradient = dilated - image

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Morphological Gradient', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=11)
    axes[0, 0].axis('off')

    # Dilated
    axes[0, 1].imshow(dilated, cmap='gray')
    axes[0, 1].set_title('Dilation', fontsize=11)
    axes[0, 1].axis('off')

    # Eroded
    axes[0, 2].imshow(eroded, cmap='gray')
    axes[0, 2].set_title('Erosion', fontsize=11)
    axes[0, 2].axis('off')

    # Morphological gradient
    axes[1, 0].imshow(gradient, cmap='hot')
    axes[1, 0].set_title('Morphological Gradient\n(Dilation - Erosion)', fontsize=11)
    axes[1, 0].axis('off')

    # Internal gradient
    axes[1, 1].imshow(internal_gradient, cmap='hot')
    axes[1, 1].set_title('Internal Gradient\n(Original - Erosion)', fontsize=11)
    axes[1, 1].axis('off')

    # External gradient
    axes[1, 2].imshow(external_gradient, cmap='hot')
    axes[1, 2].set_title('External Gradient\n(Dilation - Original)', fontsize=11)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/09_morphological_gradient.png', dpi=300, bbox_inches='tight')
    plt.close()


def grayscale_morphology():
    """Demonstrate grayscale morphological operations"""
    # Load image
    image = data.camera()

    # Structuring element
    selem = disk(5)

    # Grayscale operations
    eroded = morphology.erosion(image, selem)
    dilated = morphology.dilation(image, selem)
    opened = morphology.opening(image, selem)
    closed = morphology.closing(image, selem)

    # Top-hat and black-hat transforms
    tophat = morphology.white_tophat(image, selem)
    blackhat = morphology.black_tophat(image, selem)

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    fig.suptitle('Grayscale Morphological Operations', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=11)
    axes[0, 0].axis('off')

    # Erosion
    axes[0, 1].imshow(eroded, cmap='gray')
    axes[0, 1].set_title('Erosion (Darkens)', fontsize=11)
    axes[0, 1].axis('off')

    # Dilation
    axes[0, 2].imshow(dilated, cmap='gray')
    axes[0, 2].set_title('Dilation (Brightens)', fontsize=11)
    axes[0, 2].axis('off')

    # Opening
    axes[1, 0].imshow(opened, cmap='gray')
    axes[1, 0].set_title('Opening (Removes Light Details)', fontsize=11)
    axes[1, 0].axis('off')

    # Closing
    axes[1, 1].imshow(closed, cmap='gray')
    axes[1, 1].set_title('Closing (Removes Dark Details)', fontsize=11)
    axes[1, 1].axis('off')

    # Structuring element
    axes[1, 2].imshow(selem, cmap='gray')
    axes[1, 2].set_title('Structuring Element (Disk r=5)', fontsize=11)
    axes[1, 2].axis('off')

    # Top-hat
    axes[2, 0].imshow(tophat, cmap='hot')
    axes[2, 0].set_title('White Top-Hat\n(Original - Opening)', fontsize=11)
    axes[2, 0].axis('off')

    # Black-hat
    axes[2, 1].imshow(blackhat, cmap='hot')
    axes[2, 1].set_title('Black Top-Hat\n(Closing - Original)', fontsize=11)
    axes[2, 1].axis('off')

    # Enhanced using top-hat
    enhanced = image + tophat - blackhat
    axes[2, 2].imshow(enhanced, cmap='gray')
    axes[2, 2].set_title('Enhanced Image\n(Orig + White TH - Black TH)', fontsize=11)
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/10_grayscale_morphology.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Generating morphological operations figures...")

    print("  → Binary erosion and dilation...")
    binary_erosion_dilation()

    print("  → Binary opening and closing...")
    binary_opening_closing()

    print("  → Structuring elements...")
    structuring_elements()

    print("  → Morphological gradient...")
    morphological_gradient()

    print("  → Grayscale morphology...")
    grayscale_morphology()

    print("✓ Morphological operations figures generated successfully!")
