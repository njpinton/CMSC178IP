"""
Core Methods for Basic Image Enhancement

This script demonstrates fundamental image enhancement techniques including:
- Histogram operations and equalization
- Point operations and intensity transforms
- Basic spatial filtering

CMSC 178IP Digital Image Processing
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, exposure, filters
import os

def setup_plotting():
    """Configure matplotlib for consistent styling."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def create_test_image():
    """Create a synthetic test image for demonstrations."""
    h, w = 256, 256
    image = np.zeros((h, w), dtype=np.uint8)

    # Add geometric patterns
    cv2.rectangle(image, (50, 50), (150, 150), 128, -1)
    cv2.circle(image, (200, 200), 40, 200, -1)
    cv2.rectangle(image, (30, 180), (120, 220), 80, -1)

    # Add noise
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

    return image

def demonstrate_histogram_operations():
    """Demonstrate histogram equalization and analysis."""
    setup_plotting()

    # Create test image with low contrast
    original = create_test_image()
    low_contrast = (original * 0.4 + 80).astype(np.uint8)

    # Apply histogram equalization
    equalized = cv2.equalizeHist(low_contrast)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_result = clahe.apply(low_contrast)

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Display images
    images = [original, low_contrast, equalized]
    titles = ['Original', 'Low Contrast', 'Histogram Equalized']

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(title)
        axes[0, i].axis('off')

        # Plot histograms
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        axes[1, i].plot(hist.ravel())
        axes[1, i].set_title(f'Histogram: {title}')
        axes[1, i].set_xlim(0, 255)
        axes[1, i].set_xlabel('Intensity')
        axes[1, i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('../figures/histogram_operations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # CLAHE comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [low_contrast, equalized, clahe_result]
    titles = ['Low Contrast', 'Global Equalization', 'CLAHE']

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/histogram_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def demonstrate_point_operations():
    """Demonstrate point operations and intensity transformations."""
    setup_plotting()

    original = create_test_image()

    # Different transformations
    transformations = [
        (original, "Original"),
        (cv2.addWeighted(original, 1.5, np.zeros_like(original), 0, 0), "Linear (α=1.5)"),
        (np.power(original/255.0, 0.5) * 255, "Gamma (γ=0.5)"),
        (np.power(original/255.0, 2.0) * 255, "Gamma (γ=2.0)"),
        (255 - original, "Negative"),
        (np.where(original > 128, 255, 0), "Threshold"),
    ]

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, (img, title) in enumerate(transformations):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img.astype(np.uint8), cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/point_operations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Gamma correction demonstration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    gamma_values = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    x = np.linspace(0, 255, 256)

    for gamma in gamma_values:
        y = np.power(x/255.0, gamma) * 255
        ax1.plot(x, y, label=f'γ = {gamma}')

    ax1.set_xlabel('Input Intensity')
    ax1.set_ylabel('Output Intensity')
    ax1.set_title('Gamma Correction Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Show gamma effect on image
    gamma_corrected = np.power(original/255.0, 0.6) * 255
    ax2.imshow(gamma_corrected.astype(np.uint8), cmap='gray')
    ax2.set_title('Gamma Corrected (γ=0.6)')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/gamma_correction.png', dpi=300, bbox_inches='tight')
    plt.close()

def demonstrate_spatial_filtering():
    """Demonstrate basic spatial filtering operations."""
    setup_plotting()

    original = create_test_image()

    # Add noise for filtering demonstration
    noisy = original + np.random.normal(0, 20, original.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # Apply different filters
    mean_filtered = cv2.blur(noisy, (5, 5))
    gaussian_filtered = cv2.GaussianBlur(noisy, (5, 5), 1.0)
    median_filtered = cv2.medianBlur(noisy, 5)
    bilateral_filtered = cv2.bilateralFilter(noisy, 9, 75, 75)

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    filters_results = [
        (original, "Original"),
        (noisy, "Noisy"),
        (mean_filtered, "Mean Filter"),
        (gaussian_filtered, "Gaussian Filter"),
        (median_filtered, "Median Filter"),
        (bilateral_filtered, "Bilateral Filter")
    ]

    for i, (img, title) in enumerate(filters_results):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/spatial_filtering.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_core_figures():
    """Generate all core method figures."""
    print("Generating core method figures...")

    # Ensure output directory exists
    os.makedirs('../figures', exist_ok=True)

    # Generate all demonstrations
    demonstrate_histogram_operations()
    demonstrate_point_operations()
    demonstrate_spatial_filtering()

    print("✓ Core method figures generated successfully")

if __name__ == "__main__":
    generate_core_figures()