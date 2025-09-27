"""
Advanced Techniques for Image Enhancement

This script demonstrates advanced image enhancement methods including:
- Adaptive filtering and edge-preserving techniques
- Advanced edge detection operators
- Multi-scale and frequency domain methods

CMSC 178IP Digital Image Processing
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import filters, feature, restoration
from scipy import ndimage
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

    # Add geometric patterns with edges
    cv2.rectangle(image, (50, 50), (150, 150), 128, -1)
    cv2.circle(image, (200, 200), 40, 200, -1)
    cv2.rectangle(image, (30, 180), (120, 220), 80, -1)

    # Add some texture
    for i in range(10):
        x, y = np.random.randint(0, w, 2)
        cv2.circle(image, (x, y), np.random.randint(5, 15),
                  np.random.randint(50, 200), -1)

    return image

def demonstrate_edge_detection():
    """Demonstrate advanced edge detection operators."""
    setup_plotting()

    original = create_test_image()

    # Apply different edge detection methods
    sobel_x = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    laplacian = cv2.Laplacian(original, cv2.CV_64F)
    canny = cv2.Canny(original, 50, 150)

    # Scharr operator (improved Sobel)
    scharr_x = cv2.Scharr(original, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(original, cv2.CV_64F, 0, 1)
    scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    edge_results = [
        (original, "Original"),
        (np.abs(sobel_magnitude), "Sobel"),
        (np.abs(scharr_magnitude), "Scharr"),
        (np.abs(laplacian), "Laplacian"),
        (canny, "Canny"),
        (feature.canny(original, sigma=1), "Scikit Canny")
    ]

    for i, (img, title) in enumerate(edge_results):
        row = i // 3
        col = i % 3
        if title == "Canny" or title == "Scikit Canny":
            axes[row, col].imshow(img, cmap='gray')
        else:
            axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/edge_detection_advanced.png', dpi=300, bbox_inches='tight')
    plt.close()

def demonstrate_adaptive_filtering():
    """Demonstrate adaptive and edge-preserving filters."""
    setup_plotting()

    original = create_test_image()

    # Add noise
    noisy = original + np.random.normal(0, 25, original.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # Apply adaptive filters
    bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)

    # Non-local means denoising
    nlm = cv2.fastNlMeansDenoising(noisy, None, 10, 7, 21)

    # Simple denoising filter (alternative to Wiener)
    wiener = cv2.medianBlur(noisy, 5)  # Median filter as alternative

    # Gaussian with different sigmas
    gaussian_1 = cv2.GaussianBlur(noisy, (5, 5), 1.0)
    gaussian_2 = cv2.GaussianBlur(noisy, (9, 9), 2.0)

    # Create comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    filter_results = [
        (original, "Original"),
        (noisy, "Noisy"),
        (bilateral, "Bilateral Filter"),
        (nlm, "Non-local Means"),
        (wiener, "Median Filter"),
        (gaussian_2, "Gaussian σ=2.0")
    ]

    for i, (img, title) in enumerate(filter_results):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/adaptive_filtering.png', dpi=300, bbox_inches='tight')
    plt.close()

def demonstrate_unsharp_masking():
    """Demonstrate unsharp masking and sharpening techniques."""
    setup_plotting()

    original = create_test_image()

    # Blur the image slightly
    blurred = cv2.GaussianBlur(original, (5, 5), 2.0)

    # Create unsharp mask
    mask = original.astype(float) - blurred.astype(float)

    # Apply unsharp masking with different strengths
    unsharp_1 = original.astype(float) + 1.0 * mask
    unsharp_2 = original.astype(float) + 2.0 * mask
    unsharp_3 = original.astype(float) + 3.0 * mask

    # Clip values
    unsharp_1 = np.clip(unsharp_1, 0, 255).astype(np.uint8)
    unsharp_2 = np.clip(unsharp_2, 0, 255).astype(np.uint8)
    unsharp_3 = np.clip(unsharp_3, 0, 255).astype(np.uint8)

    # Laplacian sharpening
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    sharpened = blurred.astype(float) - laplacian
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Create comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    sharpening_results = [
        (original, "Original"),
        (blurred, "Blurred"),
        (unsharp_1, "Unsharp k=1.0"),
        (unsharp_2, "Unsharp k=2.0"),
        (unsharp_3, "Unsharp k=3.0"),
        (sharpened, "Laplacian Sharpening")
    ]

    for i, (img, title) in enumerate(sharpening_results):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/unsharp_masking.png', dpi=300, bbox_inches='tight')
    plt.close()

def demonstrate_morphological_operations():
    """Demonstrate morphological operations for enhancement."""
    setup_plotting()

    # Create binary test image
    original = create_test_image()
    binary = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY)[1]

    # Define morphological kernels
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Apply morphological operations
    erosion = cv2.erode(binary, kernel_3, iterations=1)
    dilation = cv2.dilate(binary, kernel_3, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_3)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_3)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel_3)
    tophat = cv2.morphologyEx(original, cv2.MORPH_TOPHAT, kernel_5)

    # Create comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    morph_results = [
        (binary, "Binary Original"),
        (erosion, "Erosion"),
        (dilation, "Dilation"),
        (opening, "Opening"),
        (closing, "Closing"),
        (gradient, "Gradient")
    ]

    for i, (img, title) in enumerate(morph_results):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/morphological_operations.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_advanced_figures():
    """Generate all advanced technique figures."""
    print("Generating advanced technique figures...")

    # Ensure output directory exists
    os.makedirs('../figures', exist_ok=True)

    # Generate all demonstrations
    demonstrate_edge_detection()
    demonstrate_adaptive_filtering()
    demonstrate_unsharp_masking()
    demonstrate_morphological_operations()

    print("✓ Advanced technique figures generated successfully")

if __name__ == "__main__":
    generate_advanced_figures()