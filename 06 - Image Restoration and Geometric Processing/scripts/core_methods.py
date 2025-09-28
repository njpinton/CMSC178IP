#!/usr/bin/env python3
"""
Core Image Restoration and Geometric Processing Methods
CMSC 178IP - Digital Image Processing

This script generates figures demonstrating fundamental image restoration
and geometric processing techniques including noise models, filtering methods,
and basic geometric transformations.

Author: Noel Jeffrey Pinton
Course: CMSC 178IP - Digital Image Processing
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from scipy.signal import wiener
from skimage import data, restoration, transform, util
from skimage.metrics import structural_similarity as ssim
import os

# Set up matplotlib for consistent styling
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Color scheme
PRIMARY_BLUE = '#144B8C'
SECONDARY_BLUE = '#4682B4'
ACCENT_ORANGE = '#E65F2D'

def ensure_output_dir():
    """Ensure the figures directory exists."""
    os.makedirs('../figures', exist_ok=True)

def add_noise_models():
    """Demonstrate different noise models and their characteristics."""
    # Load test image - use astronaut for rich texture and colors
    image = data.astronaut()
    # Convert to grayscale for noise analysis
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create different types of noise
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Noise Models in Image Processing', fontsize=16, fontweight='bold')

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Gaussian noise
    gaussian_noise = util.random_noise(image, mode='gaussian', var=0.01)
    axes[0, 1].imshow(gaussian_noise, cmap='gray')
    axes[0, 1].set_title('Gaussian Noise')
    axes[0, 1].axis('off')

    # Salt and pepper noise
    sp_noise = util.random_noise(image, mode='s&p', amount=0.05)
    axes[0, 2].imshow(sp_noise, cmap='gray')
    axes[0, 2].set_title('Salt & Pepper Noise')
    axes[0, 2].axis('off')

    # Speckle noise
    speckle_noise = util.random_noise(image, mode='speckle', var=0.1)
    axes[0, 3].imshow(speckle_noise, cmap='gray')
    axes[0, 3].set_title('Speckle Noise')
    axes[0, 3].axis('off')

    # Noise histograms
    noise_types = [
        (gaussian_noise - image/255.0, 'Gaussian Noise Distribution'),
        (sp_noise - image/255.0, 'S&P Noise Distribution'),
        (speckle_noise - image/255.0, 'Speckle Noise Distribution')
    ]

    for i, (noise, title) in enumerate(noise_types):
        axes[1, i+1].hist(noise.flatten(), bins=50, alpha=0.7, color=SECONDARY_BLUE, edgecolor='black')
        axes[1, i+1].set_title(title)
        axes[1, i+1].set_xlabel('Noise Value')
        axes[1, i+1].set_ylabel('Frequency')

    # Original histogram
    axes[1, 0].hist(image.flatten(), bins=50, alpha=0.7, color=PRIMARY_BLUE, edgecolor='black')
    axes[1, 0].set_title('Original Image Histogram')
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('../figures/noise_models.png', dpi=300, bbox_inches='tight')
    plt.close()

def spatial_filtering_methods():
    """Demonstrate spatial domain filtering techniques."""
    # Load and add noise to image - use coins for fine detail preservation testing
    image = data.coins()
    noisy = util.random_noise(image, mode='gaussian', var=0.01)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Spatial Domain Filtering Methods', fontsize=16, fontweight='bold')

    # Original and noisy
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Noisy Image')
    axes[0, 1].axis('off')

    # Mean filter
    mean_filtered = ndimage.uniform_filter(noisy, size=3)
    axes[0, 2].imshow(mean_filtered, cmap='gray')
    axes[0, 2].set_title('Mean Filter (3×3)')
    axes[0, 2].axis('off')

    # Gaussian filter
    gaussian_filtered = ndimage.gaussian_filter(noisy, sigma=1.0)
    axes[0, 3].imshow(gaussian_filtered, cmap='gray')
    axes[0, 3].set_title('Gaussian Filter (σ=1.0)')
    axes[0, 3].axis('off')

    # Median filter
    median_filtered = ndimage.median_filter(noisy, size=3)
    axes[1, 0].imshow(median_filtered, cmap='gray')
    axes[1, 0].set_title('Median Filter (3×3)')
    axes[1, 0].axis('off')

    # Bilateral filter (using OpenCV)
    noisy_uint8 = (noisy * 255).astype(np.uint8)
    bilateral_filtered = cv2.bilateralFilter(noisy_uint8, 9, 75, 75)
    axes[1, 1].imshow(bilateral_filtered, cmap='gray')
    axes[1, 1].set_title('Bilateral Filter')
    axes[1, 1].axis('off')

    # Wiener filter
    psf = np.ones((5, 5)) / 25  # Point spread function
    wiener_filtered = restoration.wiener(noisy, psf, 0.1)
    axes[1, 2].imshow(wiener_filtered, cmap='gray')
    axes[1, 2].set_title('Wiener Filter')
    axes[1, 2].axis('off')

    # Non-local means
    nlm_filtered = restoration.denoise_nl_means(noisy, patch_size=7, patch_distance=11, h=0.1)
    axes[1, 3].imshow(nlm_filtered, cmap='gray')
    axes[1, 3].set_title('Non-local Means')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/spatial_filtering.png', dpi=300, bbox_inches='tight')
    plt.close()

def geometric_transformations():
    """Demonstrate basic geometric transformations."""
    # Create a grid pattern with geometric shapes for clear transformation visualization
    def create_transformation_pattern(size=400):
        """Create a pattern with grid lines, circles, and rectangles for transformation demo."""
        pattern = np.ones((size, size)) * 255

        # Add grid lines
        for i in range(0, size, 40):
            pattern[i:i+2, :] = 128  # Horizontal lines
            pattern[:, i:i+2] = 128  # Vertical lines

        # Add some geometric shapes
        # Rectangle
        pattern[100:150, 100:200] = 64

        # Circle
        y, x = np.ogrid[:size, :size]
        circle = (x - 300)**2 + (y - 150)**2 < 40**2
        pattern[circle] = 0

        # Triangle (approximate)
        for i in range(50):
            start = 250 + i
            end = 350 - i
            if start < end and 200 + i < size:
                pattern[200 + i, start:end] = 32

        # Add corner markers
        pattern[10:30, 10:30] = 0    # Top-left
        pattern[10:30, -30:-10] = 0  # Top-right
        pattern[-30:-10, 10:30] = 0  # Bottom-left
        pattern[-30:-10, -30:-10] = 0 # Bottom-right

        return pattern

    def add_transformation_overlay(ax, title, corners=None, center=None, show_grid=True):
        """Add visual overlays to show transformation effects."""
        if show_grid:
            # Add coordinate axes
            ax.axhline(y=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])/2,
                      color=ACCENT_ORANGE, linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(x=ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])/2,
                      color=ACCENT_ORANGE, linestyle='--', alpha=0.7, linewidth=1)

        if corners is not None:
            # Draw corner connections
            for i in range(len(corners)):
                start = corners[i]
                end = corners[(i + 1) % len(corners)]
                ax.plot([start[1], end[1]], [start[0], end[0]],
                       color=PRIMARY_BLUE, linewidth=2, alpha=0.8)

        if center is not None:
            ax.plot(center[1], center[0], 'ro', markersize=8, alpha=0.8)

    image = create_transformation_pattern()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Geometric Transformations with Visual Guides', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Pattern')
    axes[0, 0].axis('off')
    # Add reference frame
    h, w = image.shape
    corners = np.array([[0, 0], [0, w-1], [h-1, w-1], [h-1, 0]])
    add_transformation_overlay(axes[0, 0], 'Original', corners=corners, center=(h//2, w//2))

    # Translation
    tform_translate = transform.SimilarityTransform(translation=(50, 30))
    translated = transform.warp(image, tform_translate)
    axes[0, 1].imshow(translated, cmap='gray')
    axes[0, 1].set_title('Translation (50, 30)')
    axes[0, 1].axis('off')
    # Show translation vector
    axes[0, 1].annotate('', xy=(w//2 + 50, h//2 + 30), xytext=(w//2, h//2),
                       arrowprops=dict(arrowstyle='->', color=ACCENT_ORANGE, lw=3))

    # Rotation
    rotated = transform.rotate(image, angle=30, resize=False)
    axes[0, 2].imshow(rotated, cmap='gray')
    axes[0, 2].set_title('Rotation (30°)')
    axes[0, 2].axis('off')
    # Show rotation center and angle
    add_transformation_overlay(axes[0, 2], 'Rotation', center=(h//2, w//2))

    # Scaling
    scaled = transform.rescale(image, 1.3, anti_aliasing=True)
    axes[1, 0].imshow(scaled, cmap='gray')
    axes[1, 0].set_title('Scaling (1.3×)')
    axes[1, 0].axis('off')

    # Shearing
    tform_shear = transform.AffineTransform(shear=0.3)
    sheared = transform.warp(image, tform_shear)
    axes[1, 1].imshow(sheared, cmap='gray')
    axes[1, 1].set_title('Shearing (0.3)')
    axes[1, 1].axis('off')
    # Show shearing direction
    add_transformation_overlay(axes[1, 1], 'Shearing', show_grid=True)

    # Perspective transformation
    corners = np.array([[0, 0], [0, 200], [200, 0], [200, 200]])
    new_corners = np.array([[20, 10], [10, 180], [180, 20], [190, 190]])
    tform_perspective = transform.ProjectiveTransform()
    tform_perspective.estimate(corners, new_corners)
    perspective = transform.warp(image, tform_perspective, output_shape=(200, 200))
    axes[1, 2].imshow(perspective, cmap='gray')
    axes[1, 2].set_title('Perspective Transform')
    axes[1, 2].axis('off')
    # Show perspective quadrilateral
    add_transformation_overlay(axes[1, 2], 'Perspective', corners=new_corners)

    plt.tight_layout()
    plt.savefig('../figures/geometric_transformations.png', dpi=300, bbox_inches='tight')
    plt.close()

def interpolation_methods():
    """Compare different interpolation methods."""
    # Create a small test pattern
    small_image = np.zeros((20, 20))
    small_image[5:15, 5:15] = 1
    small_image[8:12, 8:12] = 0.5

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Interpolation Methods Comparison', fontsize=16, fontweight='bold')

    # Original small image
    axes[0, 0].imshow(small_image, cmap='gray', interpolation='nearest')
    axes[0, 0].set_title('Original (20×20)')
    axes[0, 0].axis('off')

    # Nearest neighbor
    nearest = transform.resize(small_image, (100, 100), order=0, anti_aliasing=False)
    axes[0, 1].imshow(nearest, cmap='gray')
    axes[0, 1].set_title('Nearest Neighbor')
    axes[0, 1].axis('off')

    # Bilinear
    bilinear = transform.resize(small_image, (100, 100), order=1, anti_aliasing=False)
    axes[0, 2].imshow(bilinear, cmap='gray')
    axes[0, 2].set_title('Bilinear')
    axes[0, 2].axis('off')

    # Bicubic
    bicubic = transform.resize(small_image, (100, 100), order=3, anti_aliasing=False)
    axes[1, 0].imshow(bicubic, cmap='gray')
    axes[1, 0].set_title('Bicubic')
    axes[1, 0].axis('off')

    # With anti-aliasing
    anti_aliased = transform.resize(small_image, (100, 100), order=3, anti_aliasing=True)
    axes[1, 1].imshow(anti_aliased, cmap='gray')
    axes[1, 1].set_title('Bicubic + Anti-aliasing')
    axes[1, 1].axis('off')

    # Profile comparison
    axes[1, 2].plot(nearest[50, :], label='Nearest', linewidth=2, color=PRIMARY_BLUE)
    axes[1, 2].plot(bilinear[50, :], label='Bilinear', linewidth=2, color=SECONDARY_BLUE)
    axes[1, 2].plot(bicubic[50, :], label='Bicubic', linewidth=2, color=ACCENT_ORANGE)
    axes[1, 2].set_title('Cross-section Profile')
    axes[1, 2].set_xlabel('Pixel Position')
    axes[1, 2].set_ylabel('Intensity')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/interpolation_methods.png', dpi=300, bbox_inches='tight')
    plt.close()

def restoration_quality_metrics():
    """Demonstrate quality metrics for image restoration."""
    # Create test image and degraded versions - use cat for detailed analysis
    image = data.cat()
    # Convert to grayscale for quality metrics
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Add different types and levels of degradation
    noise_levels = [0.01, 0.05, 0.1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Image Restoration Quality Metrics', fontsize=16, fontweight='bold')

    # PSNR vs noise level
    psnr_values = []
    ssim_values = []

    for noise_var in noise_levels:
        noisy = util.random_noise(image, mode='gaussian', var=noise_var)

        # Calculate PSNR
        mse = np.mean((image/255.0 - noisy) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        psnr_values.append(psnr)

        # Calculate SSIM
        ssim_val = ssim(image/255.0, noisy, data_range=1)
        ssim_values.append(ssim_val)

    # Plot metrics vs noise level
    axes[0, 0].plot(noise_levels, psnr_values, 'o-', linewidth=2, markersize=8, color=PRIMARY_BLUE)
    axes[0, 0].set_title('PSNR vs Noise Level')
    axes[0, 0].set_xlabel('Noise Variance')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(noise_levels, ssim_values, 'o-', linewidth=2, markersize=8, color=ACCENT_ORANGE)
    axes[0, 1].set_title('SSIM vs Noise Level')
    axes[0, 1].set_xlabel('Noise Variance')
    axes[0, 1].set_ylabel('SSIM Index')
    axes[0, 1].grid(True, alpha=0.3)

    # Filter comparison
    noisy_image = util.random_noise(image, mode='gaussian', var=0.05)

    # Apply different filters
    mean_filtered = ndimage.uniform_filter(noisy_image, size=3)
    gaussian_filtered = ndimage.gaussian_filter(noisy_image, sigma=1.0)
    median_filtered = ndimage.median_filter(noisy_image, size=3)

    filters = [
        ('Mean', mean_filtered),
        ('Gaussian', gaussian_filtered),
        ('Median', median_filtered)
    ]

    filter_psnr = []
    filter_ssim = []
    filter_names = []

    for name, filtered in filters:
        # PSNR
        mse = np.mean((image/255.0 - filtered) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        filter_psnr.append(psnr)

        # SSIM
        ssim_val = ssim(image/255.0, filtered, data_range=1)
        filter_ssim.append(ssim_val)
        filter_names.append(name)

    # Add noisy image metrics
    mse_noisy = np.mean((image/255.0 - noisy_image) ** 2)
    psnr_noisy = 20 * np.log10(1.0 / np.sqrt(mse_noisy))
    ssim_noisy = ssim(image/255.0, noisy_image, data_range=1)

    filter_names.insert(0, 'Noisy')
    filter_psnr.insert(0, psnr_noisy)
    filter_ssim.insert(0, ssim_noisy)

    # Bar plots
    x_pos = np.arange(len(filter_names))
    axes[1, 0].bar(x_pos, filter_psnr, color=[ACCENT_ORANGE if i == 0 else PRIMARY_BLUE for i in range(len(filter_names))])
    axes[1, 0].set_title('Filter Performance - PSNR')
    axes[1, 0].set_xlabel('Filter Type')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(filter_names)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    axes[1, 1].bar(x_pos, filter_ssim, color=[ACCENT_ORANGE if i == 0 else PRIMARY_BLUE for i in range(len(filter_names))])
    axes[1, 1].set_title('Filter Performance - SSIM')
    axes[1, 1].set_xlabel('Filter Type')
    axes[1, 1].set_ylabel('SSIM Index')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(filter_names)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../figures/quality_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all core method figures."""
    print("Generating core image restoration and geometric processing figures...")

    ensure_output_dir()

    # Generate all figures
    add_noise_models()
    print("✓ Generated noise models figure")

    spatial_filtering_methods()
    print("✓ Generated spatial filtering figure")

    geometric_transformations()
    print("✓ Generated geometric transformations figure")

    interpolation_methods()
    print("✓ Generated interpolation methods figure")

    restoration_quality_metrics()
    print("✓ Generated quality metrics figure")

    print("Core methods figures generation complete!")

if __name__ == "__main__":
    main()