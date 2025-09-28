"""
Core Noise Reduction Methods Visualization Script

This script generates visualizations for fundamental noise reduction techniques
including linear and non-linear filtering approaches.

Author: CMSC 178IP - Digital Image Processing
Course: Noise Reduction Techniques
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage import data, filters, restoration
from skimage.util import random_noise
from matplotlib.patches import Rectangle
import os

# Set style for consistent plots
plt.style.use('default')

def ensure_output_dir():
    """Ensure the figures directory exists"""
    output_dir = '../figures'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def add_noise_types(image):
    """Add different types of noise to an image"""
    noisy_images = {}

    # Gaussian noise
    noisy_images['gaussian'] = random_noise(image, mode='gaussian', var=0.01)

    # Salt and pepper noise
    noisy_images['salt_pepper'] = random_noise(image, mode='s&p', amount=0.05)

    # Speckle noise
    noisy_images['speckle'] = random_noise(image, mode='speckle', var=0.01)

    # Poisson noise
    noisy_images['poisson'] = random_noise(image, mode='poisson')

    return noisy_images

def create_noise_types_comparison():
    """Create comparison of different noise types"""
    # Load test image
    image = data.camera()
    image = image / 255.0  # Normalize to [0,1]

    # Add different noise types
    noisy_images = add_noise_types(image)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Types of Image Noise', fontsize=16, fontweight='bold')

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')

    # Gaussian noise
    axes[0, 1].imshow(noisy_images['gaussian'], cmap='gray')
    axes[0, 1].set_title('Gaussian Noise\n(σ² = 0.01)', fontweight='bold')
    axes[0, 1].axis('off')

    # Salt and pepper noise
    axes[0, 2].imshow(noisy_images['salt_pepper'], cmap='gray')
    axes[0, 2].set_title('Salt & Pepper Noise\n(5% pixels)', fontweight='bold')
    axes[0, 2].axis('off')

    # Speckle noise
    axes[1, 0].imshow(noisy_images['speckle'], cmap='gray')
    axes[1, 0].set_title('Speckle Noise\n(Multiplicative)', fontweight='bold')
    axes[1, 0].axis('off')

    # Poisson noise
    axes[1, 1].imshow(noisy_images['poisson'], cmap='gray')
    axes[1, 1].set_title('Poisson Noise\n(Shot noise)', fontweight='bold')
    axes[1, 1].axis('off')

    # Hide the last subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(f'{output_dir}/noise_types_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_linear_filters_comparison():
    """Compare different linear filtering approaches"""
    # Create noisy test image
    image = data.camera()
    noisy_image = random_noise(image, mode='gaussian', var=0.01)

    # Apply different linear filters
    # Mean filter
    mean_filtered = ndimage.uniform_filter(noisy_image, size=3)

    # Gaussian filter
    gaussian_filtered = ndimage.gaussian_filter(noisy_image, sigma=1.0)

    # Gaussian filter with larger sigma
    gaussian_large = ndimage.gaussian_filter(noisy_image, sigma=2.0)

    # Box filter (larger kernel)
    box_filtered = ndimage.uniform_filter(noisy_image, size=5)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Linear Noise Reduction Filters', fontsize=16, fontweight='bold')

    # Original noisy image
    axes[0, 0].imshow(noisy_image, cmap='gray')
    axes[0, 0].set_title('Noisy Image\n(Gaussian noise)', fontweight='bold')
    axes[0, 0].axis('off')

    # Mean filter
    axes[0, 1].imshow(mean_filtered, cmap='gray')
    axes[0, 1].set_title('Mean Filter\n(3×3 kernel)', fontweight='bold')
    axes[0, 1].axis('off')

    # Gaussian filter
    axes[0, 2].imshow(gaussian_filtered, cmap='gray')
    axes[0, 2].set_title('Gaussian Filter\n(σ = 1.0)', fontweight='bold')
    axes[0, 2].axis('off')

    # Gaussian filter large
    axes[1, 0].imshow(gaussian_large, cmap='gray')
    axes[1, 0].set_title('Gaussian Filter\n(σ = 2.0)', fontweight='bold')
    axes[1, 0].axis('off')

    # Box filter
    axes[1, 1].imshow(box_filtered, cmap='gray')
    axes[1, 1].set_title('Box Filter\n(5×5 kernel)', fontweight='bold')
    axes[1, 1].axis('off')

    # Hide the last subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(f'{output_dir}/linear_filters_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_nonlinear_filters_comparison():
    """Compare different non-linear filtering approaches"""
    # Create noisy test image with salt & pepper noise
    image = data.camera()
    noisy_image = random_noise(image, mode='s&p', amount=0.05)

    # Apply different non-linear filters
    # Median filter
    median_filtered = ndimage.median_filter(noisy_image, size=3)

    # Median filter with larger kernel
    median_large = ndimage.median_filter(noisy_image, size=5)

    # Maximum filter
    max_filtered = ndimage.maximum_filter(noisy_image, size=3)

    # Minimum filter
    min_filtered = ndimage.minimum_filter(noisy_image, size=3)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Non-Linear Noise Reduction Filters', fontsize=16, fontweight='bold')

    # Original noisy image
    axes[0, 0].imshow(noisy_image, cmap='gray')
    axes[0, 0].set_title('Noisy Image\n(Salt & Pepper)', fontweight='bold')
    axes[0, 0].axis('off')

    # Median filter
    axes[0, 1].imshow(median_filtered, cmap='gray')
    axes[0, 1].set_title('Median Filter\n(3×3 kernel)', fontweight='bold')
    axes[0, 1].axis('off')

    # Median filter large
    axes[0, 2].imshow(median_large, cmap='gray')
    axes[0, 2].set_title('Median Filter\n(5×5 kernel)', fontweight='bold')
    axes[0, 2].axis('off')

    # Maximum filter
    axes[1, 0].imshow(max_filtered, cmap='gray')
    axes[1, 0].set_title('Maximum Filter\n(3×3 kernel)', fontweight='bold')
    axes[1, 0].axis('off')

    # Minimum filter
    axes[1, 1].imshow(min_filtered, cmap='gray')
    axes[1, 1].set_title('Minimum Filter\n(3×3 kernel)', fontweight='bold')
    axes[1, 1].axis('off')

    # Hide the last subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(f'{output_dir}/nonlinear_filters_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_filter_kernels_visualization():
    """Visualize different filter kernels"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Filter Kernels for Noise Reduction', fontsize=16, fontweight='bold')

    # Mean filter kernel (3x3)
    mean_kernel = np.ones((3, 3)) / 9
    im1 = axes[0, 0].imshow(mean_kernel, cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('Mean Filter\n(3×3)', fontweight='bold')
    for i in range(3):
        for j in range(3):
            axes[0, 0].text(j, i, f'{mean_kernel[i, j]:.2f}',
                          ha="center", va="center", color="white", fontsize=10)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # Gaussian kernel approximation (3x3)
    gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    im2 = axes[0, 1].imshow(gaussian_kernel, cmap='viridis', interpolation='nearest')
    axes[0, 1].set_title('Gaussian Filter\n(3×3 approx)', fontweight='bold')
    for i in range(3):
        for j in range(3):
            axes[0, 1].text(j, i, f'{gaussian_kernel[i, j]:.3f}',
                          ha="center", va="center", color="white", fontsize=9)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # Weighted average kernel
    weighted_kernel = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]]) / 10
    im3 = axes[0, 2].imshow(weighted_kernel, cmap='viridis', interpolation='nearest')
    axes[0, 2].set_title('Weighted Average\n(3×3)', fontweight='bold')
    for i in range(3):
        for j in range(3):
            axes[0, 2].text(j, i, f'{weighted_kernel[i, j]:.2f}',
                          ha="center", va="center", color="white", fontsize=10)
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])

    # Larger mean kernel (5x5)
    mean_5x5 = np.ones((5, 5)) / 25
    im4 = axes[0, 3].imshow(mean_5x5, cmap='viridis', interpolation='nearest')
    axes[0, 3].set_title('Mean Filter\n(5×5)', fontweight='bold')
    for i in range(5):
        for j in range(5):
            axes[0, 3].text(j, i, f'{mean_5x5[i, j]:.2f}',
                          ha="center", va="center", color="white", fontsize=8)
    axes[0, 3].set_xticks([])
    axes[0, 3].set_yticks([])

    # Median filter illustration (conceptual)
    median_concept = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    im5 = axes[1, 0].imshow(median_concept, cmap='viridis', interpolation='nearest')
    axes[1, 0].set_title('Median Filter\n(Order statistic)', fontweight='bold')
    axes[1, 0].text(1, 1, 'median', ha="center", va="center", color="white", fontsize=10)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    # Alpha-trimmed mean concept
    alpha_concept = np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])
    im6 = axes[1, 1].imshow(alpha_concept, cmap='viridis', interpolation='nearest')
    axes[1, 1].set_title('Alpha-Trimmed\nMean', fontweight='bold')
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f'{alpha_concept[i, j]:.1f}',
                          ha="center", va="center", color="white", fontsize=10)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    # Bilateral filter concept
    bilateral_concept = np.array([[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]])
    im7 = axes[1, 2].imshow(bilateral_concept, cmap='viridis', interpolation='nearest')
    axes[1, 2].set_title('Bilateral Filter\n(Adaptive)', fontweight='bold')
    for i in range(3):
        for j in range(3):
            axes[1, 2].text(j, i, f'{bilateral_concept[i, j]:.1f}',
                          ha="center", va="center", color="white", fontsize=10)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    # Hide the last subplot
    axes[1, 3].axis('off')

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(f'{output_dir}/filter_kernels_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_noise_model_illustration():
    """Illustrate mathematical noise models"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Mathematical Noise Models', fontsize=16, fontweight='bold')

    # Generate clean signal
    x = np.linspace(0, 4*np.pi, 200)
    clean_signal = np.sin(x) + 0.5*np.sin(3*x)

    # Additive Gaussian noise
    gaussian_noise = np.random.normal(0, 0.2, len(x))
    noisy_additive = clean_signal + gaussian_noise

    axes[0, 0].plot(x, clean_signal, 'b-', linewidth=2, label='Clean Signal')
    axes[0, 0].plot(x, noisy_additive, 'r-', alpha=0.7, label='With Additive Noise')
    axes[0, 0].set_title('Additive Noise Model\ng(x,y) = f(x,y) + n(x,y)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Multiplicative noise
    multiplicative_noise = 1 + np.random.normal(0, 0.1, len(x))
    noisy_multiplicative = clean_signal * multiplicative_noise

    axes[0, 1].plot(x, clean_signal, 'b-', linewidth=2, label='Clean Signal')
    axes[0, 1].plot(x, noisy_multiplicative, 'g-', alpha=0.7, label='With Multiplicative Noise')
    axes[0, 1].set_title('Multiplicative Noise Model\ng(x,y) = f(x,y) × n(x,y)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Impulse noise (salt and pepper)
    impulse_signal = clean_signal.copy()
    # Add salt noise
    salt_locations = np.random.random(len(x)) < 0.02
    impulse_signal[salt_locations] = 2
    # Add pepper noise
    pepper_locations = np.random.random(len(x)) < 0.02
    impulse_signal[pepper_locations] = -2

    axes[1, 0].plot(x, clean_signal, 'b-', linewidth=2, label='Clean Signal')
    axes[1, 0].plot(x, impulse_signal, 'm-', alpha=0.7, label='With Impulse Noise')
    axes[1, 0].set_title('Impulse Noise Model\n(Salt & Pepper)', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Noise histograms
    axes[1, 1].hist(gaussian_noise, bins=30, alpha=0.7, label='Gaussian', density=True)
    uniform_noise = np.random.uniform(-0.3, 0.3, len(x))
    axes[1, 1].hist(uniform_noise, bins=30, alpha=0.7, label='Uniform', density=True)
    axes[1, 1].set_title('Noise Probability Distributions', fontweight='bold')
    axes[1, 1].set_xlabel('Noise Value')
    axes[1, 1].set_ylabel('Probability Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(f'{output_dir}/noise_model_illustration.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all core method visualizations"""
    print("Generating core noise reduction method visualizations...")

    create_noise_types_comparison()
    print("✓ Created noise types comparison")

    create_linear_filters_comparison()
    print("✓ Created linear filters comparison")

    create_nonlinear_filters_comparison()
    print("✓ Created non-linear filters comparison")

    create_filter_kernels_visualization()
    print("✓ Created filter kernels visualization")

    create_noise_model_illustration()
    print("✓ Created noise model illustration")

    print("Core methods visualization complete!")

if __name__ == "__main__":
    main()