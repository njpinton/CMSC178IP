"""
Real-World Examples for Image Enhancement

This script demonstrates practical applications of image enhancement using:
- Real image datasets and examples
- Medical imaging scenarios
- Photography and artistic enhancement
- Industrial and surveillance applications

CMSC 178IP Digital Image Processing
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data, exposure, filters, color
import os

def setup_plotting():
    """Configure matplotlib for consistent styling."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def load_sample_images():
    """Load sample images from scikit-image datasets."""
    images = {
        'camera': data.camera(),
        'coins': data.coins(),
        'checkerboard': data.checkerboard(),
        'moon': data.moon(),
        'astronaut': data.astronaut()
    }
    return images

def demonstrate_medical_imaging_enhancement():
    """Demonstrate enhancement techniques for medical imaging."""
    setup_plotting()

    # Use X-ray like image (coins dataset simulates this well)
    xray = data.coins()

    # Simulate low contrast medical image
    low_contrast = (xray * 0.6 + 50).astype(np.uint8)

    # Apply medical imaging enhancements
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_enhanced = clahe.apply(low_contrast)

    # Histogram equalization
    hist_eq = cv2.equalizeHist(low_contrast)

    # Gamma correction for better visibility
    gamma_corrected = exposure.adjust_gamma(low_contrast, gamma=0.7)

    # Contrast stretching
    p2, p98 = np.percentile(low_contrast, (2, 98))
    contrast_stretched = exposure.rescale_intensity(low_contrast, in_range=(p2, p98))

    # Create medical imaging comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    medical_results = [
        (xray, "Original X-ray"),
        (low_contrast, "Low Contrast"),
        (hist_eq, "Histogram Equalization"),
        (clahe_enhanced, "CLAHE Enhanced"),
        (gamma_corrected, "Gamma Corrected"),
        (contrast_stretched, "Contrast Stretched")
    ]

    for i, (img, title) in enumerate(medical_results):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/medical_imaging_enhancement.png', dpi=300, bbox_inches='tight')
    plt.close()

def demonstrate_photography_enhancement():
    """Demonstrate enhancement for photography applications."""
    setup_plotting()

    # Use astronaut image for color enhancement
    photo = data.astronaut()
    # Ensure image is in float format between 0 and 1
    if photo.dtype == np.uint8:
        photo = photo.astype(float) / 255.0

    # Convert to different color spaces for enhancement
    hsv = color.rgb2hsv(photo)
    lab = color.rgb2lab(photo)

    # Enhance saturation in HSV
    hsv_enhanced = hsv.copy()
    hsv_enhanced[:, :, 1] = np.clip(hsv_enhanced[:, :, 1] * 1.4, 0, 1)
    saturation_enhanced = color.hsv2rgb(hsv_enhanced)

    # Enhance contrast in LAB space
    lab_enhanced = lab.copy()
    # Normalize L channel to 0-1 range for equalize_adapthist
    l_channel = lab_enhanced[:, :, 0] / 100.0  # LAB L is typically 0-100
    l_channel = np.clip(l_channel, 0, 1)
    lab_enhanced[:, :, 0] = exposure.equalize_adapthist(l_channel) * 100.0
    contrast_enhanced = color.lab2rgb(lab_enhanced)

    # Apply gamma correction
    gamma_enhanced = exposure.adjust_gamma(photo, gamma=0.8)

    # Apply unsharp masking for sharpening
    blurred = filters.gaussian(photo, sigma=1, channel_axis=-1)
    mask = photo - blurred
    sharpened = photo + 0.5 * mask
    sharpened = np.clip(sharpened, 0, 1)

    # Create photography enhancement comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    photo_results = [
        (photo, "Original"),
        (saturation_enhanced, "Enhanced Saturation"),
        (contrast_enhanced, "Enhanced Contrast"),
        (gamma_enhanced, "Gamma Corrected"),
        (sharpened, "Sharpened"),
        (exposure.equalize_adapthist(photo), "Adaptive Equalization")
    ]

    for i, (img, title) in enumerate(photo_results):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/photography_enhancement.png', dpi=300, bbox_inches='tight')
    plt.close()

def demonstrate_surveillance_enhancement():
    """Demonstrate enhancement for surveillance and security applications."""
    setup_plotting()

    # Use camera image and simulate surveillance conditions
    surveillance = data.camera()

    # Simulate low light conditions
    low_light = (surveillance * 0.3 + 20).astype(np.uint8)

    # Add noise to simulate poor camera conditions
    noisy = low_light + np.random.normal(0, 15, low_light.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # Apply surveillance-specific enhancements
    # 1. Noise reduction
    denoised = cv2.bilateralFilter(noisy, 9, 75, 75)

    # 2. Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)

    # 3. Edge enhancement for feature detection
    laplacian = cv2.Laplacian(contrast_enhanced, cv2.CV_64F)
    edge_enhanced = contrast_enhanced.astype(float) - 0.3 * laplacian
    edge_enhanced = np.clip(edge_enhanced, 0, 255).astype(np.uint8)

    # 4. Histogram stretching
    stretched = exposure.rescale_intensity(edge_enhanced)

    # Create surveillance enhancement comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    surveillance_results = [
        (surveillance, "Original"),
        (low_light, "Low Light"),
        (noisy, "Noisy"),
        (denoised, "Denoised"),
        (contrast_enhanced, "Contrast Enhanced"),
        (edge_enhanced, "Edge Enhanced")
    ]

    for i, (img, title) in enumerate(surveillance_results):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/surveillance_enhancement.png', dpi=300, bbox_inches='tight')
    plt.close()

def demonstrate_before_after_comparison():
    """Create comprehensive before/after comparison."""
    setup_plotting()

    # Use different images for different scenarios
    images = load_sample_images()

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    scenarios = [
        (images['camera'], "Camera", lambda x: cv2.equalizeHist(x)),
        (images['coins'], "Coins", lambda x: cv2.createCLAHE(clipLimit=3.0).apply(x)),
        (images['moon'], "Moon", lambda x: exposure.adjust_gamma(x, 0.6))
    ]

    for i, (img, name, enhance_func) in enumerate(scenarios):
        # Original
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"{name} - Original")
        axes[i, 0].axis('off')

        # Enhanced
        enhanced = enhance_func(img)
        axes[i, 1].imshow(enhanced, cmap='gray')
        axes[i, 1].set_title(f"{name} - Enhanced")
        axes[i, 1].axis('off')

        # Histograms
        axes[i, 2].hist(img.ravel(), bins=50, alpha=0.7, label='Original')
        axes[i, 2].hist(enhanced.ravel(), bins=50, alpha=0.7, label='Enhanced')
        axes[i, 2].set_title(f"{name} - Histograms")
        axes[i, 2].legend()

        # Difference
        diff = cv2.absdiff(img, enhanced)
        axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title(f"{name} - Difference")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/before_after_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def demonstrate_quality_metrics():
    """Demonstrate quality assessment metrics."""
    setup_plotting()

    original = data.camera()

    # Create different enhancement versions
    enhancements = {
        'Histogram Eq': cv2.equalizeHist(original),
        'CLAHE': cv2.createCLAHE(clipLimit=2.0).apply(original),
        'Gamma 0.7': exposure.adjust_gamma(original, 0.7),
        'Contrast Stretch': exposure.rescale_intensity(original)
    }

    # Calculate quality metrics
    metrics = []
    for name, enhanced in enhancements.items():
        # Calculate contrast (standard deviation)
        contrast = np.std(enhanced)

        # Calculate edge strength using Sobel
        sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))

        # Calculate histogram spread (entropy approximation)
        hist = cv2.calcHist([enhanced], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        metrics.append({
            'Method': name,
            'Contrast': contrast,
            'Edge Strength': edge_strength,
            'Entropy': entropy
        })

    # Create metrics comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    methods = [m['Method'] for m in metrics]
    contrasts = [m['Contrast'] for m in metrics]
    edge_strengths = [m['Edge Strength'] for m in metrics]
    entropies = [m['Entropy'] for m in metrics]

    axes[0, 0].bar(methods, contrasts)
    axes[0, 0].set_title('Contrast Comparison')
    axes[0, 0].set_ylabel('Standard Deviation')
    axes[0, 0].tick_params(axis='x', rotation=45)

    axes[0, 1].bar(methods, edge_strengths)
    axes[0, 1].set_title('Edge Strength Comparison')
    axes[0, 1].set_ylabel('Mean Gradient Magnitude')
    axes[0, 1].tick_params(axis='x', rotation=45)

    axes[1, 0].bar(methods, entropies)
    axes[1, 0].set_title('Entropy Comparison')
    axes[1, 0].set_ylabel('Information Content')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Show original and best enhancement
    best_contrast_idx = np.argmax(contrasts)
    best_method = list(enhancements.keys())[best_contrast_idx]
    axes[1, 1].imshow(enhancements[best_method], cmap='gray')
    axes[1, 1].set_title(f'Best Contrast: {best_method}')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/quality_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_real_world_figures():
    """Generate all real-world example figures."""
    print("Generating real-world example figures...")

    # Ensure output directory exists
    os.makedirs('../figures', exist_ok=True)

    # Generate all demonstrations
    demonstrate_medical_imaging_enhancement()
    demonstrate_photography_enhancement()
    demonstrate_surveillance_enhancement()
    demonstrate_before_after_comparison()
    demonstrate_quality_metrics()

    print("✓ Real-world example figures generated successfully")

if __name__ == "__main__":
    generate_real_world_figures()