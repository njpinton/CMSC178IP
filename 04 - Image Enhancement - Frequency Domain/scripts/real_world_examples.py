#!/usr/bin/env python3
"""
Real World Examples for Frequency Domain Image Enhancement
Generates visualizations for practical applications and case studies
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
from skimage import data, restoration, filters, measure
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for high-quality output
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def create_periodic_noise_removal_demo():
    """Demonstrate removal of periodic noise using frequency domain techniques."""
    # Load sample image
    image = data.camera().astype(float) / 255.0

    # Add various types of periodic noise
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]

    # Sinusoidal interference patterns
    noise1 = 0.3 * np.sin(2 * np.pi * 40 * x / image.shape[1])  # Vertical stripes
    noise2 = 0.2 * np.sin(2 * np.pi * 30 * y / image.shape[0])  # Horizontal stripes
    noise3 = 0.15 * np.sin(2 * np.pi * 25 * (x + y) / (image.shape[1] + image.shape[0]))  # Diagonal pattern

    # Combine all noise
    total_noise = noise1 + noise2 + noise3
    noisy_image = image + total_noise
    noisy_image = np.clip(noisy_image, 0, 1)

    # Frequency domain analysis
    F_clean = fftshift(fft2(image))
    F_noisy = fftshift(fft2(noisy_image))

    # Design notch filter to remove periodic components
    # Identify noise peaks in frequency domain
    magnitude_noisy = np.abs(F_noisy)
    magnitude_clean = np.abs(F_clean)

    # Find peaks (simplified approach)
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2

    # Create notch filter
    notch_filter = np.ones_like(F_noisy, dtype=float)

    # Define notch locations based on the periodic noise frequencies
    notch_locations = [
        (center_y, center_x + 40),   # Vertical stripes
        (center_y, center_x - 40),
        (center_y + 30, center_x),   # Horizontal stripes
        (center_y - 30, center_x),
        (center_y + 18, center_x + 18),  # Diagonal pattern
        (center_y - 18, center_x - 18)
    ]

    notch_radius = 8
    for loc_y, loc_x in notch_locations:
        y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = (x_coords - loc_x)**2 + (y_coords - loc_y)**2 <= notch_radius**2
        notch_filter[mask] = 0

    # Apply Gaussian smoothing to notch filter to reduce ringing
    notch_filter = gaussian_filter(notch_filter, sigma=2)

    # Apply notch filter
    F_filtered = F_noisy * notch_filter
    filtered_image = np.real(ifft2(ifftshift(F_filtered)))
    filtered_image = np.clip(filtered_image, 0, 1)

    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Original and noisy images
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(total_noise, cmap='gray')
    axes[0, 1].set_title('Periodic Noise Pattern')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(noisy_image, cmap='gray')
    axes[0, 2].set_title('Noisy Image')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(filtered_image, cmap='gray')
    axes[0, 3].set_title('Filtered Image')
    axes[0, 3].axis('off')

    # Frequency domain analysis
    axes[1, 0].imshow(np.log(magnitude_clean + 1), cmap='hot')
    axes[1, 0].set_title('Clean Spectrum')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.log(magnitude_noisy + 1), cmap='hot')
    axes[1, 1].set_title('Noisy Spectrum')
    axes[1, 1].axis('off')

    # Mark noise peaks
    for loc_y, loc_x in notch_locations:
        axes[1, 1].plot(loc_x, loc_y, 'r*', markersize=8)

    axes[1, 2].imshow(notch_filter, cmap='gray')
    axes[1, 2].set_title('Notch Filter')
    axes[1, 2].axis('off')

    # Filtered spectrum
    magnitude_filtered = np.abs(F_filtered)
    axes[1, 3].imshow(np.log(magnitude_filtered + 1), cmap='hot')
    axes[1, 3].set_title('Filtered Spectrum')
    axes[1, 3].axis('off')

    # Quantitative analysis
    # Profile analysis
    center_row = image.shape[0] // 2
    profile_original = image[center_row, :]
    profile_noisy = noisy_image[center_row, :]
    profile_filtered = filtered_image[center_row, :]

    axes[2, 0].plot(profile_original, 'b-', linewidth=2, label='Original')
    axes[2, 0].plot(profile_noisy, 'r-', linewidth=1, label='Noisy')
    axes[2, 0].plot(profile_filtered, 'g-', linewidth=2, label='Filtered')
    axes[2, 0].set_title('Intensity Profile Comparison')
    axes[2, 0].set_xlabel('Pixel Position')
    axes[2, 0].set_ylabel('Intensity')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Error analysis
    error_noisy = np.abs(image - noisy_image)
    error_filtered = np.abs(image - filtered_image)

    axes[2, 1].imshow(error_noisy, cmap='hot')
    axes[2, 1].set_title('Error: Noisy vs Original')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(error_filtered, cmap='hot')
    axes[2, 2].set_title('Error: Filtered vs Original')
    axes[2, 2].axis('off')

    # Quality metrics
    mse_noisy = np.mean(error_noisy**2)
    mse_filtered = np.mean(error_filtered**2)
    psnr_noisy = 20 * np.log10(1.0 / np.sqrt(mse_noisy))
    psnr_filtered = 20 * np.log10(1.0 / np.sqrt(mse_filtered))

    improvement = psnr_filtered - psnr_noisy

    axes[2, 3].text(0.1, 0.8, 'Quality Metrics:', fontsize=12, fontweight='bold', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.6, f'PSNR (Noisy): {psnr_noisy:.1f} dB', fontsize=11, color='red', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.5, f'PSNR (Filtered): {psnr_filtered:.1f} dB', fontsize=11, color='green', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.4, f'Improvement: {improvement:.1f} dB', fontsize=11, fontweight='bold', transform=axes[2, 3].transAxes)

    axes[2, 3].text(0.1, 0.2, 'Notch Filter Benefits:', fontsize=10, fontweight='bold', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.1, '• Precise frequency targeting', fontsize=9, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.05, '• Preserves image content', fontsize=9, transform=axes[2, 3].transAxes)
    axes[2, 3].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/periodic_noise_removal.png', bbox_inches='tight')
    plt.close()

    print("✅ Generated periodic noise removal demonstration")

def create_medical_imaging_demo():
    """Demonstrate frequency domain enhancement for medical imaging."""
    # Simulate medical image (X-ray like)
    image = data.camera().astype(float) / 255.0

    # Simulate medical imaging artifacts
    # 1. Low frequency shading (illumination variation)
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
    shading = 1 - 0.3 * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(image.shape) // 2)**2))

    # 2. High frequency noise (electronic noise)
    noise = np.random.normal(0, 0.05, image.shape)

    # 3. Periodic interference (power line, equipment)
    interference = 0.03 * np.sin(2 * np.pi * 60 * x / image.shape[1])

    # Combine all artifacts
    medical_image = image * shading + noise + interference
    medical_image = np.clip(medical_image, 0, 1)

    # Design enhancement pipeline
    # Step 1: Remove periodic interference
    F_medical = fftshift(fft2(medical_image))

    # Notch filter for periodic noise
    notch_filter = np.ones_like(F_medical, dtype=float)
    center_y, center_x = medical_image.shape[0] // 2, medical_image.shape[1] // 2

    # Remove power line interference
    power_line_locations = [(center_y, center_x + 60), (center_y, center_x - 60)]
    for loc_y, loc_x in power_line_locations:
        y_coords, x_coords = np.ogrid[:medical_image.shape[0], :medical_image.shape[1]]
        mask = (x_coords - loc_x)**2 + (y_coords - loc_y)**2 <= 16
        notch_filter[mask] = 0

    notch_filter = gaussian_filter(notch_filter, sigma=1)

    # Step 2: High-frequency emphasis for edge enhancement
    y_coords, x_coords = np.ogrid[:medical_image.shape[0], :medical_image.shape[1]]
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    d_normalized = distance / (min(medical_image.shape) / 2)

    # High-frequency emphasis filter
    gamma_l = 0.8  # Low frequency suppression
    gamma_h = 1.5  # High frequency enhancement
    d0 = 0.3
    hfe_filter = (gamma_h - gamma_l) * (1 - np.exp(-(d_normalized / d0)**2)) + gamma_l

    # Step 3: Combine filters
    combined_filter = notch_filter * hfe_filter

    # Apply enhancement
    F_enhanced = F_medical * combined_filter
    enhanced_image = np.real(ifft2(ifftshift(F_enhanced)))

    # Step 4: Contrast adjustment (simulating CLAHE in spatial domain)
    enhanced_image = np.clip(enhanced_image, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Original progression
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Medical Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(shading, cmap='gray')
    axes[0, 1].set_title('Illumination Variation')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(medical_image, cmap='gray')
    axes[0, 2].set_title('Degraded Medical Image')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(enhanced_image, cmap='gray')
    axes[0, 3].set_title('Enhanced Medical Image')
    axes[0, 3].axis('off')

    # Frequency domain analysis
    magnitude_original = np.abs(fftshift(fft2(image)))
    magnitude_degraded = np.abs(F_medical)
    magnitude_enhanced = np.abs(F_enhanced)

    axes[1, 0].imshow(np.log(magnitude_original + 1), cmap='hot')
    axes[1, 0].set_title('Original Spectrum')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.log(magnitude_degraded + 1), cmap='hot')
    axes[1, 1].set_title('Degraded Spectrum')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(combined_filter, cmap='gray')
    axes[1, 2].set_title('Combined Filter')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(np.log(magnitude_enhanced + 1), cmap='hot')
    axes[1, 3].set_title('Enhanced Spectrum')
    axes[1, 3].axis('off')

    # Detail enhancement analysis
    # Edge strength analysis using Sobel operator
    def calculate_edge_strength(img):
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobel_x**2 + sobel_y**2)

    edges_original = calculate_edge_strength(image)
    edges_degraded = calculate_edge_strength(medical_image)
    edges_enhanced = calculate_edge_strength(enhanced_image)

    axes[2, 0].imshow(edges_original, cmap='gray')
    axes[2, 0].set_title('Original Edges')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(edges_degraded, cmap='gray')
    axes[2, 1].set_title('Degraded Edges')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(edges_enhanced, cmap='gray')
    axes[2, 2].set_title('Enhanced Edges')
    axes[2, 2].axis('off')

    # Quantitative metrics
    edge_strength_original = np.mean(edges_original)
    edge_strength_degraded = np.mean(edges_degraded)
    edge_strength_enhanced = np.mean(edges_enhanced)

    contrast_original = np.std(image)
    contrast_degraded = np.std(medical_image)
    contrast_enhanced = np.std(enhanced_image)

    edge_improvement = edge_strength_enhanced / edge_strength_degraded
    contrast_improvement = contrast_enhanced / contrast_degraded

    axes[2, 3].text(0.1, 0.8, 'Enhancement Metrics:', fontsize=12, fontweight='bold', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.6, f'Edge enhancement: {edge_improvement:.2f}x', fontsize=11, color='green', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.5, f'Contrast improvement: {contrast_improvement:.2f}x', fontsize=11, color='blue', transform=axes[2, 3].transAxes)

    axes[2, 3].text(0.1, 0.3, 'Medical Imaging Benefits:', fontsize=10, fontweight='bold', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.2, '• Enhanced diagnostic features', fontsize=9, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.15, '• Reduced noise artifacts', fontsize=9, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.1, '• Improved edge definition', fontsize=9, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.05, '• Quantitative enhancement', fontsize=9, transform=axes[2, 3].transAxes)

    axes[2, 3].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/medical_frequency_enhancement.png', bbox_inches='tight')
    plt.close()

    print("✅ Generated medical imaging enhancement demonstration")

def create_satellite_processing_demo():
    """Demonstrate frequency domain processing for satellite/remote sensing images."""
    # Load sample image to simulate satellite data
    image = data.astronaut()
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.astype(float) / 255.0

    # Simulate atmospheric and sensor effects
    # 1. Atmospheric haze (low frequency component)
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    haze_pattern = 0.3 * (1 + 0.5 * np.sin(2 * np.pi * x / (image.shape[1] * 0.3)) *
                         np.sin(2 * np.pi * y / (image.shape[0] * 0.4)))

    # 2. Sensor striping (periodic noise)
    striping = 0.1 * np.sin(2 * np.pi * 15 * y / image.shape[0])

    # 3. Random noise
    sensor_noise = np.random.normal(0, 0.02, image.shape)

    # Apply degradations
    degraded_image = image + haze_pattern + striping + sensor_noise
    degraded_image = np.clip(degraded_image, 0, 1)

    # Multi-step frequency domain processing
    F_degraded = fftshift(fft2(degraded_image))

    # Step 1: Remove sensor striping with notch filter
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
    notch_filter = np.ones_like(F_degraded, dtype=float)

    # Remove horizontal striping
    stripe_locations = [(center_y + 15, center_x), (center_y - 15, center_x)]
    for loc_y, loc_x in stripe_locations:
        y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = (x_coords - loc_x)**2 + (y_coords - loc_y)**2 <= 36
        notch_filter[mask] = 0

    notch_filter = gaussian_filter(notch_filter, sigma=1)

    # Step 2: Atmospheric correction using high-pass filtering
    y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    d_normalized = distance / (min(image.shape) / 2)

    # High-pass filter for atmospheric haze removal
    d0 = 0.15
    atmospheric_filter = 1 - np.exp(-(d_normalized / d0)**2)

    # Step 3: Combine with edge enhancement
    gamma_l = 0.7
    gamma_h = 1.3
    d0_enhance = 0.25
    enhancement_filter = (gamma_h - gamma_l) * (1 - np.exp(-(d_normalized / d0_enhance)**2)) + gamma_l

    # Apply combined processing
    F_processed1 = F_degraded * notch_filter  # Remove striping
    processed1 = np.real(ifft2(ifftshift(F_processed1)))

    F_processed2 = F_processed1 * atmospheric_filter  # Atmospheric correction
    processed2 = np.real(ifft2(ifftshift(F_processed2)))

    F_final = F_processed2 * enhancement_filter  # Enhancement
    final_processed = np.real(ifft2(ifftshift(F_final)))
    final_processed = np.clip(final_processed, 0, 1)

    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Processing pipeline
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Satellite Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(degraded_image, cmap='gray')
    axes[0, 1].set_title('Degraded (Haze + Striping + Noise)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(processed1, cmap='gray')
    axes[0, 2].set_title('Step 1: Destriping')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(processed2, cmap='gray')
    axes[0, 3].set_title('Step 2: Atmospheric Correction')
    axes[0, 3].axis('off')

    # Filter design
    axes[1, 0].imshow(notch_filter, cmap='gray')
    axes[1, 0].set_title('Notch Filter (Destriping)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(atmospheric_filter, cmap='gray')
    axes[1, 1].set_title('High-pass Filter (Atmospheric)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(enhancement_filter, cmap='gray')
    axes[1, 2].set_title('Enhancement Filter')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(final_processed, cmap='gray')
    axes[1, 3].set_title('Final Enhanced Image')
    axes[1, 3].axis('off')

    # Spectral analysis
    magnitude_original = np.abs(fftshift(fft2(image)))
    magnitude_degraded = np.abs(F_degraded)
    magnitude_final = np.abs(F_final)

    axes[2, 0].imshow(np.log(magnitude_original + 1), cmap='hot')
    axes[2, 0].set_title('Original Spectrum')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(np.log(magnitude_degraded + 1), cmap='hot')
    axes[2, 1].set_title('Degraded Spectrum')
    axes[2, 1].axis('off')

    # Mark the striping artifacts
    for loc_y, loc_x in stripe_locations:
        axes[2, 1].plot(loc_x, loc_y, 'r*', markersize=10)

    axes[2, 2].imshow(np.log(magnitude_final + 1), cmap='hot')
    axes[2, 2].set_title('Enhanced Spectrum')
    axes[2, 2].axis('off')

    # Quantitative assessment
    # Calculate various quality metrics
    def calculate_sharpness(img):
        """Calculate image sharpness using Laplacian variance."""
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        return np.var(laplacian)

    def calculate_contrast(img):
        """Calculate RMS contrast."""
        return np.std(img)

    sharpness_original = calculate_sharpness(image)
    sharpness_degraded = calculate_sharpness(degraded_image)
    sharpness_enhanced = calculate_sharpness(final_processed)

    contrast_original = calculate_contrast(image)
    contrast_degraded = calculate_contrast(degraded_image)
    contrast_enhanced = calculate_contrast(final_processed)

    sharpness_improvement = sharpness_enhanced / sharpness_degraded
    contrast_improvement = contrast_enhanced / contrast_degraded

    axes[2, 3].text(0.1, 0.8, 'Processing Results:', fontsize=12, fontweight='bold', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.6, f'Sharpness improvement: {sharpness_improvement:.2f}x', fontsize=11, color='green', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.5, f'Contrast improvement: {contrast_improvement:.2f}x', fontsize=11, color='blue', transform=axes[2, 3].transAxes)

    axes[2, 3].text(0.1, 0.3, 'Remote Sensing Benefits:', fontsize=10, fontweight='bold', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.2, '• Atmospheric correction', fontsize=9, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.15, '• Sensor artifact removal', fontsize=9, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.1, '• Feature enhancement', fontsize=9, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.05, '• Quantitative analysis ready', fontsize=9, transform=axes[2, 3].transAxes)

    axes[2, 3].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/satellite_frequency_processing.png', bbox_inches='tight')
    plt.close()

    print("✅ Generated satellite processing demonstration")

def create_frequency_sharpening_demo():
    """Demonstrate frequency domain sharpening techniques."""
    # Load sample image
    image = data.camera().astype(float) / 255.0

    # Create blurred version to demonstrate sharpening
    blurred = gaussian_filter(image, sigma=2.0)

    # Design various frequency domain sharpening filters
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
    y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    d_normalized = distance / (min(image.shape) / 2)

    # 1. High-pass filter
    d0 = 0.3
    high_pass = 1 - np.exp(-(d_normalized / d0)**2)

    # 2. High-frequency emphasis
    gamma_l = 0.5
    gamma_h = 2.0
    hfe_filter = (gamma_h - gamma_l) * high_pass + gamma_l

    # 3. Unsharp masking in frequency domain
    unsharp_k = 1.5
    unsharp_filter = 1 + unsharp_k * high_pass

    # 4. Butterworth high-pass
    n = 2
    butterworth_hp = 1 / (1 + (d0 / (d_normalized + 1e-10))**(2*n))

    # Apply filters
    F_blurred = fftshift(fft2(blurred))

    # High-pass sharpening
    F_hp = F_blurred * high_pass
    sharpened_hp = np.real(ifft2(ifftshift(F_hp)))

    # High-frequency emphasis
    F_hfe = F_blurred * hfe_filter
    sharpened_hfe = np.real(ifft2(ifftshift(F_hfe)))

    # Unsharp masking
    F_unsharp = F_blurred * unsharp_filter
    sharpened_unsharp = np.real(ifft2(ifftshift(F_unsharp)))

    # Butterworth sharpening
    F_butterworth = F_blurred * butterworth_hp
    sharpened_butterworth = np.real(ifft2(ifftshift(F_butterworth)))

    # Normalize results
    results = [sharpened_hp, sharpened_hfe, sharpened_unsharp, sharpened_butterworth]
    for i, result in enumerate(results):
        results[i] = np.clip(result, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Original images
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Sharp Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title('Blurred Image')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(results[1], cmap='gray')  # HFE looks best typically
    axes[0, 2].set_title('High-Frequency Emphasis')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(results[2], cmap='gray')  # Unsharp masking
    axes[0, 3].set_title('Unsharp Masking')
    axes[0, 3].axis('off')

    # Filter profiles
    axes[1, 0].imshow(high_pass, cmap='gray')
    axes[1, 0].set_title('High-pass Filter')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(hfe_filter, cmap='gray')
    axes[1, 1].set_title('High-Frequency Emphasis')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(unsharp_filter, cmap='gray')
    axes[1, 2].set_title('Unsharp Masking Filter')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(butterworth_hp, cmap='gray')
    axes[1, 3].set_title('Butterworth High-pass')
    axes[1, 3].axis('off')

    # 1D filter profiles
    center_line = image.shape[0] // 2
    x_profile = np.arange(image.shape[1])
    d_profile = np.abs(x_profile - center_x) / center_x

    axes[2, 0].plot(d_profile, high_pass[center_line, :], 'b-', linewidth=2, label='High-pass')
    axes[2, 0].plot(d_profile, hfe_filter[center_line, :], 'r-', linewidth=2, label='HFE')
    axes[2, 0].plot(d_profile, unsharp_filter[center_line, :], 'g-', linewidth=2, label='Unsharp')
    axes[2, 0].set_title('Filter Response Profiles')
    axes[2, 0].set_xlabel('Normalized Distance')
    axes[2, 0].set_ylabel('Filter Gain')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Sharpness comparison
    def calculate_edge_strength(img):
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.mean(np.sqrt(sobel_x**2 + sobel_y**2))

    edge_original = calculate_edge_strength(image)
    edge_blurred = calculate_edge_strength(blurred)
    edge_results = [calculate_edge_strength(result) for result in results]

    methods = ['High-pass', 'HFE', 'Unsharp', 'Butterworth']
    improvements = [edge / edge_blurred for edge in edge_results]

    axes[2, 1].bar(methods, improvements, color=['blue', 'red', 'green', 'orange'])
    axes[2, 1].axhline(y=edge_original/edge_blurred, color='black', linestyle='--', label='Original level')
    axes[2, 1].set_title('Sharpening Performance')
    axes[2, 1].set_ylabel('Edge Strength Improvement')
    axes[2, 1].legend()
    axes[2, 1].tick_params(axis='x', rotation=45)

    # Detail comparison
    # Extract a region of interest for detailed comparison
    roi_slice = slice(100, 200), slice(150, 250)
    roi_original = image[roi_slice]
    roi_blurred = blurred[roi_slice]
    roi_enhanced = results[1][roi_slice]  # HFE result

    axes[2, 2].plot(roi_original[50, :], 'b-', linewidth=2, label='Original')
    axes[2, 2].plot(roi_blurred[50, :], 'r-', linewidth=1, label='Blurred')
    axes[2, 2].plot(roi_enhanced[50, :], 'g-', linewidth=2, label='Enhanced')
    axes[2, 2].set_title('Detail Enhancement Profile')
    axes[2, 2].set_xlabel('Pixel Position')
    axes[2, 2].set_ylabel('Intensity')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    # Summary
    axes[2, 3].text(0.1, 0.8, 'Sharpening Methods:', fontsize=12, fontweight='bold', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.6, '• High-pass: Removes low frequencies', fontsize=10, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.5, '• HFE: Emphasizes high frequencies', fontsize=10, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.4, '• Unsharp: Adds high-frequency content', fontsize=10, transform=axes[2, 3].transAxes)

    axes[2, 3].text(0.1, 0.2, 'Best Practices:', fontsize=10, fontweight='bold', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.1, '• HFE preserves overall brightness', fontsize=9, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.05, '• Avoid over-sharpening artifacts', fontsize=9, transform=axes[2, 3].transAxes)

    axes[2, 3].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/frequency_sharpening.png', bbox_inches='tight')
    plt.close()

    print("✅ Generated frequency domain sharpening demonstration")

if __name__ == "__main__":
    print("🔧 Generating real-world frequency domain application figures...")

    create_periodic_noise_removal_demo()
    create_medical_imaging_demo()
    create_satellite_processing_demo()
    create_frequency_sharpening_demo()

    print("\n✅ All real-world application figures generated successfully!")
    print("📁 Figures saved to ../figures/")