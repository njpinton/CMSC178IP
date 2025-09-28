#!/usr/bin/env python3
"""
Advanced Techniques for Frequency Domain Image Enhancement
Generates visualizations for sophisticated frequency domain methods
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
from skimage import data, restoration, filters
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for high-quality output
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def create_filter_profiles():
    """Create visualization of different filter types and their profiles."""
    # Create frequency distance matrix
    size = 512
    center = size // 2
    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((x - center)**2 + (y - center)**2)

    # Normalize distance
    d_normalized = distance / (size / 2)

    # Filter parameters
    d0 = 0.3  # Cutoff frequency
    n = 2     # Filter order

    # Low-pass filters
    ideal_lp = (d_normalized <= d0).astype(float)
    butterworth_lp = 1 / (1 + (d_normalized / d0)**(2*n))
    gaussian_lp = np.exp(-(d_normalized**2) / (2 * d0**2))

    # High-pass filters
    ideal_hp = (d_normalized > d0).astype(float)
    butterworth_hp = 1 / (1 + (d0 / (d_normalized + 1e-10))**(2*n))
    gaussian_hp = 1 - gaussian_lp

    # Band-pass and band-reject filters
    d1, d2 = 0.2, 0.4
    band_pass = ((d_normalized >= d1) & (d_normalized <= d2)).astype(float)
    band_reject = 1 - band_pass

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Low-pass filters
    filters_lp = [ideal_lp, butterworth_lp, gaussian_lp]
    titles_lp = ['Ideal Low-pass', 'Butterworth Low-pass', 'Gaussian Low-pass']

    for i, (filt, title) in enumerate(zip(filters_lp, titles_lp)):
        axes[0, i].imshow(filt, cmap='gray')
        axes[0, i].set_title(title)
        axes[0, i].axis('off')

    # High-pass filters
    filters_hp = [ideal_hp, butterworth_hp, gaussian_hp]
    titles_hp = ['Ideal High-pass', 'Butterworth High-pass', 'Gaussian High-pass']

    for i, (filt, title) in enumerate(zip(filters_hp, titles_hp)):
        axes[1, i].imshow(filt, cmap='gray')
        axes[1, i].set_title(title)
        axes[1, i].axis('off')

    # Band filters
    axes[2, 0].imshow(band_pass, cmap='gray')
    axes[2, 0].set_title('Band-pass Filter')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(band_reject, cmap='gray')
    axes[2, 1].set_title('Band-reject Filter')
    axes[2, 1].axis('off')

    # 1D profiles
    center_line = size // 2
    x_profile = np.arange(size)
    d_profile = np.abs(x_profile - center) / center

    axes[0, 3].plot(d_profile, ideal_lp[center_line, :], 'b-', linewidth=2, label='Ideal')
    axes[0, 3].plot(d_profile, butterworth_lp[center_line, :], 'r-', linewidth=2, label='Butterworth')
    axes[0, 3].plot(d_profile, gaussian_lp[center_line, :], 'g-', linewidth=2, label='Gaussian')
    axes[0, 3].set_title('Low-pass Filter Profiles')
    axes[0, 3].set_xlabel('Normalized Distance')
    axes[0, 3].set_ylabel('Filter Response')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)

    axes[1, 3].plot(d_profile, ideal_hp[center_line, :], 'b-', linewidth=2, label='Ideal')
    axes[1, 3].plot(d_profile, butterworth_hp[center_line, :], 'r-', linewidth=2, label='Butterworth')
    axes[1, 3].plot(d_profile, gaussian_hp[center_line, :], 'g-', linewidth=2, label='Gaussian')
    axes[1, 3].set_title('High-pass Filter Profiles')
    axes[1, 3].set_xlabel('Normalized Distance')
    axes[1, 3].set_ylabel('Filter Response')
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)

    # Comparison of ringing artifacts
    axes[2, 2].plot(d_profile, ideal_lp[center_line, :], 'b-', linewidth=2, label='Ideal (Ringing)')
    axes[2, 2].plot(d_profile, butterworth_lp[center_line, :], 'r-', linewidth=2, label='Butterworth (Smooth)')
    axes[2, 2].plot(d_profile, gaussian_lp[center_line, :], 'g-', linewidth=2, label='Gaussian (Smoothest)')
    axes[2, 2].set_title('Artifact Comparison')
    axes[2, 2].set_xlabel('Normalized Distance')
    axes[2, 2].set_ylabel('Filter Response')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    # Filter design trade-offs
    axes[2, 3].text(0.1, 0.8, 'Filter Design Trade-offs:', fontsize=12, fontweight='bold', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.7, '• Ideal: Sharp cutoff, ringing artifacts', fontsize=10, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.6, '• Butterworth: Smooth, no ringing', fontsize=10, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.5, '• Gaussian: Smoothest, gradual transition', fontsize=10, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.3, 'Selection Criteria:', fontsize=12, fontweight='bold', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.2, '• Artifact sensitivity: Gaussian > Butterworth > Ideal', fontsize=10, transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.1, '• Computational cost: Ideal < Butterworth < Gaussian', fontsize=10, transform=axes[2, 3].transAxes)
    axes[2, 3].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/low_pass_filters.png', bbox_inches='tight')
    plt.close()

    # Save individual filter types
    # High-pass filters
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

    axes2[0, 0].imshow(ideal_hp, cmap='gray')
    axes2[0, 0].set_title('Ideal High-pass Filter')
    axes2[0, 0].axis('off')

    axes2[0, 1].imshow(butterworth_hp, cmap='gray')
    axes2[0, 1].set_title('Butterworth High-pass Filter')
    axes2[0, 1].axis('off')

    axes2[1, 0].imshow(gaussian_hp, cmap='gray')
    axes2[1, 0].set_title('Gaussian High-pass Filter')
    axes2[1, 0].axis('off')

    # High-frequency emphasis filter
    hfe_filter = 0.5 + 2 * gaussian_hp
    axes2[1, 1].imshow(hfe_filter, cmap='gray')
    axes2[1, 1].set_title('High-frequency Emphasis Filter')
    axes2[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/high_pass_filters.png', bbox_inches='tight')
    plt.close()

    # Band-pass filters
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

    axes3[0, 0].imshow(band_pass, cmap='gray')
    axes3[0, 0].set_title('Band-pass Filter')
    axes3[0, 0].axis('off')

    axes3[0, 1].imshow(band_reject, cmap='gray')
    axes3[0, 1].set_title('Band-reject Filter')
    axes3[0, 1].axis('off')

    # Notch filter example
    notch_filter = np.ones_like(d_normalized)
    # Create notches at specific frequencies
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        notch_x = center + int(100 * np.cos(angle))
        notch_y = center + int(100 * np.sin(angle))
        if 0 <= notch_x < size and 0 <= notch_y < size:
            mask = (x - notch_x)**2 + (y - notch_y)**2 <= 100
            notch_filter[mask] = 0

    axes3[1, 0].imshow(notch_filter, cmap='gray')
    axes3[1, 0].set_title('Notch Filter (Periodic Noise Removal)')
    axes3[1, 0].axis('off')

    # Selective frequency removal
    selective_filter = np.ones_like(d_normalized)
    # Remove horizontal and vertical lines
    selective_filter[center-5:center+5, :] = 0  # Horizontal
    selective_filter[:, center-5:center+5] = 0  # Vertical

    axes3[1, 1].imshow(selective_filter, cmap='gray')
    axes3[1, 1].set_title('Selective Frequency Filter')
    axes3[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/band_pass_filters.png', bbox_inches='tight')
    plt.close()

    print("✅ Generated filter profile demonstrations")

def create_homomorphic_filtering_demo():
    """Demonstrate homomorphic filtering for illumination correction."""
    # Load sample image and create uneven illumination
    image = data.camera().astype(float) / 255.0

    # Create illumination pattern
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2

    # Gaussian illumination fall-off
    illumination = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(image.shape) // 3)**2))
    illumination = 0.3 + 0.7 * illumination  # Scale to reasonable range

    # Apply illumination
    illuminated_image = image * illumination

    # Homomorphic filtering
    # Step 1: Take logarithm
    log_image = np.log(illuminated_image + 1e-10)

    # Step 2: Apply FFT
    f_transform = fftshift(fft2(log_image))

    # Step 3: Design homomorphic filter
    size = image.shape[0]
    center = size // 2
    y_coords, x_coords = np.ogrid[:size, :size]
    distance = np.sqrt((x_coords - center)**2 + (y_coords - center)**2)
    d_normalized = distance / (size / 2)

    # Homomorphic filter parameters
    gamma_l = 0.3  # Low frequency gain (illumination suppression)
    gamma_h = 2.0  # High frequency gain (reflectance enhancement)
    d0 = 0.25
    c = 1.0

    homomorphic_filter = (gamma_h - gamma_l) * (1 - np.exp(-c * (d_normalized / d0)**2)) + gamma_l

    # Step 4: Apply filter
    filtered_transform = f_transform * homomorphic_filter

    # Step 5: Inverse FFT
    filtered_log = np.real(ifft2(ifftshift(filtered_transform)))

    # Step 6: Exponential
    corrected_image = np.exp(filtered_log)

    # Normalize for display
    corrected_image = (corrected_image - corrected_image.min()) / (corrected_image.max() - corrected_image.min())

    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Original and illuminated
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(illumination, cmap='gray')
    axes[0, 1].set_title('Illumination Pattern')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(illuminated_image, cmap='gray')
    axes[0, 2].set_title('Illuminated Image')
    axes[0, 2].axis('off')

    # Logarithmic transform
    axes[1, 0].imshow(log_image, cmap='gray')
    axes[1, 0].set_title('Log Transform')
    axes[1, 0].axis('off')

    # Homomorphic filter
    axes[1, 1].imshow(homomorphic_filter, cmap='gray')
    axes[1, 1].set_title('Homomorphic Filter')
    axes[1, 1].axis('off')

    # Filter profile
    center_line = size // 2
    x_profile = np.arange(size)
    d_profile = np.abs(x_profile - center) / center

    axes[1, 2].plot(d_profile, homomorphic_filter[center_line, :], 'r-', linewidth=2)
    axes[1, 2].axhline(y=gamma_l, color='b', linestyle='--', label=f'γL = {gamma_l}')
    axes[1, 2].axhline(y=gamma_h, color='g', linestyle='--', label=f'γH = {gamma_h}')
    axes[1, 2].set_title('Filter Profile')
    axes[1, 2].set_xlabel('Normalized Distance')
    axes[1, 2].set_ylabel('Filter Gain')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Results
    axes[2, 0].imshow(corrected_image, cmap='gray')
    axes[2, 0].set_title('Corrected Image')
    axes[2, 0].axis('off')

    # Before/after comparison
    axes[2, 1].plot(illuminated_image[center, :], 'b-', linewidth=2, label='Before')
    axes[2, 1].plot(corrected_image[center, :], 'r-', linewidth=2, label='After')
    axes[2, 1].set_title('Intensity Profile Comparison')
    axes[2, 1].set_xlabel('Pixel Position')
    axes[2, 1].set_ylabel('Intensity')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # Improvement metrics
    # Contrast improvement
    contrast_before = np.std(illuminated_image)
    contrast_after = np.std(corrected_image)
    improvement = contrast_after / contrast_before

    axes[2, 2].text(0.1, 0.8, 'Homomorphic Filtering Results:', fontsize=12, fontweight='bold', transform=axes[2, 2].transAxes)
    axes[2, 2].text(0.1, 0.6, f'Contrast improvement: {improvement:.2f}x', fontsize=11, transform=axes[2, 2].transAxes)
    axes[2, 2].text(0.1, 0.5, f'Illumination suppression: {gamma_l:.1f}', fontsize=11, transform=axes[2, 2].transAxes)
    axes[2, 2].text(0.1, 0.4, f'Reflectance enhancement: {gamma_h:.1f}', fontsize=11, transform=axes[2, 2].transAxes)

    axes[2, 2].text(0.1, 0.2, 'Benefits:', fontsize=12, fontweight='bold', transform=axes[2, 2].transAxes)
    axes[2, 2].text(0.1, 0.1, '• Separates illumination/reflectance', fontsize=10, transform=axes[2, 2].transAxes)
    axes[2, 2].text(0.1, 0.05, '• Enhances contrast uniformly', fontsize=10, transform=axes[2, 2].transAxes)
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/homomorphic_filtering.png', bbox_inches='tight')
    plt.close()

    print("✅ Generated homomorphic filtering demonstration")

def create_wiener_filtering_demo():
    """Demonstrate Wiener filtering for image restoration."""
    # Load sample image
    original = data.camera().astype(float) / 255.0

    # Create motion blur kernel
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1 / kernel_size  # Horizontal motion blur

    # Apply blur and noise
    blurred = cv2.filter2D(original, -1, kernel)
    noise = np.random.normal(0, 0.02, original.shape)
    degraded = blurred + noise
    degraded = np.clip(degraded, 0, 1)

    # Wiener filtering
    # Get degradation function in frequency domain
    H = fft2(kernel, s=original.shape)
    H_shifted = fftshift(H)

    # Image and noise power spectra (estimated)
    F_original = fftshift(fft2(original))
    F_degraded = fftshift(fft2(degraded))

    # Estimate noise-to-signal ratio
    noise_power = np.var(noise)
    signal_power = np.var(original)
    K = noise_power / signal_power

    # Wiener filter
    H_conj = np.conj(H_shifted)
    H_mag_sq = np.abs(H_shifted)**2
    wiener_filter = H_conj / (H_mag_sq + K)

    # Apply Wiener filter
    F_restored = F_degraded * wiener_filter
    restored = np.real(ifft2(ifftshift(F_restored)))
    restored = np.clip(restored, 0, 1)

    # Compare with simple inverse filtering
    # Inverse filter (with regularization to avoid division by zero)
    epsilon = 0.01
    inverse_filter = 1 / (H_shifted + epsilon)
    F_inverse = F_degraded * inverse_filter
    inverse_restored = np.real(ifft2(ifftshift(F_inverse)))
    inverse_restored = np.clip(inverse_restored, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Original images
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(degraded, cmap='gray')
    axes[0, 1].set_title('Degraded Image (Blur + Noise)')
    axes[0, 1].axis('off')

    # Motion blur kernel
    kernel_display = np.zeros((50, 50))
    start = (50 - kernel_size) // 2
    kernel_display[start:start+kernel_size, start:start+kernel_size] = kernel * 255
    axes[0, 2].imshow(kernel_display, cmap='gray')
    axes[0, 2].set_title('Motion Blur Kernel')
    axes[0, 2].axis('off')

    # Frequency domain representations
    axes[1, 0].imshow(np.log(np.abs(H_shifted) + 1), cmap='gray')
    axes[1, 0].set_title('Degradation Function |H(u,v)|')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.abs(wiener_filter), cmap='gray')
    axes[1, 1].set_title('Wiener Filter')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.abs(inverse_filter), cmap='gray')
    axes[1, 2].set_title('Inverse Filter')
    axes[1, 2].axis('off')

    # Restoration results
    axes[2, 0].imshow(restored, cmap='gray')
    axes[2, 0].set_title('Wiener Restoration')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(inverse_restored, cmap='gray')
    axes[2, 1].set_title('Inverse Filter Restoration')
    axes[2, 1].axis('off')

    # Quality comparison
    # Calculate PSNR
    mse_wiener = np.mean((original - restored)**2)
    mse_inverse = np.mean((original - inverse_restored)**2)
    mse_degraded = np.mean((original - degraded)**2)

    psnr_wiener = 20 * np.log10(1.0 / np.sqrt(mse_wiener))
    psnr_inverse = 20 * np.log10(1.0 / np.sqrt(mse_inverse))
    psnr_degraded = 20 * np.log10(1.0 / np.sqrt(mse_degraded))

    axes[2, 2].text(0.1, 0.8, 'Restoration Quality (PSNR):', fontsize=12, fontweight='bold', transform=axes[2, 2].transAxes)
    axes[2, 2].text(0.1, 0.6, f'Degraded: {psnr_degraded:.1f} dB', fontsize=11, transform=axes[2, 2].transAxes)
    axes[2, 2].text(0.1, 0.5, f'Wiener: {psnr_wiener:.1f} dB', fontsize=11, color='green', transform=axes[2, 2].transAxes)
    axes[2, 2].text(0.1, 0.4, f'Inverse: {psnr_inverse:.1f} dB', fontsize=11, color='red', transform=axes[2, 2].transAxes)

    axes[2, 2].text(0.1, 0.2, 'Wiener Filter Advantages:', fontsize=12, fontweight='bold', transform=axes[2, 2].transAxes)
    axes[2, 2].text(0.1, 0.1, '• Optimal MSE minimization', fontsize=10, transform=axes[2, 2].transAxes)
    axes[2, 2].text(0.1, 0.05, '• Noise regularization', fontsize=10, transform=axes[2, 2].transAxes)
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/wiener_filtering.png', bbox_inches='tight')
    plt.close()

    print("✅ Generated Wiener filtering demonstration")

def create_fft_algorithm_demo():
    """Demonstrate FFT algorithm and complexity."""
    # FFT complexity comparison
    sizes = [2**i for i in range(4, 11)]  # Powers of 2 from 16 to 1024
    dft_complexity = [n**2 for n in sizes]
    fft_complexity = [n * np.log2(n) for n in sizes]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Complexity comparison
    axes[0, 0].loglog(sizes, dft_complexity, 'ro-', linewidth=2, markersize=8, label='DFT O(N²)')
    axes[0, 0].loglog(sizes, fft_complexity, 'bo-', linewidth=2, markersize=8, label='FFT O(N log N)')
    axes[0, 0].set_xlabel('Image Size (N)')
    axes[0, 0].set_ylabel('Operations')
    axes[0, 0].set_title('Algorithm Complexity Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Speedup factor
    speedup = np.array(dft_complexity) / np.array(fft_complexity)
    axes[0, 1].semilogx(sizes, speedup, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Image Size (N)')
    axes[0, 1].set_ylabel('Speedup Factor')
    axes[0, 1].set_title('FFT Speedup over DFT')
    axes[0, 1].grid(True, alpha=0.3)

    # FFT butterfly diagram (simplified)
    # 8-point FFT example
    n = 8
    stages = int(np.log2(n))

    axes[1, 0].text(0.5, 0.9, '8-Point FFT Butterfly Diagram', ha='center', fontsize=12, fontweight='bold', transform=axes[1, 0].transAxes)

    # Draw input points
    y_positions = np.linspace(0.1, 0.8, n)
    x_input = 0.1
    x_stages = np.linspace(0.3, 0.9, stages)

    # Input labels
    input_order = [0, 4, 2, 6, 1, 5, 3, 7]  # Bit-reversed order
    for i, (y, idx) in enumerate(zip(y_positions, input_order)):
        axes[1, 0].text(x_input, y, f'x[{idx}]', ha='center', va='center', transform=axes[1, 0].transAxes)

    # Draw connections (simplified)
    for stage in range(stages):
        x = x_stages[stage]
        group_size = 2**(stage + 1)
        num_groups = n // group_size

        for group in range(num_groups):
            start_idx = group * group_size
            mid_idx = start_idx + group_size // 2

            for i in range(group_size // 2):
                y1 = y_positions[start_idx + i]
                y2 = y_positions[mid_idx + i]

                # Draw butterfly connections
                axes[1, 0].plot([x - 0.02, x + 0.02], [y1, y1], 'k-', transform=axes[1, 0].transAxes)
                axes[1, 0].plot([x - 0.02, x + 0.02], [y2, y2], 'k-', transform=axes[1, 0].transAxes)
                axes[1, 0].plot([x - 0.02, x - 0.02], [y1, y2], 'k-', transform=axes[1, 0].transAxes)

    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')

    # 2D FFT process
    # Create sample 2D signal
    size = 64
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    # Create test pattern
    signal_2d = np.sin(2 * np.pi * 5 * X / size) * np.cos(2 * np.pi * 3 * Y / size)

    # Show 2D FFT process steps
    axes[1, 1].text(0.5, 0.9, '2D FFT Process', ha='center', fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, '1. Row-wise 1D FFT', fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, '2. Column-wise 1D FFT', fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, '3. Result: 2D frequency spectrum', fontsize=10, transform=axes[1, 1].transAxes)

    axes[1, 1].text(0.1, 0.3, 'Advantages:', fontsize=10, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.2, '• Separable: O(N² log N) vs O(N⁴)', fontsize=9, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.15, '• Memory efficient', fontsize=9, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.1, '• Parallelizable', fontsize=9, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.05, '• Cache-friendly', fontsize=9, transform=axes[1, 1].transAxes)

    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/fft_algorithm.png', bbox_inches='tight')
    plt.close()

    print("✅ Generated FFT algorithm demonstration")

if __name__ == "__main__":
    print("🔧 Generating advanced frequency domain technique figures...")

    create_filter_profiles()
    create_homomorphic_filtering_demo()
    create_wiener_filtering_demo()
    create_fft_algorithm_demo()

    print("\n✅ All advanced technique figures generated successfully!")
    print("📁 Figures saved to ../figures/")