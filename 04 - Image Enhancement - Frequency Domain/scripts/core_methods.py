#!/usr/bin/env python3
"""
Core Methods for Frequency Domain Image Enhancement
Generates visualizations for fundamental frequency domain concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
from skimage import data, filters
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for high-quality output
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def create_2d_fourier_transform_demo():
    """Demonstrate 2D Fourier Transform concepts."""
    # Load sample image
    image = data.camera().astype(float)

    # Compute FFT
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)

    # Compute magnitude and phase
    magnitude = np.abs(f_shift)
    phase = np.angle(f_shift)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Magnitude spectrum (log scale)
    im1 = axes[0, 1].imshow(np.log(magnitude + 1), cmap='gray')
    axes[0, 1].set_title('Magnitude Spectrum (Log)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Phase spectrum
    im2 = axes[0, 2].imshow(phase, cmap='gray')
    axes[0, 2].set_title('Phase Spectrum')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # 3D magnitude plot
    x = np.arange(-magnitude.shape[1]//2, magnitude.shape[1]//2)
    y = np.arange(-magnitude.shape[0]//2, magnitude.shape[0]//2)
    X, Y = np.meshgrid(x[::8], y[::8])  # Subsample for visualization
    Z = np.log(magnitude[::8, ::8] + 1)

    ax_3d = fig.add_subplot(2, 2, 3, projection='3d')
    ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax_3d.set_title('3D Magnitude Spectrum')
    ax_3d.set_xlabel('u (frequency)')
    ax_3d.set_ylabel('v (frequency)')
    ax_3d.set_zlabel('Log Magnitude')

    # Reconstruction demonstration
    # Zero out high frequencies
    magnitude_filtered = magnitude.copy()
    center_x, center_y = magnitude.shape[0]//2, magnitude.shape[1]//2
    radius = 50

    y_coords, x_coords = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
    mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 > radius**2
    magnitude_filtered[mask] = 0

    # Reconstruct
    f_filtered = magnitude_filtered * np.exp(1j * phase)
    f_filtered_shift = ifftshift(f_filtered)
    reconstructed = np.real(ifft2(f_filtered_shift))

    axes[1, 1].imshow(reconstructed, cmap='gray')
    axes[1, 1].set_title('Low-pass Reconstructed')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/fourier_transform_2d.png', bbox_inches='tight')
    plt.close()

    print("✅ Generated 2D Fourier Transform demonstration")

def create_frequency_representation_demo():
    """Demonstrate frequency domain representation concepts."""
    # Create test patterns
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Test images with different frequency content
    size = 256
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    # Low frequency sinusoid
    low_freq = np.sin(2 * np.pi * 2 * X / size) * np.sin(2 * np.pi * 2 * Y / size)

    # High frequency sinusoid
    high_freq = np.sin(2 * np.pi * 20 * X / size) * np.sin(2 * np.pi * 20 * Y / size)

    # Mixed frequencies
    mixed = low_freq + 0.5 * high_freq

    # Real image
    real_image = data.camera().astype(float)
    real_image = cv2.resize(real_image, (size, size))

    images = [low_freq, high_freq, mixed, real_image]
    titles = ['Low Frequency Pattern', 'High Frequency Pattern', 'Mixed Frequencies', 'Natural Image']

    for i, (img, title) in enumerate(zip(images, titles)):
        # Original pattern
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(title)
        axes[0, i].axis('off')

        # FFT magnitude
        f_transform = fftshift(fft2(img))
        magnitude = np.abs(f_transform)

        axes[1, i].imshow(np.log(magnitude + 1), cmap='hot')
        axes[1, i].set_title(f'Magnitude Spectrum')
        axes[1, i].axis('off')

        # Power spectrum
        power = magnitude**2
        axes[2, i].imshow(np.log(power + 1), cmap='plasma')
        axes[2, i].set_title(f'Power Spectrum')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/frequency_representation.png', bbox_inches='tight')
    plt.close()

    print("✅ Generated frequency representation demonstration")

def create_frequency_quadrants_demo():
    """Demonstrate frequency domain quadrants and interpretation."""
    # Load sample image
    image = data.camera().astype(float)

    # Compute FFT
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    magnitude = np.abs(f_shift)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Full magnitude spectrum
    axes[0, 1].imshow(np.log(magnitude + 1), cmap='gray')
    axes[0, 1].set_title('Full Magnitude Spectrum')
    axes[0, 1].axis('off')

    # Add frequency labels
    center_x, center_y = magnitude.shape[0]//2, magnitude.shape[1]//2
    axes[0, 1].plot(center_y, center_x, 'r+', markersize=10, markeredgewidth=2)
    axes[0, 1].text(center_y + 20, center_x, 'DC (0,0)', color='red', fontsize=12)

    # Draw circles for frequency ranges
    circle_radii = [30, 60, 90]
    circle_colors = ['red', 'yellow', 'green']
    circle_labels = ['Low freq', 'Mid freq', 'High freq']

    for radius, color, label in zip(circle_radii, circle_colors, circle_labels):
        circle = plt.Circle((center_y, center_x), radius, fill=False, color=color, linewidth=2)
        axes[0, 1].add_patch(circle)
        axes[0, 1].text(center_y + radius/np.sqrt(2), center_x - radius/np.sqrt(2),
                       label, color=color, fontsize=10)

    # Radial frequency profile
    y_coords, x_coords = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
    distances = np.sqrt((x_coords - center_y)**2 + (y_coords - center_x)**2)

    # Compute radial average
    max_dist = int(min(center_x, center_y))
    radial_profile = np.zeros(max_dist)

    for r in range(max_dist):
        mask = (distances >= r) & (distances < r + 1)
        if np.any(mask):
            radial_profile[r] = np.mean(magnitude[mask])

    axes[0, 2].plot(radial_profile)
    axes[0, 2].set_title('Radial Frequency Profile')
    axes[0, 2].set_xlabel('Distance from DC')
    axes[0, 2].set_ylabel('Average Magnitude')
    axes[0, 2].grid(True, alpha=0.3)

    # Demonstrate frequency filtering effects
    # Low-pass filter
    low_pass_mask = distances <= 30
    f_lowpass = f_shift.copy()
    f_lowpass[~low_pass_mask] = 0
    img_lowpass = np.real(ifft2(ifftshift(f_lowpass)))

    axes[1, 0].imshow(img_lowpass, cmap='gray')
    axes[1, 0].set_title('Low-pass Filtered (r ≤ 30)')
    axes[1, 0].axis('off')

    # High-pass filter
    high_pass_mask = distances >= 30
    f_highpass = f_shift.copy()
    f_highpass[~high_pass_mask] = 0
    img_highpass = np.real(ifft2(ifftshift(f_highpass)))

    axes[1, 1].imshow(img_highpass, cmap='gray')
    axes[1, 1].set_title('High-pass Filtered (r ≥ 30)')
    axes[1, 1].axis('off')

    # Band-pass filter
    band_pass_mask = (distances >= 20) & (distances <= 60)
    f_bandpass = f_shift.copy()
    f_bandpass[~band_pass_mask] = 0
    img_bandpass = np.real(ifft2(ifftshift(f_bandpass)))

    axes[1, 2].imshow(img_bandpass, cmap='gray')
    axes[1, 2].set_title('Band-pass Filtered (20 ≤ r ≤ 60)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/frequency_quadrants.png', bbox_inches='tight')
    plt.close()

    print("✅ Generated frequency quadrants demonstration")

def create_motivation_demo():
    """Create motivation visualization comparing spatial vs frequency domain."""
    # Create a noisy image with periodic interference
    image = data.camera().astype(float)

    # Add periodic noise
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X, Y = np.meshgrid(x, y)

    # Periodic interference pattern
    noise_pattern = 30 * np.sin(2 * np.pi * 40 * X / image.shape[1]) * \
                   np.sin(2 * np.pi * 30 * Y / image.shape[0])

    # Add random noise
    random_noise = np.random.normal(0, 10, image.shape)

    noisy_image = image + noise_pattern + random_noise
    noisy_image = np.clip(noisy_image, 0, 255)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original and noisy images
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_image, cmap='gray')
    axes[0, 1].set_title('Noisy Image (Periodic + Random)')
    axes[0, 1].axis('off')

    # Frequency domain analysis
    f_clean = fftshift(fft2(image))
    f_noisy = fftshift(fft2(noisy_image))

    # Show magnitude spectra
    mag_clean = np.log(np.abs(f_clean) + 1)
    mag_noisy = np.log(np.abs(f_noisy) + 1)

    axes[1, 0].imshow(mag_clean, cmap='hot')
    axes[1, 0].set_title('Clean Image Spectrum')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mag_noisy, cmap='hot')
    axes[1, 1].set_title('Noisy Image Spectrum')
    axes[1, 1].axis('off')

    # Mark the periodic noise peaks
    center_x, center_y = f_noisy.shape[0]//2, f_noisy.shape[1]//2
    axes[1, 1].plot(center_y + 40, center_x, 'r*', markersize=15, label='Periodic noise')
    axes[1, 1].plot(center_y - 40, center_x, 'r*', markersize=15)
    axes[1, 1].plot(center_y, center_x + 30, 'r*', markersize=15)
    axes[1, 1].plot(center_y, center_x - 30, 'r*', markersize=15)
    axes[1, 1].legend()

    # Demonstrate frequency domain filtering advantage
    # Create notch filter to remove periodic noise
    y_coords, x_coords = np.ogrid[:f_noisy.shape[0], :f_noisy.shape[1]]

    # Notch filter (remove specific frequencies)
    notch_filter = np.ones_like(f_noisy, dtype=float)

    # Remove periodic noise frequencies
    noise_locations = [(center_x, center_y + 40), (center_x, center_y - 40),
                      (center_x + 30, center_y), (center_x - 30, center_y)]

    for loc_x, loc_y in noise_locations:
        mask = (x_coords - loc_y)**2 + (y_coords - loc_x)**2 <= 25
        notch_filter[mask] = 0

    # Apply filter
    f_filtered = f_noisy * notch_filter
    filtered_image = np.real(ifft2(ifftshift(f_filtered)))

    axes[0, 2].imshow(filtered_image, cmap='gray')
    axes[0, 2].set_title('Frequency Domain Filtered')
    axes[0, 2].axis('off')

    # Show the notch filter
    axes[1, 2].imshow(notch_filter, cmap='gray')
    axes[1, 2].set_title('Notch Filter')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/frequency_domain_motivation.png', bbox_inches='tight')
    plt.close()

    print("✅ Generated frequency domain motivation demonstration")

if __name__ == "__main__":
    print("🔧 Generating core frequency domain method figures...")

    create_motivation_demo()
    create_2d_fourier_transform_demo()
    create_frequency_representation_demo()
    create_frequency_quadrants_demo()

    print("\n✅ All core method figures generated successfully!")
    print("📁 Figures saved to ../figures/")